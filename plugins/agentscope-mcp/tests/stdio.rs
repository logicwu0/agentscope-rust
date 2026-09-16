use agentscope::*;
use agentscope_mcp::{McpClient, StdioConfig};
use agentscope_offload_file::FileOffloadStore;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Duration};

fn config(mode: &str) -> StdioConfig {
    let mut config = StdioConfig::new("/usr/bin/python3");
    config.args = vec![
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/server.py")
            .into_os_string(),
        mode.into(),
    ];
    config.timeout = Duration::from_secs(2);
    config.shutdown_timeout = Duration::from_millis(200);
    config
}

async fn invoke(client: &McpClient, text: &str) -> ToolResult<ToolResultOutput> {
    client
        .tools("test")
        .await?
        .remove(0)
        .execute(json!({"text":text}), ToolContext::new())
        .await
}

#[tokio::test]
async fn handshake_pagination_ping_and_explicit_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");
    let mut config = config("pages");
    config.args.push(path.clone().into_os_string());
    let client = McpClient::connect(config).await.unwrap();
    assert_eq!(client.protocol_version(), "2025-11-25");
    let tools = client.tools("test").await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].definition().name, "test__echo");
    assert_eq!(tools[1].remote_name(), "second");
    assert_eq!(
        tools[0]
            .execute(json!({"text":"hello"}), ToolContext::new())
            .await
            .unwrap(),
        ToolResultOutput::Text("hello".into())
    );
    client.close().await.unwrap();
    client.close().await.unwrap();
    assert!(
        tools[0]
            .execute(json!({"text":"closed"}), ToolContext::new())
            .await
            .is_err()
    );
    let audit = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Value> = audit
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(lines[0]["method"], "initialize");
    assert_eq!(lines[1]["method"], "notifications/initialized");
    assert!(lines.iter().any(|v| v["event"] == "stdin_closed"));
    assert!(
        lines
            .iter()
            .any(|v| v["id"] == "server-ping" && v["result"].is_object())
    );
    assert!(
        lines
            .iter()
            .any(|v| v["id"] == "unsupported" && v["error"]["code"] == -32601)
    );
}

#[tokio::test]
async fn rejects_invalid_handshakes_and_lists() {
    for mode in ["bad_version", "no_tools"] {
        assert!(McpClient::connect(config(mode)).await.is_err());
    }
    let older = McpClient::connect(config("older")).await.unwrap();
    assert_eq!(older.protocol_version(), "2025-06-18");
    older.close().await.unwrap();
    for mode in ["bad_schema", "bad_name", "duplicate", "cursor_loop"] {
        let client = McpClient::connect(config(mode)).await.unwrap();
        assert!(client.tools("test").await.is_err(), "{mode}");
        client.close().await.unwrap();
    }
    let mut limited = config("pages");
    limited.max_tools = 1;
    let client = McpClient::connect(limited).await.unwrap();
    assert!(client.tools("test").await.is_err());
    assert!(client.tools("bad.namespace").await.is_err());
    client.close().await.unwrap();
}

#[tokio::test]
async fn validates_inputs_maps_errors_and_structured_content() {
    for (mode, code) in [
        ("tool_error", "mcp_tool_error"),
        ("rpc_error", "mcp_rpc"),
        ("image", "tool_execution_in_doubt"),
        ("bad_output", "tool_execution_in_doubt"),
    ] {
        let client = McpClient::connect(config(mode)).await.unwrap();
        assert_eq!(
            invoke(&client, "error detail")
                .await
                .unwrap_err()
                .code
                .as_deref(),
            Some(code)
        );
        client.close().await.unwrap();
    }
    let client = McpClient::connect(config("structured")).await.unwrap();
    let tool = client.tools("test").await.unwrap().remove(0);
    assert_eq!(
        tool.execute(json!({"text":42}), ToolContext::new())
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("mcp_input")
    );
    let ToolResultOutput::Text(text) = tool
        .execute(json!({"text":"answer"}), ToolContext::new())
        .await
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["structuredContent"]["answer"],
        42
    );
    client.close().await.unwrap();
}

#[tokio::test]
async fn timeout_disconnect_and_malformed_response_poison_connection() {
    for mode in ["hang", "exit", "bad_json", "wrong_id", "oversize"] {
        let mut cfg = config(mode);
        cfg.timeout = Duration::from_millis(500);
        cfg.max_frame_bytes = 2048;
        let client = McpClient::connect(cfg).await.unwrap();
        let tool = client.tools("test").await.unwrap().remove(0);
        let error = tool
            .execute(json!({"text":"hello"}), ToolContext::new())
            .await
            .unwrap_err();
        assert_eq!(
            error.code.as_deref(),
            Some("tool_execution_in_doubt"),
            "{mode}"
        );
        assert!(!error.retryable);
        assert_eq!(
            tool.execute(json!({"text":"again"}), ToolContext::new())
                .await
                .unwrap_err()
                .code
                .as_deref(),
            Some("mcp_closed")
        );
    }
    let mut cfg = config("init_hang");
    cfg.timeout = Duration::from_millis(500);
    assert!(McpClient::connect(cfg).await.is_err());
}

#[tokio::test]
async fn metadata_is_not_forwarded_and_stderr_cannot_block_protocol() {
    let mut cfg = config("env");
    cfg.env
        .insert("AGENTSCOPE_MCP_TEST_ALLOWED".into(), "explicit".into());
    let client = McpClient::connect(cfg).await.unwrap();
    let tool = client.tools("test").await.unwrap().remove(0);
    let ToolResultOutput::Text(env_result) = tool
        .execute(
            json!({"text":"hi"}),
            ToolContext::new().with_idempotency_key("not-forwarded"),
        )
        .await
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        serde_json::from_str::<Value>(&env_result).unwrap(),
        json!({"value":"explicit","has_home":false,"has_path":false})
    );
    client.close().await.unwrap();
    let client = McpClient::connect(config("stderr")).await.unwrap();
    assert!(invoke(&client, "ok").await.is_ok());
    client.close().await.unwrap();
}

fn request_tool() -> ChatResponse {
    ChatResponse::finished(
        [
            ToolCallBlock::complete("call", "test__echo", r#"{"text":"hello"}"#)
                .unwrap()
                .into(),
        ],
        FinishReason::ToolCalls,
    )
}

#[tokio::test]
async fn react_confirmation_streaming_and_large_result_offload() {
    let client = McpClient::connect(config("large")).await.unwrap();
    let model = Arc::new(
        MockChatModel::new("offline")
            .with_response(request_tool())
            .with_stream([
                Ok(ChatEvent::TextDelta {
                    block_id: "answer".into(),
                    delta: "done".into(),
                }),
                Ok(ChatEvent::Finished {
                    reason: FinishReason::Completed,
                }),
            ]),
    );
    let dir = tempfile::tempdir().unwrap();
    let agent = ReActAgent::from_shared(
        "agent",
        model.clone(),
        ToolExecutor::new(client.registry("test").await.unwrap()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_tool_confirmation_required("test__echo")
    .with_tool_result_offload(
        ToolResultOffload::new(
            Arc::new(FileOffloadStore::new(dir.path()).unwrap()),
            1024,
            64,
            128,
        )
        .unwrap(),
    )
    .unwrap();
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("run")).await
    else {
        panic!("expected approval")
    };
    let mut stream = agent
        .stream_resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::Finished { .. } => finished = true,
            AgentEvent::Error { error, .. } => panic!("{error}"),
            _ => {}
        }
    }
    drop(stream);
    assert!(finished);
    let requests = model.recorded_requests();
    assert!(
        serde_json::to_string(&requests[1])
            .unwrap()
            .contains("offloaded_text")
    );
    let state = agent.snapshot().await.unwrap();
    assert!(state.messages().iter().flat_map(|m| &m.content).any(|b| matches!(b, ContentBlock::ToolResult(r) if matches!(r.output(), ToolResultOutput::Text(t) if t.len() > 10000))));
    client.close().await.unwrap();
}

#[tokio::test]
async fn confirmed_timeout_retains_durable_uncertain_checkpoint() {
    let mut cfg = config("hang");
    cfg.timeout = Duration::from_millis(500);
    let client = McpClient::connect(cfg).await.unwrap();
    let state_store = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("user", "mcp").unwrap();
    let agent = ReActAgent::new(
        "agent",
        MockChatModel::new("offline").with_response(request_tool()),
        ToolExecutor::new(client.registry("test").await.unwrap()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), state_store.clone())
    .with_tool_confirmation_required("test__echo");
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("run")).await
    else {
        panic!("expected approval")
    };
    assert!(matches!(
        agent
            .resume_tool_calls(
                checkpoint.reply_id(),
                vec![ToolConfirmation::approve("call")]
            )
            .await,
        Err(AgentError::ToolExecutionInDoubt { .. })
    ));
    assert!(
        state_store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_execution()
            .is_some()
    );
    assert!(matches!(
        agent.reply(Msg::user("do not rerun")).await,
        Err(AgentError::ToolExecutionInDoubt { .. })
    ));
}

#[tokio::test]
async fn concurrent_calls_are_correlated() {
    let client = McpClient::connect(config("normal")).await.unwrap();
    let tool = client.tools("test").await.unwrap().remove(0);
    let (a, b) = tokio::join!(
        tool.execute(json!({"text":"first"}), ToolContext::new()),
        tool.execute(json!({"text":"second"}), ToolContext::new())
    );
    assert_eq!(a.unwrap(), ToolResultOutput::Text("first".into()));
    assert_eq!(b.unwrap(), ToolResultOutput::Text("second".into()));
    client.close().await.unwrap();
}

#[cfg(unix)]
async fn assert_child_exited(pid: &str) {
    for _ in 0..100 {
        let status = tokio::process::Command::new("/bin/kill")
            .args(["-0", pid])
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .unwrap();
        if !status.success() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("fixture child did not exit");
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_call_closes_connection_and_kills_child() {
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit");
    let mut cfg = config("hang");
    cfg.args.push(audit.clone().into_os_string());
    let client = McpClient::connect(cfg).await.unwrap();
    let tool = client.tools("test").await.unwrap().remove(0);
    let mut call = Box::pin(tool.execute(json!({"text":"hello"}), ToolContext::new()));
    tokio::select! {
        result = &mut call => panic!("call unexpectedly completed: {result:?}"),
        () = async {
            for _ in 0..100 {
                if std::fs::read_to_string(&audit).unwrap_or_default().contains("tools/call") { return; }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("fixture never received call");
        } => {}
    }
    drop(call);
    assert_eq!(
        tool.execute(json!({"text":"again"}), ToolContext::new())
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("mcp_closed")
    );
    let pid = std::fs::read_to_string(audit.with_extension("pid")).unwrap();
    assert_child_exited(&pid).await;
    assert_eq!(
        std::fs::read_to_string(audit)
            .unwrap()
            .lines()
            .filter(|l| l.contains("tools/call"))
            .count(),
        1
    );
}

#[cfg(unix)]
#[tokio::test]
async fn dropping_final_handle_kills_child_and_spawn_failure_is_safe() {
    assert!(
        McpClient::connect(StdioConfig::new("/nonexistent/mcp-test-server"))
            .await
            .is_err()
    );
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit");
    let mut cfg = config("normal");
    cfg.args.push(audit.clone().into_os_string());
    let client = McpClient::connect(cfg).await.unwrap();
    let tool = client.tools("test").await.unwrap().remove(0);
    drop(client);
    assert!(
        tool.execute(json!({"text":"still owned"}), ToolContext::new())
            .await
            .is_ok()
    );
    let pid = std::fs::read_to_string(audit.with_extension("pid")).unwrap();
    drop(tool);
    assert_child_exited(&pid).await;
}

#[cfg(unix)]
#[tokio::test]
async fn explicit_close_kills_server_that_ignores_stdin_eof() {
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit");
    let mut cfg = config("stubborn");
    cfg.args.push(audit.clone().into_os_string());
    let client = McpClient::connect(cfg).await.unwrap();
    client.registry("test").await.unwrap();
    let pid = std::fs::read_to_string(audit.with_extension("pid")).unwrap();
    client.close().await.unwrap();
    assert_child_exited(&pid).await;
}
