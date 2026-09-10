use agentscope::{
    AgentError, AgentEvent, IdempotencyRequest, IdempotencyStore, Msg, ReActAgent,
    ToolConfirmation, ToolContext,
};
use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
use futures_util::StreamExt;
use std::{
    io::{self, Write},
    sync::Arc,
};

use crate::Result;

pub struct Session {
    pub agent: ReActAgent,
    pub idempotency: Arc<SQLiteIdempotencyStore>,
    pub namespace: String,
}

impl Session {
    pub async fn status(&self) -> Result<()> {
        let state = self.agent.snapshot().await?;
        println!("Saved messages: {}", state.messages().len());
        if let Some(pending) = state.pending_tool_calls() {
            println!("Approval required: {}", pending.reply_id());
            for call in pending.calls() {
                println!("  {}: {} {}", call.id(), call.name(), call.input());
            }
            println!(
                "/approve executes ALL listed calls; /deny rejects ALL; /quit leaves them pending."
            );
        } else if let Some(execution) = state.pending_tool_execution() {
            println!("Uncertain execution: {}", execution.execution_id());
            for call in execution.confirmation().calls() {
                if let Some(key) = execution.idempotency_key(call.id()) {
                    println!(
                        "  {}: {} {} [key: {}]",
                        call.id(),
                        call.name(),
                        call.input(),
                        key
                    );
                }
            }
            println!(
                "/retry explicitly retries using the original keys. Unfinished durable records block execution.\nAfter stopping the original worker and checking the outcome: /resolve CALL_ID VERIFIED_TEXT, then /retry."
            );
        }
        Ok(())
    }

    pub async fn handle(&self, line: &str) -> Result<()> {
        match line {
            "/help" => println!(
                "/status /history /approve /deny [reason] /retry /resolve CALL_ID VERIFIED_TEXT /quit\nChat, approval, and recovery replies stream with tool progress.\nOffline tool prompt: multiply 6 7. Input is one line at a time. Use one process per session."
            ),
            "/status" => self.status().await?,
            "/history" => {
                for msg in self.agent.snapshot().await?.messages() {
                    println!(
                        "{}: {}",
                        msg.name,
                        msg.text_content("")
                            .unwrap_or_else(|| format!("{:?}", msg.content))
                    );
                }
            }
            "/approve" => self.confirm(None).await?,
            "/deny" => self.confirm(Some("User denied execution")).await?,
            "/retry" => {
                let state = self.agent.snapshot().await?;
                let execution = state
                    .pending_tool_execution()
                    .ok_or("No uncertain execution")?;
                self.show_stream(
                    self.agent
                        .stream_retry_tool_execution(execution.confirmation().reply_id())
                        .await,
                )
                .await?;
            }
            _ if line.starts_with("/deny ") => self.confirm(Some(line[6..].trim())).await?,
            _ if line.starts_with("/resolve ") => self.resolve(line[9..].trim()).await?,
            _ if line.starts_with('/') => return Err("Unknown command; use /help".into()),
            _ => self.chat(line).await?,
        }
        Ok(())
    }

    async fn confirm(&self, denial: Option<&str>) -> Result<()> {
        let state = self.agent.snapshot().await?;
        let pending = state
            .pending_tool_calls()
            .ok_or("No pending confirmation")?;
        let decisions = pending
            .calls()
            .iter()
            .map(|call| match denial {
                Some(reason) => ToolConfirmation::deny(call.id(), reason),
                None => ToolConfirmation::approve(call.id()),
            })
            .collect();
        self.show_stream(
            self.agent
                .stream_resume_tool_calls(pending.reply_id(), decisions)
                .await,
        )
        .await
    }

    async fn show_error(&self, error: AgentError) -> Result<()> {
        match error {
            AgentError::ToolConfirmationRequired { .. }
            | AgentError::ToolExecutionInDoubt { .. } => self.status().await?,
            error => return Err(error.into()),
        }
        Ok(())
    }

    async fn chat(&self, line: &str) -> Result<()> {
        self.show_stream(self.agent.stream(Msg::user(line)).await)
            .await
    }

    async fn show_stream(
        &self,
        result: agentscope::AgentResult<agentscope::AgentEventStream<'_>>,
    ) -> Result<()> {
        let mut events = match result {
            Ok(events) => events,
            Err(error) => return self.show_error(error).await,
        };
        print!("assistant> ");
        io::stdout().flush()?;
        // Always consume through the terminal event so the agent saves state.
        let mut paused = false;
        while let Some(event) = events.next().await {
            match event? {
                AgentEvent::TextDelta { delta, .. } => {
                    print!("{delta}");
                    io::stdout().flush()?;
                }
                AgentEvent::ToolConfirmationRequired { .. } => paused = true,
                AgentEvent::ToolStarted { call, .. } => {
                    println!("\n[tool started] {} {}", call.name(), call.input());
                }
                AgentEvent::ToolFinished { result, .. } => {
                    println!("[tool finished] {}: {:?}", result.name(), result.state());
                }
                AgentEvent::Error { error, .. } => {
                    println!();
                    drop(events);
                    return self.show_error(error).await;
                }
                _ => {}
            }
        }
        drop(events);
        println!();
        if paused {
            self.status().await?;
        }
        Ok(())
    }

    async fn resolve(&self, arguments: &str) -> Result<()> {
        let (id, text) = arguments
            .split_once(' ')
            .ok_or("Usage: /resolve CALL_ID VERIFIED_TEXT")?;
        if text.trim().is_empty() {
            return Err("Verified result cannot be empty".into());
        }
        let state = self.agent.snapshot().await?;
        let execution = state
            .pending_tool_execution()
            .ok_or("No uncertain execution")?;
        let call = execution
            .confirmation()
            .calls()
            .iter()
            .find(|call| call.id() == id)
            .ok_or("Unknown call ID")?;
        let key = execution
            .idempotency_key(id)
            .ok_or("Call was not approved")?;
        let request = IdempotencyRequest::new(
            &self.namespace,
            call.name(),
            call.parsed_input()?,
            ToolContext::new().with_idempotency_key(key),
        )?;
        self.idempotency
            .reconcile(request, Ok(text.to_owned().into()))
            .await?;
        println!("Verified result saved. Use /retry to continue from cached results.");
        Ok(())
    }
}
