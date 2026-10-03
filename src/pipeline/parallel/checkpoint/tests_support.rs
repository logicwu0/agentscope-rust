use crate::*;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::Notify;

pub(super) fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(
        MockChatModel::new("offline")
            .with_response(ChatResponse::completed([ContentBlock::from(text)])),
    )
}

pub(super) fn agent(name: &str, model: Arc<dyn ChatModel>) -> Arc<ReActAgent> {
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

pub(super) fn reply(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}

pub(super) fn checkpoint(
    names: &[&str],
    input: Msg,
    branches: Vec<ParallelBranchCheckpoint>,
) -> ParallelCheckpoint {
    ParallelCheckpoint {
        version: PARALLEL_CHECKPOINT_VERSION,
        agent_names: names.iter().map(|name| (*name).into()).collect(),
        input,
        branches,
    }
}

#[derive(Default)]
pub(super) struct Activity {
    pub(super) active: AtomicUsize,
    pub(super) peak: AtomicUsize,
}

pub(super) struct GateModel {
    activity: Arc<Activity>,
    witness: Option<(Arc<InMemoryParallelStore>, StateKey, usize)>,
    requests: Mutex<Vec<ChatRequest>>,
    release: Notify,
    released: AtomicBool,
    pub(super) dropped: AtomicUsize,
}

struct ActiveCall<'a>(&'a GateModel);
impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        self.0.activity.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl GateModel {
    pub(super) fn new(
        activity: &Arc<Activity>,
        witness: Option<(Arc<InMemoryParallelStore>, StateKey, usize)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            activity: activity.clone(),
            witness,
            requests: Mutex::new(Vec::new()),
            release: Notify::new(),
            released: AtomicBool::new(false),
            dropped: AtomicUsize::new(0),
        })
    }
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_one();
    }
    pub(super) fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "checkpoint-gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            if let Some((store, key, index)) = &self.witness {
                let record = store
                    .load(key)
                    .await
                    .unwrap()
                    .expect("checkpoint must exist before invocation");
                assert_eq!(
                    record.checkpoint.branches[*index],
                    ParallelBranchCheckpoint::InFlight,
                    "the branch marker must be durable before a model call"
                );
            }
            self.requests.lock().unwrap().push(request);
            let active = self.activity.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.activity.peak.fetch_max(active, Ordering::SeqCst);
            let _call = ActiveCall(self);
            while !self.released.load(Ordering::SeqCst) {
                self.release.notified().await;
            }
            Ok(ChatResponse::completed([ContentBlock::from(
                "gate completed",
            )]))
        })
    }
    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("unused stream")) })
    }
}

#[derive(Clone, Copy)]
pub(super) enum SavePoint {
    Initial,
    Marker,
    Terminal,
}

impl SavePoint {
    fn matches(self, expected: Option<u64>, checkpoint: &ParallelCheckpoint) -> bool {
        match self {
            Self::Initial => expected.is_none(),
            Self::Marker => {
                expected.is_some()
                    && checkpoint
                        .branches
                        .iter()
                        .any(|branch| matches!(branch, ParallelBranchCheckpoint::InFlight))
                    && !checkpoint.branches.iter().any(|branch| {
                        matches!(
                            branch,
                            ParallelBranchCheckpoint::Completed(_)
                                | ParallelBranchCheckpoint::Failed(_)
                        )
                    })
            }
            Self::Terminal => checkpoint.branches.iter().any(|branch| {
                matches!(
                    branch,
                    ParallelBranchCheckpoint::Completed(_) | ParallelBranchCheckpoint::Failed(_)
                )
            }),
        }
    }
}

pub(super) struct ControlledStore {
    pub(super) inner: InMemoryParallelStore,
    reject: Option<SavePoint>,
    signal_on: Option<SavePoint>,
    pub(super) interrupt: Mutex<Option<AgentInterruptHandle>>,
}

impl ControlledStore {
    pub(super) fn reject(point: SavePoint) -> Self {
        Self {
            inner: InMemoryParallelStore::new(),
            reject: Some(point),
            signal_on: None,
            interrupt: Mutex::new(None),
        }
    }
    pub(super) fn interrupt_after(point: SavePoint) -> Self {
        Self {
            inner: InMemoryParallelStore::new(),
            reject: None,
            signal_on: Some(point),
            interrupt: Mutex::new(None),
        }
    }
}

impl ParallelStore for ControlledStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        self.inner.load(key)
    }
    fn save(
        &self,
        key: StateKey,
        expected: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            if self
                .reject
                .is_some_and(|point| point.matches(expected, &checkpoint))
            {
                return Err(PipelineStoreError::new("deliberate write rejection"));
            }
            let signal = self
                .signal_on
                .is_some_and(|point| point.matches(expected, &checkpoint));
            let record = self.inner.save(key, expected, checkpoint).await?;
            if signal {
                self.interrupt.lock().unwrap().as_ref().unwrap().interrupt();
            }
            Ok(record)
        })
    }
}
