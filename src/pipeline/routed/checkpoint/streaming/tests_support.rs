use crate::*;
use futures_util::{StreamExt, future::poll_fn};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};

pub(super) fn key(session: &str) -> StateKey {
    StateKey::new("routed-stream-test", session).unwrap()
}

pub(super) fn checkpoint(status: RoutedCheckpointStatus) -> RoutedCheckpoint {
    RoutedCheckpoint {
        version: ROUTED_CHECKPOINT_VERSION,
        route: "selected".into(),
        agent_name: "worker".into(),
        input: Msg::user("saved input"),
        status,
    }
}

#[derive(Default)]
pub(super) struct StoreGate {
    pub(super) polls: AtomicUsize,
    pub(super) effects: AtomicUsize,
    pub(super) dropped: AtomicUsize,
    released: AtomicBool,
    waker: Mutex<Option<Waker>>,
}
impl StoreGate {
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
    fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if self.released.load(Ordering::SeqCst) {
            self.effects.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(())
        } else {
            *self.waker.lock().unwrap() = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
struct SaveMarker<'a>(&'a StoreGate);
impl Drop for SaveMarker<'_> {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
pub(super) enum SavePoint {
    Initial,
    Marker,
    Terminal,
}
impl SavePoint {
    fn matches(self, checkpoint: &RoutedCheckpoint) -> bool {
        matches!(
            (self, &checkpoint.status),
            (Self::Initial, RoutedCheckpointStatus::Ready)
                | (Self::Marker, RoutedCheckpointStatus::InFlight)
                | (
                    Self::Terminal,
                    RoutedCheckpointStatus::Completed(_) | RoutedCheckpointStatus::Failed(_)
                )
        )
    }
}
#[derive(Clone)]
pub(super) enum SaveAction {
    Pass,
    Gate(Arc<StoreGate>),
    Reject,
    Ambiguous,
    BadAck,
}
pub(super) struct ControlledStore {
    pub(super) inner: InMemoryRoutedStore,
    pub(super) loads: AtomicUsize,
    pub(super) saves: AtomicUsize,
    point: SavePoint,
    action: SaveAction,
}
impl ControlledStore {
    pub(super) fn new(point: SavePoint, action: SaveAction) -> Self {
        Self {
            inner: InMemoryRoutedStore::new(),
            loads: AtomicUsize::new(0),
            saves: AtomicUsize::new(0),
            point,
            action,
        }
    }
    pub(super) fn plain() -> Self {
        Self::new(SavePoint::Initial, SaveAction::Pass)
    }
}
impl RoutedStore for ControlledStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<RoutedRecord>> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(key)
    }
    fn save(
        &self,
        key: StateKey,
        expected: Option<u64>,
        checkpoint: RoutedCheckpoint,
    ) -> PipelineStoreFuture<'_, RoutedRecord> {
        Box::pin(async move {
            self.saves.fetch_add(1, Ordering::SeqCst);
            let selected = self.point.matches(&checkpoint);
            if selected && matches!(self.action, SaveAction::Reject) {
                return Err(PipelineStoreError::new("write rejected before commit"));
            }
            let mut record = self.inner.save(key, expected, checkpoint).await?;
            if selected {
                match &self.action {
                    SaveAction::Gate(gate) => {
                        let _marker = SaveMarker(gate);
                        poll_fn(|cx| gate.poll(cx)).await;
                    }
                    SaveAction::Ambiguous => {
                        return Err(PipelineStoreError::new("commit acknowledgement lost"));
                    }
                    SaveAction::BadAck => record.revision = 0,
                    SaveAction::Pass | SaveAction::Reject => {}
                }
            }
            Ok(record)
        })
    }
}

pub(super) async fn assert_status(
    store: &dyn RoutedStore,
    state_key: &StateKey,
    status: RoutedCheckpointStatus,
) -> RoutedRecord {
    let record = store.load(state_key).await.unwrap().unwrap();
    assert_eq!(record.checkpoint.status, status);
    record
}

pub(super) async fn assert_unsafe_stream_resume(
    pipeline: &RoutedPipeline,
    store: &dyn RoutedStore,
    state_key: StateKey,
) {
    let Err(error) = pipeline.resume_checkpointed_stream(store, state_key).await else {
        panic!("unsafe progress must not resume")
    };
    assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
}

pub(super) async fn collect_to_terminal(stream: &mut RoutedEventStream<'_>) -> Vec<RoutedEvent> {
    let mut observed = Vec::new();
    while let Some(event) = stream.next().await {
        let terminal = matches!(
            event,
            RoutedEvent::Finished { .. } | RoutedEvent::Error { .. }
        );
        observed.push(event);
        if terminal {
            break;
        }
    }
    assert!(matches!(
        observed.last(),
        Some(RoutedEvent::Finished { .. } | RoutedEvent::Error { .. })
    ));
    observed
}

pub(super) fn rich_fixture() -> (Msg, Msg, Vec<AgentEvent>) {
    let mut input = Msg::new(
        "input-author",
        Role::System,
        [
            ThinkingBlock::new("private input").into(),
            DataBlock::url("https://example.com/image.png", "image/png")
                .unwrap()
                .into(),
        ],
    );
    input
        .metadata
        .insert("opaque".into(), serde_json::json!({"value":7}));
    let mut message = Msg::new(
        "original-author",
        Role::User,
        [
            ThinkingBlock::new("private reply").into(),
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .into(),
        ],
    );
    message
        .metadata
        .insert("private".into(), serde_json::json!(true));
    let events = vec![
        AgentEvent::ContextCompactionFailed {
            error: AgentError::Model(ModelError::new("nonterminal")),
        },
        AgentEvent::ThinkingDelta {
            step: 17,
            block_id: "thinking".into(),
            delta: "reasoning".into(),
        },
        AgentEvent::TextDelta {
            step: 17,
            block_id: "text".into(),
            delta: "visible".into(),
        },
        AgentEvent::StepFinished {
            step: 17,
            reason: FinishReason::Completed,
        },
        AgentEvent::Finished {
            steps: 42,
            message: message.clone(),
        },
    ];
    (input, message, events)
}
