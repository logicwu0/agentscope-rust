use super::*;
use crate::{ToolCallBlock, ToolResultBlock};

fn call(id: &str) -> Msg {
    Msg::new(
        "agent",
        Role::Assistant,
        [ContentBlock::from(
            ToolCallBlock::complete(id, "tool", "{}").unwrap(),
        )],
    )
}

fn result(id: &str) -> Msg {
    Msg::new(
        "tool",
        Role::Assistant,
        [ContentBlock::from(
            ToolResultBlock::success(id, "tool", "done").unwrap(),
        )],
    )
}

#[test]
fn validates_turn_limit() {
    assert_eq!(RecentTurns::new(0), Err(ZeroContextTurns));
    assert_eq!(RecentTurns::new(3).unwrap().max_turns(), 3);
}

#[test]
fn full_context_and_short_histories_are_unchanged() {
    let policy = RecentTurns::new(1).unwrap();
    for history in [
        vec![],
        vec![Msg::assistant("other", "observation")],
        vec![
            Msg::system("instructions"),
            Msg::user("question"),
            call("c"),
            result("c"),
        ],
    ] {
        assert_eq!(FullContext.select_messages(&history), history);
        assert_eq!(policy.select_messages(&history), history);
        assert_eq!(
            RecentTurns::new(usize::MAX)
                .unwrap()
                .select_messages(&history),
            history
        );
    }
}

#[test]
fn retains_system_messages_and_whole_recent_turns_in_order() {
    let history = vec![
        Msg::system("initial"),
        Msg::user("old"),
        call("old"),
        result("old"),
        Msg::assistant("a", "old answer"),
        Msg::system("updated"),
        Msg::user("recent"),
        call("a"),
        call("b"),
        result("a"),
        result("b"),
        Msg::assistant("a", "answer"),
        Msg::user("current"),
        call("c"),
        result("c"),
    ];
    let original = history.clone();
    let expected = [history[..1].to_vec(), history[5..].to_vec()].concat();
    assert_eq!(
        RecentTurns::new(2).unwrap().select_messages(&history),
        expected
    );
    let expected = [
        vec![history[0].clone(), history[5].clone()],
        history[12..].to_vec(),
    ]
    .concat();
    assert_eq!(
        RecentTurns::new(1).unwrap().select_messages(&history),
        expected
    );
    assert_eq!(history, original);
}

#[test]
fn crosses_user_boundary_without_orphaning_tool_results() {
    let history = vec![
        Msg::user("discard"),
        Msg::assistant("a", "old"),
        Msg::user("keep"),
        call("c"),
        Msg::user("current"),
        result("c"),
    ];
    assert_eq!(
        RecentTurns::new(1).unwrap().select_messages(&history),
        history[2..]
    );
}

#[test]
fn expands_transitive_dependencies_and_retains_batched_siblings() {
    let history = vec![
        Msg::user("discard"),
        Msg::assistant("a", "old"),
        Msg::user("first"),
        call("a"),
        call("b"),
        result("b"),
        Msg::user("second"),
        result("a"),
        call("c"),
        Msg::user("third"),
        result("c"),
    ];
    assert_eq!(
        RecentTurns::new(1).unwrap().select_messages(&history),
        history[2..]
    );
}

#[test]
fn consecutive_user_messages_are_separate_turns() {
    let history = vec![Msg::user("first"), Msg::user("second"), Msg::user("third")];
    assert_eq!(
        RecentTurns::new(2).unwrap().select_messages(&history),
        history[1..]
    );
}
