//! Per-prefix snapshot retention (`max_snapshots_per_prefix`).
//!
//! Spawns a dedicated context manager (cap = 2) rather than using the
//! shared bootstrap env, whose cap is 0/off. Saves are pure bookkeeping
//! for empty contexts, so no mock driver is needed.

use std::sync::OnceLock;

const MODEL: usize = 0;
const USER: &str = "retention-user";
const CAP: usize = 2;

struct TestState {
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            pie::context::spawn(
                4,          // page_size
                vec![64],   // gpu pages
                vec![64],   // cpu pages
                4,          // max_forward_requests
                vec![0],    // rs slots
                vec![false],
                4,    // endowment
                None, // token limit
                32.0, // oversubscription
                0.85, // restore pause
                CAP,  // max_snapshots_per_prefix
            );
        });
        TestState { rt }
    })
}

async fn fresh_pid() -> uuid::Uuid {
    let pid = uuid::Uuid::new_v4();
    pie::context::register_process(pid, None).await.unwrap();
    pid
}

async fn save_named(name: &str) {
    let id = pie::context::create(MODEL, fresh_pid().await).await.unwrap();
    pie::context::save(MODEL, id, USER.to_string(), Some(name.into()))
        .await
        .unwrap();
}

async fn exists(name: &str) -> bool {
    pie::context::lookup(MODEL, USER.to_string(), name.into())
        .await
        .is_ok()
}

#[test]
fn oldest_snapshot_evicted_beyond_prefix_cap() {
    let s = state();
    s.rt.block_on(async {
        save_named("ns/a").await;
        save_named("ns/b").await;
        assert!(exists("ns/a").await && exists("ns/b").await);

        // Third save under the same prefix evicts the oldest (a).
        save_named("ns/c").await;
        assert!(!exists("ns/a").await, "oldest under prefix should be evicted");
        assert!(exists("ns/b").await && exists("ns/c").await);

        // And a fourth evicts b.
        save_named("ns/d").await;
        assert!(!exists("ns/b").await);
        assert!(exists("ns/c").await && exists("ns/d").await);
    });
}

#[test]
fn other_prefixes_and_flat_names_are_unaffected() {
    let s = state();
    s.rt.block_on(async {
        // A different namespace has its own budget.
        save_named("other/x").await;
        save_named("other/y").await;

        // Flat (non-namespaced) names never participate in retention.
        save_named("flat1").await;
        save_named("flat2").await;
        save_named("flat3").await;

        // Filling `deep/` beyond cap must not touch `other/` or flat names.
        save_named("deep/1").await;
        save_named("deep/2").await;
        save_named("deep/3").await;

        assert!(!exists("deep/1").await);
        assert!(exists("deep/2").await && exists("deep/3").await);
        assert!(exists("other/x").await && exists("other/y").await);
        assert!(exists("flat1").await && exists("flat2").await && exists("flat3").await);
    });
}
