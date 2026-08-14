//! End-to-end inferlet execution tests.
//!
//! These tests actually **run** real WASM inferlets through the full stack:
//! program add → install → process::spawn() → linker instantiation → WASM
//! execution → process completion.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::EchoBehavior};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

/// Timeout for a single process to complete.
const PROCESS_TIMEOUT: Duration = Duration::from_secs(10);

/// Shared state: MockEnv + tokio runtime.
struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("test-model", 1, 16, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// Spawn a process within the tokio runtime and wait for it to complete.
/// Returns true if the process exited within the timeout.
fn spawn_and_wait(s: &TestState, name: &str, input: String) -> bool {
    let pid = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        process::spawn(
            "test-user".into(),
            program_name(name),
            input,
            None,
            false,
            None,
        )
        .expect("spawn")
    });

    // Wait in a tokio context so the WASM task can make progress
    s.rt.block_on(async {
        tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok()
    })
}

/// Spawn a process and capture its actual return value (the inferlet's
/// `Result<String, String>`), so a test can assert the inferlet ran to a
/// meaningful result rather than merely exiting. `Err` if the process did not
/// terminate within the timeout (e.g. a hang on an unresolved host RPC).
fn spawn_and_capture(s: &TestState, name: &str, input: String) -> Result<String, String> {
    let rx = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "test-user".into(),
            program_name(name),
            input,
            None,
            false,
            Some(tx),
        )
        .expect("spawn");
        rx
    });

    s.rt.block_on(async {
        match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("process result channel dropped before completion".to_string()),
            Err(_) => Err("process did not complete within timeout (hang)".to_string()),
        }
    })
}

// =============================================================================
// Basic E2E Tests
// =============================================================================

#[test]
fn echo_runs_to_completion() {
    let s = state();
    assert!(
        spawn_and_wait(s, "echo", r#"{"message":"hello world"}"#.into()),
        "echo inferlet should complete within timeout"
    );
}

#[test]
fn error_inferlet_exits() {
    let s = state();
    assert!(
        spawn_and_wait(s, "error", "{}".into()),
        "error inferlet should complete even on error"
    );
}

#[test]
fn context_inferlet_exercises_host_apis() {
    let s = state();
    assert!(
        spawn_and_wait(s, "context", "{}".into()),
        "context inferlet should complete (exercises model, tokenizer, context host APIs)"
    );
}

#[test]
fn direct_ptir_inferlet_exercises_bound_channels() {
    let s = state();
    let result = spawn_and_capture(s, "direct-channel-e2e", "{}".into());
    assert!(
        result.as_deref() == Ok("value=42"),
        "direct PTIR inferlet should publish and read bound channels (got {result:?})"
    );
}

#[test]
fn direct_ptir_mixed_outputs_execute_end_to_end() {
    let s = state();
    let result = spawn_and_capture(s, "direct-mixed-e2e", "{}".into())
        .expect("mixed direct-channel inferlet should complete successfully");
    for required in [
        "MIXED_OK=true",
        "ENTROPY_OK=true",
        "VECTOR_OK=true",
        "EMPTY_PREFIX_OK=true",
        "MULTISAMPLER_OK=true",
    ] {
        assert!(
            result.contains(required),
            "mixed direct-channel e2e missing {required}: {result:?}"
        );
    }
}

#[test]
fn explicit_prefix_index_submits_only_the_suffix() {
    let s = state();
    // Round 0 publishes one full page under an explicit key; round 1 loads
    // that WorkingSet and submits only the 8-token suffix.
    let result = spawn_and_capture(s, "prefix-cache-e2e", "{}".into());
    assert!(
        result
            .as_deref()
            .is_ok_and(|r| r.contains("PREFIX_CACHE_E2E n=24")),
        "prefix-cache inferlet should complete (got {result:?})"
    );
    let shapes: Vec<String> = s
        .env
        .operations()
        .into_iter()
        .filter(|op| op.starts_with("launch-shape"))
        .collect();
    // Match this inferlet's PER-PROGRAM spans (`per=[..]`): the shared-engine
    // suite runs concurrently and the wait-all quorum may co-batch another
    // test's fire into the same launch, so batch totals are not stable.
    let per_counts = |op: &str| -> Vec<u32> {
        op.split_once("per=[")
            .and_then(|(_, rest)| rest.split_once(']'))
            .map(|(list, _)| {
                list.split(',')
                    .filter_map(|count| count.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let cold = shapes.iter().position(|op| per_counts(op).contains(&24));
    assert!(
        cold.is_some(),
        "round 0 should prefill all 24 tokens: {shapes:?}"
    );
    assert!(
        shapes[cold.unwrap() + 1..]
            .iter()
            .any(|op| per_counts(op).contains(&8)),
        "round 1 should load the indexed page and fire only the 8-token \
         suffix: {shapes:?}"
    );
}

/// Does a WorkingSet published by one inferlet INSTANCE survive for the next?
///
/// `explicit_prefix_index_submits_only_the_suffix` publishes and looks up
/// inside a single invocation, so it cannot answer this. A per-request prefix
/// cache — one wasm instance per HTTP request, the shape RatioThink's chat-apc
/// uses and the shape `inferlets/chat-completions` marks as a SEAM — is only
/// possible if the index outlives the instance that wrote it.
#[test]
fn explicit_prefix_index_survives_across_inferlet_instances() {
    let s = state();
    let published = spawn_and_capture(s, "prefix-cache-e2e", "\"publish-only\"".into());
    assert!(
        published.as_deref().is_ok_and(|r| r.contains("published")),
        "the publishing instance should complete (got {published:?})"
    );
    // A SECOND instance, with no shared guest state, looking up the same key.
    let found = spawn_and_capture(s, "prefix-cache-e2e", "\"lookup-only\"".into());
    assert!(
        found.as_deref().is_ok_and(|r| r.contains("cross_instance_hit")),
        "a later instance must find the published prefix; a miss here means the \
         index is process-local and a per-request APC cannot work (got {found:?})"
    );
}

/// Can an instance park a prefix it RESUMED, for the instance after it?
///
/// This is the shape a per-request prefix cache actually runs
/// (`inferlets/chat-completions`): every turn but the first loads a parked
/// prefix, grafts its delta on, and parks a deeper prefix. The slice it parks
/// therefore spans inherited pages — structurally shared with every other
/// holder of that key — and pages it just wrote. `explicit_prefix_index_
/// survives_across_inferlet_instances` stops one step short of that: its
/// publisher always starts cold.
///
/// The failure this catches is silent. A graft onto a prefix that ends in the
/// wrong place still decodes fluent text, so nothing downstream would report it.
#[test]
fn a_resumed_prefix_can_be_republished_deeper() {
    let s = state();
    // Three separate instances, no shared guest state: cold → resume 16 and
    // park 32 → resume 32.
    let mut fired = Vec::new();
    for step in 0..3 {
        let out = spawn_and_capture(s, "prefix-cache-e2e", format!("\"chain-{step}\""));
        let out = out.unwrap_or_else(|e| {
            panic!("chain step {step} should complete, got {e:?} (a miss here means the \
                    previous step could not park a prefix built on inherited pages)")
        });
        assert!(
            out.contains(&format!("chain step={step}")),
            "chain step {step} reported {out:?}"
        );
        fired.push(out);
    }
    // The whole point is that each step prefills only its delta. A step that
    // silently fell back to a cold rebuild would still answer — and still pass
    // a hit/miss assertion — so assert the DEPTH.
    for (step, expect) in [(0usize, "cached=0 fired=24"), (1, "cached=16 fired=24"), (2, "cached=32 fired=16")] {
        assert!(
            fired[step].contains(expect),
            "chain step {step} should report {expect}, got {:?}",
            fired[step]
        );
    }
}

#[test]
fn spawn_after_termination() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        // Spawn and immediately terminate
        let pid1 = process::spawn(
            "term-user".into(),
            program_name("echo"),
            r#"{"msg":"will-be-terminated"}"#.into(),
            None,
            false,
            None,
        )
        .expect("spawn for termination");

        process::terminate(pid1, Err("test termination".into()));

        let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid1) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok();
        assert!(completed, "terminated process should disappear");

        // Spawn another — should work fine, no stale state
        let pid2 = process::spawn(
            "term-user".into(),
            program_name("echo"),
            r#"{"msg":"after-termination"}"#.into(),
            None,
            false,
            None,
        )
        .expect("spawn after termination");

        let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid2) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok();
        assert!(
            completed,
            "process spawned after termination should complete normally"
        );
    });
}
// =============================================================================
// Stress & Concurrency
// =============================================================================

#[test]
fn concurrent_spawns() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        let count = 10;
        let pids: Vec<_> = (0..count)
            .map(|i| {
                let pid = process::spawn(
                    "stress-user".into(),
                    program_name("echo"),
                    format!(r#"{{"batch":"{i}"}}"#),
                    None,
                    false,
                    None,
                )
                .unwrap_or_else(|e| panic!("spawn {i} failed: {e}"));
                pid
            })
            .collect();

        for (i, pid) in pids.iter().enumerate() {
            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(
                completed,
                "concurrent echo {i} (pid {pid}) did not complete"
            );
        }
    });
}

#[test]
fn rapid_sequential_spawns() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        for i in 0..50 {
            let pid = process::spawn(
                "seq-user".into(),
                program_name("echo"),
                format!(r#"{{"seq":"{i}"}}"#),
                None,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("sequential spawn {i} failed: {e}"));

            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(&pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(
                completed,
                "sequential echo {i} (pid {pid}) did not complete"
            );
        }

        // After all processes finish, the process list should not grow unboundedly
        let remaining = process::list().len();
        assert!(
            remaining < 5,
            "expected few residual processes, got {remaining}"
        );
    });
}

#[test]
fn mixed_success_and_error() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;
        inferlets::add_and_install("error").await;

        let mut pids = Vec::new();
        for i in 0..10 {
            let (name, input) = if i % 2 == 0 {
                ("echo", format!(r#"{{"msg":"ok-{i}"}}"#))
            } else {
                ("error", "{}".to_string())
            };
            let pid = process::spawn(
                "mixed-user".into(),
                program_name(name),
                input,
                None,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("mixed spawn {i} ({name}) failed: {e}"));
            pids.push((i, name, pid));
        }

        for (i, name, pid) in &pids {
            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(completed, "mixed {i} ({name}, pid {pid}) did not complete");
        }
    });
}
