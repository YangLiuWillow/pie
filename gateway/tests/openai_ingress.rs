//! OpenAI-compat ingress regression — `/v1/chat/completions` end-to-end
//! against the real assembled gateway, with a stub worker speaking the
//! gateway⇄inferlet envelope contract (`ingress/openai.rs` module docs):
//! first message `{"status": <u16>}`, then chunk payloads (streaming) or one
//! body (unary), then a clean `Eos`.
//!
//! In-proc like `gateway_smoke.rs` (seeded routing table, dialed-in stub
//! worker), but driven through the real axum listener with a raw HTTP/1.1
//! client so the wire framing (SSE `data:` lines, `[DONE]`, JSON error
//! bodies, status codes) is what's asserted.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;

use pie_client_api::{ClientMessage, ServerMessage};
use pie_controller_rpc::{
    Ack, GatewayInfo, Health, Role, RoutableWorker, RoutingTable, WorkerStatus,
};
use pie_gateway::{Gateway, GatewayConfig, GatewayControl, bind};
use pie_ids::{GatewayId, NodeId, WorkerId};
use pie_worker_rpc::{Accepted, Control, Priority, Request, Tokens};
use pie_worker_rpc::{GatewayInboundClient, WorkerControl, connect_gateway_link, dispatch_codec};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};

const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

// ───────────────────────── stub control plane ─────────────────────────

#[derive(Clone)]
struct StubControl {
    routing: Arc<watch::Sender<RoutingTable>>,
}

impl StubControl {
    fn seeded(table: RoutingTable) -> Self {
        let (tx, _) = watch::channel(table);
        Self {
            routing: Arc::new(tx),
        }
    }
}

impl GatewayControl for StubControl {
    async fn register_gateway(&self, _info: GatewayInfo) -> Result<GatewayId> {
        Ok(GatewayId(1))
    }

    async fn heartbeat(&self, _id: NodeId) -> Result<Ack> {
        Ok(Ack::Ok)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.routing.subscribe()
    }
}

// ───────────────────────── envelope stub worker ─────────────────────────

/// Speaks the serving-inferlet envelope per the dispatched request's body:
/// `{"trigger":"reject"}` → 400 + error body; `"stream": true` → status +
/// two chunks; else → status + one unary body. Always ends with a clean Eos.
#[derive(Clone)]
struct EnvelopeWorker {
    worker_id: WorkerId,
    gateway: GatewayInboundClient,
}

fn msg_event(value: &Value) -> ServerMessage {
    ServerMessage::ProcessEvent {
        process_id: "p0".into(),
        event: "message".into(),
        value: value.to_string(),
    }
}

impl WorkerControl for EnvelopeWorker {
    async fn dispatch(self, _: tarpc::context::Context, req: Request) -> Accepted {
        let gateway = self.gateway.clone();
        let req_id = req.req_id;
        let ClientMessage::LaunchProcess { input, inferlet, .. } = req.message else {
            panic!("openai ingress must dispatch LaunchProcess turns");
        };
        // The engine's ProgramName::parse requires `name@major.minor.patch`;
        // a bare name nacks every launch (universal 500s). Mirror that
        // contract here so the ingress can't regress it unnoticed again.
        let (name, version) = inferlet
            .split_once('@')
            .expect("inferlet id must be name@version — the engine rejects bare names");
        assert!(!name.is_empty());
        assert_eq!(
            version.split('.').count(),
            3,
            "inferlet version must be full semver, got {version:?}"
        );
        let body: Value = serde_json::from_str(&input).expect("ingress forwards valid JSON");

        tokio::spawn(async move {
            let mut script: Vec<ServerMessage> = vec![
                // Launch ack — the ingress must skip it.
                ServerMessage::Response {
                    corr_id: 0,
                    ok: true,
                    result: "launched".into(),
                },
                // Instrumentation noise — the ingress must drop it.
                ServerMessage::ProcessEvent {
                    process_id: "p0".into(),
                    event: "stdout".into(),
                    value: "debug line".into(),
                },
            ];
            if body.get("trigger").and_then(Value::as_str) == Some("reject") {
                script.push(msg_event(&json!({"status": 400})));
                script.push(msg_event(&json!({
                    "error": {"message": "bad request", "type": "invalid_request_error",
                              "param": null, "code": null}
                })));
            } else if body.get("stream").and_then(Value::as_bool) == Some(true) {
                script.push(msg_event(&json!({"status": 200})));
                script.push(msg_event(&json!({"id": "c0", "choices": [{"delta": {"role": "assistant"}}]})));
                script.push(msg_event(&json!({"id": "c1", "choices": [{"delta": {"content": "hi"}}]})));
            } else {
                script.push(msg_event(&json!({"status": 200})));
                script.push(msg_event(&json!({"id": "u0", "choices": [{"message": {"content": "hi"}}]})));
            }

            for msg in script {
                match gateway
                    .push_tokens(tarpc::context::current(), req_id, Tokens::Chunk(msg))
                    .await
                {
                    Ok(Control::Continue) => {}
                    _ => return,
                }
            }
            let _ = gateway
                .push_tokens(tarpc::context::current(), req_id, Tokens::Eos)
                .await;
        });
        Accepted::Ok {
            worker: self.worker_id,
        }
    }

    async fn cancel(self, _: tarpc::context::Context, _req_id: pie_ids::ReqId) {}
    async fn set_priority(
        self,
        _: tarpc::context::Context,
        _req_id: pie_ids::ReqId,
        _p: Priority,
    ) {
    }
    async fn drain(self, _: tarpc::context::Context) {}
}

async fn spawn_stub_worker(
    worker_addr: SocketAddr,
    worker_id: WorkerId,
) -> Result<tokio::task::JoinHandle<()>> {
    let mut conn = tcp::connect(worker_addr, dispatch_codec);
    conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
    let transport = conn.await?;
    let (server_half, gateway) = connect_gateway_link(transport);

    gateway
        .register(tarpc::context::current(), worker_id)
        .await?;

    let server = EnvelopeWorker { worker_id, gateway };
    let task = tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(server.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    );
    Ok(task)
}

// ───────────────────────── helpers ─────────────────────────

fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn seeded_table(worker_id: WorkerId) -> RoutingTable {
    RoutingTable {
        epoch: 1,
        workers: vec![RoutableWorker {
            id: worker_id,
            addr: "127.0.0.1:0".to_string(),
            role: Role::Decode,
            model: "stub-model".to_string(),
            health: Health::Healthy,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: 0,
                inflight: 0,
            },
        }],
    }
}

async fn wait_for_connected(
    connected: &watch::Receiver<Arc<HashSet<WorkerId>>>,
    worker_id: WorkerId,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if connected.borrow().contains(&worker_id) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("worker {worker_id} never appeared in the gateway connected set");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Bind, capture the connected-watch, then `into_handle()` — `bind` alone
/// never serves the client-facing edge (the hard-won lesson this helper
/// exists to encode), so a raw `TcpStream` to `listen_addr` would connect
/// (bound socket, kernel backlog) and then hang forever on the response.
async fn boot() -> Result<(
    SocketAddr,
    pie_gateway::GatewayHandle,
    tokio::task::JoinHandle<()>,
)> {
    let worker_id = WorkerId(7);
    let control = StubControl::seeded(seeded_table(worker_id));
    let config = GatewayConfig {
        listen: loopback(),
        worker_listen: loopback(),
        controller: String::new(),
    };
    let gw: Gateway = bind(config, control).await?;
    let connected = gw.state.workers.connected_watch();
    let listen_addr = gw.listen_addr;
    let worker_addr = gw.worker_addr;
    let handle = gw.into_handle();
    let worker = spawn_stub_worker(worker_addr, worker_id).await?;
    wait_for_connected(&connected, worker_id).await?;
    Ok((listen_addr, handle, worker))
}

/// Raw HTTP/1.1 request; returns (status line, headers, body). `Connection:
/// close` so the body is everything-until-EOF — which is exactly how the SSE
/// stream terminates after `[DONE]`.
async fn raw_http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    extra_headers: &str,
    body: &str,
) -> Result<(String, String, String)> {
    let mut stream = tokio::net::TcpStream::connect(addr).await?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: pie\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra_headers}\r\n{body}",
        body.len(),
    );
    stream.write_all(req.as_bytes()).await?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, payload) = text
        .split_once("\r\n\r\n")
        .context("malformed HTTP response")?;
    let (status_line, headers) = head.split_once("\r\n").unwrap_or((head, ""));
    Ok((
        status_line.to_string(),
        headers.to_string(),
        payload.to_string(),
    ))
}

/// Strip HTTP/1.1 chunked transfer framing (size lines), keeping payload lines.
fn dechunk(body: &str) -> String {
    body.lines()
        .filter(|l| !l.chars().all(|c| c.is_ascii_hexdigit()) || l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

const AUTH: &str = "Authorization: Bearer test-key\r\n";

// ───────────────────────── the tests ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streaming_turn_round_trips_as_sse() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, headers, body) = raw_http(
        addr,
        "POST",
        "/v1/chat/completions",
        AUTH,
        &json!({"model": "m", "stream": true, "messages": [{"role":"user","content":"hi"}]})
            .to_string(),
    )
    .await?;

    assert!(status.contains("200"), "got: {status}");
    assert!(
        headers.to_ascii_lowercase().contains("text/event-stream"),
        "streaming must be SSE; headers: {headers}"
    );
    let payload = dechunk(&body);
    // Chunks arrive in order, verbatim, one data: line each; then [DONE].
    let data_lines: Vec<&str> = payload
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(data_lines.len(), 3, "payload: {payload}");
    assert!(data_lines[0].contains("\"c0\""));
    assert!(data_lines[1].contains("\"c1\""));
    assert_eq!(data_lines[2], "[DONE]");
    // The envelope status header and stdout noise never reach the wire.
    assert!(!payload.contains("\"status\""));
    assert!(!payload.contains("debug line"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unary_turn_returns_json_body() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, headers, body) = raw_http(
        addr,
        "POST",
        "/v1/chat/completions",
        AUTH,
        &json!({"model": "m", "messages": [{"role":"user","content":"hi"}]}).to_string(),
    )
    .await?;

    assert!(status.contains("200"), "got: {status}");
    assert!(headers.to_ascii_lowercase().contains("application/json"));
    let v: Value = serde_json::from_str(dechunk(&body).trim()).context("body must be JSON")?;
    assert_eq!(v["id"], "u0");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pre_stream_rejection_is_a_json_error_not_sse() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, headers, body) = raw_http(
        addr,
        "POST",
        "/v1/chat/completions",
        AUTH,
        &json!({"model": "m", "stream": true, "trigger": "reject",
                "messages": [{"role":"user","content":"hi"}]})
        .to_string(),
    )
    .await?;

    assert!(status.contains("400"), "got: {status}");
    assert!(headers.to_ascii_lowercase().contains("application/json"));
    let v: Value = serde_json::from_str(dechunk(&body).trim())?;
    assert_eq!(v["error"]["type"], "invalid_request_error");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn missing_auth_is_401_openai_error() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, _, body) = raw_http(
        addr,
        "POST",
        "/v1/chat/completions",
        "",
        &json!({"messages": []}).to_string(),
    )
    .await?;

    assert!(status.contains("401"), "got: {status}");
    let v: Value = serde_json::from_str(dechunk(&body).trim())?;
    assert_eq!(v["error"]["type"], "authentication_error");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn invalid_json_is_400_without_a_launch() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, _, body) = raw_http(
        addr,
        "POST",
        "/v1/chat/completions",
        AUTH,
        "{not json",
    )
    .await?;

    assert!(status.contains("400"), "got: {status}");
    let v: Value = serde_json::from_str(dechunk(&body).trim())?;
    assert_eq!(v["error"]["type"], "invalid_request_error");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_and_models_respond() -> Result<()> {
    let (addr, _gw, _worker) = boot().await?;
    let (status, _, body) = raw_http(addr, "GET", "/health", "", "").await?;
    assert!(status.contains("200"));
    assert!(body.contains("ok"));

    let (status, _, body) = raw_http(addr, "GET", "/v1/models", "", "").await?;
    assert!(status.contains("200"), "got: {status}");
    let v: Value = serde_json::from_str(dechunk(&body).trim())?;
    assert_eq!(v["object"], "list");
    Ok(())
}
