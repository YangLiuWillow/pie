//! A refused turn must not destroy the session.
//!
//! On this gateway a closed client WebSocket is not a closed request, it is a
//! closed *process*: `serve()`'s exit runs `handle.close()`, the worker tears
//! the session's inferlet down, and every working set it retained dies with it.
//! So `break`ing out of the loop on a refused turn spent the session's whole
//! warm KV to report a transient condition.
//!
//! It is transient by construction. `gateway::admission` reports "cluster
//! saturated" from the worker's `kv_pressure_bucket`, which the planner clamps
//! to 240 whenever it has ANY queued allocation — exactly the gate's
//! `kv_saturate_bucket`. One momentarily queued allocation therefore refused a
//! turn and killed a session holding 55k tokens of warm KV
//! (`integrations/opencode/finding-inferlet-killed-at-large-context.md`).
//!
//! This drives the real axum ingress over a real socket, because the behaviour
//! under test lives in `serve()`'s loop and nowhere else: the refusal is
//! produced by the same `admit` the production path calls, and the assertion is
//! that the socket is still usable afterwards.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use futures::{SinkExt, StreamExt};
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;

use pie_client_api::{ClientMessage, ServerMessage};
use pie_controller_rpc::{
    Ack, GatewayInfo, Health, Role, RoutableWorker, RoutingTable, WorkerStatus,
};
use pie_gateway::{Gateway, GatewayConfig, GatewayControl, bind};
use pie_ids::{GatewayId, NodeId, ReqId, WorkerId};
use pie_worker_rpc::{Accepted, Control, Priority, Request, Tokens};
use pie_worker_rpc::{GatewayInboundClient, WorkerControl, connect_gateway_link, dispatch_codec};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};

const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// The bucket the planner clamps to on `waiters != 0`, and the gate's
/// `kv_saturate_bucket`. Written out because the collision of these two
/// independently-chosen constants is what makes the refusal reachable at all.
const SATURATED_BUCKET: u8 = 240;

// ───────────────────────── stub control plane ─────────────────────────

/// Yields a routing table the test can flip mid-connection, which is the whole
/// point: the session is established while the cluster has headroom and refused
/// after it loses it.
#[derive(Clone)]
struct StubControl {
    routing: Arc<watch::Sender<RoutingTable>>,
}

impl StubControl {
    fn seeded(table: RoutingTable) -> Self {
        let (tx, _rx) = watch::channel(table);
        Self {
            routing: Arc::new(tx),
        }
    }

    fn publish(&self, table: RoutingTable) {
        self.routing.send_replace(table);
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

// ───────────────────────── stub worker (dials in) ─────────────────────────

#[derive(Clone)]
struct StubWorker {
    worker_id: WorkerId,
    gateway: GatewayInboundClient,
}

impl WorkerControl for StubWorker {
    async fn dispatch(self, _: tarpc::context::Context, req: Request) -> Accepted {
        let gateway = self.gateway.clone();
        let req_id = req.req_id;
        tokio::spawn(async move {
            let msg = ServerMessage::Response {
                corr_id: 1,
                ok: true,
                result: "tok".to_string(),
            };
            if let Ok(Control::Continue) = gateway
                .push_tokens(tarpc::context::current(), req_id, Tokens::Chunk(msg))
                .await
            {
                let _ = gateway
                    .push_tokens(tarpc::context::current(), req_id, Tokens::Eos)
                    .await;
            }
        });
        Accepted::Ok {
            worker: self.worker_id,
        }
    }

    async fn cancel(self, _: tarpc::context::Context, _req_id: ReqId) {}
    async fn set_priority(self, _: tarpc::context::Context, _req_id: ReqId, _p: Priority) {}
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
    let server = StubWorker { worker_id, gateway };
    Ok(tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(server.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    ))
}

// ───────────────────────── helpers ─────────────────────────

fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn table_with(worker_id: WorkerId, kv_bucket: u8) -> RoutingTable {
    RoutingTable {
        epoch: 1,
        workers: vec![RoutableWorker {
            id: worker_id,
            addr: "127.0.0.1:0".to_string(),
            role: Role::Decode,
            model: "stub-model".to_string(),
            health: Health::Healthy,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: kv_bucket,
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

fn turn_frame(corr_id: u32) -> WsMessage {
    let msg = ClientMessage::Ping { corr_id };
    WsMessage::Binary(rmp_serde::to_vec_named(&msg).expect("encode").into())
}

/// The next TEXT frame, or `None` if the socket ends first. Binary frames are
/// the turn's own token stream and are skipped — the gateway states its
/// failures on text (`error_json`).
async fn next_text<S>(ws: &mut S) -> Option<String>
where
    S: StreamExt<Item = tokio_tungstenite::tungstenite::Result<WsMessage>> + Unpin,
{
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), ws.next()).await {
            Ok(Some(Ok(WsMessage::Text(t)))) => return Some(t.to_string()),
            Ok(Some(Ok(WsMessage::Binary(_)))) => continue,
            Ok(Some(Ok(_))) => continue,
            // Close frame, transport error, or EOF: the socket is gone.
            Ok(Some(Err(_)) | None) => return None,
            Err(_) => return None,
        }
    }
    None
}

// ───────────────────────── the regression ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_turn_keeps_the_session_socket_open() -> Result<()> {
    let worker_id = WorkerId(7);
    let control = StubControl::seeded(table_with(worker_id, 0));
    let gw: Gateway = bind(
        GatewayConfig {
            listen: loopback(),
            worker_listen: loopback(),
            controller: String::new(),
        },
        control.clone(),
    )
    .await?;

    let _worker = spawn_stub_worker(gw.worker_addr, worker_id).await?;
    wait_for_connected(&gw.state.workers.connected_watch(), worker_id).await?;

    let listen_addr = gw.listen_addr;
    let handle = gw.into_handle();

    let mut req = format!("ws://{listen_addr}/v1/ws").into_client_request()?;
    req.headers_mut()
        .insert("x-pie-identity", "t/u".parse().unwrap());
    let (mut ws, _) = tokio_tungstenite::connect_async(req).await?;

    // 1. Open the session while the cluster has headroom. This turn is the one
    //    that mints the session (and, in production, the warm KV worth keeping).
    ws.send(turn_frame(1)).await?;
    let done = next_text(&mut ws).await;
    assert_eq!(
        done.as_deref(),
        Some(r#"{"type":"turn_done"}"#),
        "the first turn must complete cleanly so a session exists to defend"
    );

    // 2. The cluster loses headroom — a single queued allocation is enough in
    //    production, since the planner clamps the bucket to exactly this value.
    control.publish(table_with(worker_id, SATURATED_BUCKET));

    // 3. The refusal must arrive as an error the client can read...
    let mut refusal = None;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut corr = 2;
    while Instant::now() < deadline {
        ws.send(turn_frame(corr)).await?;
        corr += 1;
        match next_text(&mut ws).await {
            Some(t) if t.contains("\"error\"") => {
                refusal = Some(t);
                break;
            }
            // The table update has not propagated to the routing watch yet.
            Some(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            None => bail!("socket closed instead of reporting the refusal"),
        }
    }
    let refusal = refusal.expect("admission must refuse once the cluster is saturated");
    assert!(
        refusal.contains("saturated"),
        "the refusal must say why, so the client is not left with a bare close: {refusal}"
    );

    // 4. ...and the session must SURVIVE it. This is the regression: `break`
    //    here closed the socket, which tore down the process and every working
    //    set it retained. A second refusal proves the socket is still live.
    ws.send(turn_frame(corr)).await?;
    let after = next_text(&mut ws).await;
    assert!(
        after.is_some_and(|t| t.contains("\"error\"")),
        "the socket must still be usable after a refusal -- closing it destroys \
         the session's retained KV to report a transient condition"
    );

    // 5. And it must recover: headroom returns, turns flow again on the SAME
    //    socket, which is the whole value of not having closed it.
    control.publish(table_with(worker_id, 0));
    let mut recovered = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        ws.send(turn_frame(corr)).await?;
        corr += 1;
        match next_text(&mut ws).await {
            Some(t) if t.contains("turn_done") => {
                recovered = true;
                break;
            }
            Some(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            None => bail!("socket closed during recovery"),
        }
    }
    assert!(
        recovered,
        "the session must serve turns again once the cluster has headroom"
    );

    handle.shutdown().await;
    Ok(())
}
