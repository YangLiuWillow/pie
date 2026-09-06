//! WebSocket adapter: a multi-turn interactive session, lifetime the
//! connection. The general case; `http.rs` is its 1-turn degenerate form.
//!
//! One `select!` multiplexes both directions: worker->user drains every live
//! turn's `TokenRx` to WS frames, user->worker reads the next client turn or
//! a `cancel`. Turns are concurrent, not one-at-a-time — a client may launch
//! several processes on one socket, so every stream is kept until it ends on
//! its own rather than being dropped when a second turn opens.

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};

use crate::GatewayState;
use crate::ingress::identity;
use crate::session::{Affinity, Identity, TokenRx, TurnInput};
use crate::worker::{MAX_CLIENT_FRAME_BYTES, MAX_CLIENT_FRAME_RECV_BYTES};
use client_api::ClientMessage;
use worker_api::{Priority, Tokens};

/// `GET /v1/ws` — upgrade to a multi-turn session. Identity is extracted from the
/// edge headers *before* the upgrade (a bad edge never gets a socket).
pub async fn ws(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let ident = match identity::extract(&headers) {
        Ok(id) => id,
        Err(e) => return (StatusCode::UNAUTHORIZED, format!("identity: {e}")).into_response(),
    };
    // Bound what the transport will accept BEFORE a frame can become a
    // `dispatch`. axum's defaults (64 MiB message / 16 MiB frame) sit above the
    // worker link's own cap, so an oversized turn used to reach the codec and
    // break the link for every other session on this server.
    upgrade
        .max_message_size(MAX_CLIENT_FRAME_RECV_BYTES)
        .max_frame_size(MAX_CLIENT_FRAME_RECV_BYTES)
        .on_upgrade(move |socket| serve(socket, state, ident))
}

/// Is this frame within what a turn may carry? The check is on the RAW bytes,
/// before any parse, because the point is to answer without ever building
/// something that cannot be dispatched.
fn too_large(len: usize) -> Option<String> {
    (len > MAX_CLIENT_FRAME_BYTES).then(|| {
        format!(
            "request too large: {len} bytes, limit {MAX_CLIENT_FRAME_BYTES} \
             (send large inputs as a blob reference, not inline)"
        )
    })
}

/// One parsed client frame: a new turn, or a cancel of the live turns.
enum Incoming {
    Turn(TurnInput),
    Cancel,
}

/// Every turn currently streaming on this socket. A turn is removed when its
/// stream ends (clean `Eos` or a mid-stream abort); the set is empty between
/// turns, which is what parks the token arm.
struct LiveTurns {
    streams: Vec<TokenRx>,
    /// Where the next poll starts. Always polling from index 0 would starve
    /// later turns whenever an earlier one has a token buffered; advancing
    /// this makes the poll order round-robin.
    cursor: usize,
}

/// What one poll of the live turn set produced.
enum TurnEvent {
    /// A stream yielded an item.
    Item(Tokens),
    /// A stream closed without `Eos` — that turn aborted mid-flight.
    Aborted,
}

impl LiveTurns {
    fn new() -> Self {
        Self {
            streams: Vec::new(),
            cursor: 0,
        }
    }

    fn push(&mut self, rx: TokenRx) {
        self.streams.push(rx);
    }

    /// Await the next item from any live turn, parking forever when there is
    /// none so the caller's `select!` waits only on the client. A stream that
    /// ends is dropped: `Eos` is the clean end, `None` without one is the
    /// abort the Tokens contract defines. Polled round-robin from `cursor`,
    /// so no turn can starve another.
    async fn next(&mut self) -> TurnEvent {
        let live = self.streams.len();
        if live == 0 {
            return std::future::pending::<TurnEvent>().await;
        }
        // Rotate the set itself rather than indexing around it: the order of
        // `streams` carries no meaning, and this keeps the borrow simple.
        self.streams.rotate_left(self.cursor % live);
        self.cursor = 0;
        // `select_all` needs a non-empty iterator, which the guard above
        // guarantees; it resolves as soon as ONE stream yields and hands back
        // the index so the finished stream can be retired.
        let (item, index, _) = futures::future::select_all(
            self.streams
                .iter_mut()
                .map(|rx| Box::pin(rx.recv()))
                .collect::<Vec<_>>(),
        )
        .await;
        // Start the next poll after the turn just served.
        self.cursor = index + 1;
        match item {
            Some(Tokens::Eos) => {
                self.streams.remove(index);
                TurnEvent::Item(Tokens::Eos)
            }
            Some(other) => TurnEvent::Item(other),
            None => {
                self.streams.remove(index);
                TurnEvent::Aborted
            }
        }
    }
}

/// Drive one WebSocket connection across many turns.
async fn serve(socket: WebSocket, state: GatewayState, ident: Identity) {
    let (mut tx, mut rx_ws) = socket.split();

    // The first client frame opens the session (its `create` mints the ReqId).
    let first = match read_first_turn(&mut rx_ws).await {
        Ok(req) => req,
        // Refused before a session exists: say why, then close. Silence here
        // would be a hang, which is the failure mode this guard exists to end.
        Err(Some(why)) => {
            let _ = tx.send(Message::Text(error_json(&why).into())).await;
            let _ = tx.send(Message::Close(None)).await;
            return;
        }
        Err(None) => return, // closed / errored before any turn
    };
    // Multi-turn: sticky affinity so every turn prefers the warm-KV worker.
    let (handle, first_rx) = match state.sessions.create(ident, first, Affinity::Sticky).await {
        Ok(pair) => pair,
        Err(e) => {
            let _ = tx
                .send(Message::Text(error_json(&e.to_string()).into()))
                .await;
            let _ = tx.send(Message::Close(None)).await;
            return;
        }
    };

    // Every turn streaming right now; empty = idle between turns.
    let mut live = LiveTurns::new();
    live.push(first_rx);

    loop {
        tokio::select! {
            ev = live.next() => match ev {
                TurnEvent::Item(Tokens::Chunk(msg)) => {
                    // `pie-client` decodes as MessagePack over binary frames
                    // (its reader drops Text). Encode failure: log + drop.
                    match encode(&msg) {
                        Some(bytes) => {
                            if tx.send(Message::Binary(bytes.into())).await.is_err() {
                                break; // user hung up
                            }
                        }
                        None => continue,
                    }
                }
                // Clean end of ONE turn: tell the client. Other turns on this
                // socket keep streaming; the socket closes only when the
                // client hangs up.
                TurnEvent::Item(Tokens::Eos) => {
                    if tx
                        .send(Message::Text(turn_done_json().into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                // Channel closed without Eos ⇒ that turn aborted mid-stream.
                // Report it and keep the session: the client's other turns are
                // unaffected, and it may submit more.
                TurnEvent::Aborted => {
                    if tx
                        .send(Message::Text(error_json("stream aborted").into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            },

            incoming = rx_ws.next() => match incoming {
                // Oversized is non-fatal and answered, like a bad frame: the
                // session survives and the client learns its own limit.
                Some(Ok(Message::Text(t))) if too_large(t.len()).is_some() => {
                    let why = too_large(t.len()).expect("guard matched");
                    if tx.send(Message::Text(error_json(&why).into())).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Binary(b))) if too_large(b.len()).is_some() => {
                    let why = too_large(b.len()).expect("guard matched");
                    if tx.send(Message::Text(error_json(&why).into())).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                    // A turn the cluster would not take (admission, no route) is
                    // that turn's failure, not the connection's: the other turns
                    // this socket carries keep streaming, and the client may
                    // retry. Closing here lost every in-flight process's events
                    // to one transient "saturated".
                    Ok(Incoming::Turn(req)) => {
                        let corr = req.message.corr_id();
                        match handle.turn(req).await {
                            Ok(new_rx) => live.push(new_rx),
                            Err(e) => {
                                if tx.send(refusal(corr, &e.to_string())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        // Bad frame is non-fatal: report and keep the session.
                        let _ = tx.send(Message::Text(error_json(&e).into())).await;
                    }
                },
                Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                    Ok(Incoming::Turn(req)) => {
                        let corr = req.message.corr_id();
                        match handle.turn(req).await {
                            Ok(new_rx) => live.push(new_rx),
                            Err(e) => {
                                if tx.send(refusal(corr, &e.to_string())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        let _ = tx.send(Message::Text(error_json(&e).into())).await;
                    }
                },
                Some(Ok(Message::Close(_))) | None => break,
                // axum auto-replies to pings; ignore control frames.
                Some(Ok(_)) => {}
                Some(Err(_)) => break,
            },
        }
    }

    handle.close().await;
}

/// Read frames until the first turn-bearing one.
///
/// `Err(Some(why))` is a refusal the caller must report before closing;
/// `Err(None)` is the client closing (or erroring) before it ever sent a turn.
async fn read_first_turn(
    rx_ws: &mut futures::stream::SplitStream<WebSocket>,
) -> Result<TurnInput, Option<String>> {
    loop {
        match rx_ws.next().await {
            // Same rule as mid-session, one turn earlier. No session exists to
            // keep alive, so this ends the connection instead of continuing it.
            Some(Ok(Message::Text(t))) if too_large(t.len()).is_some() => {
                return Err(too_large(t.len()));
            }
            Some(Ok(Message::Binary(b))) if too_large(b.len()).is_some() => {
                return Err(too_large(b.len()));
            }
            Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                Ok(Incoming::Turn(req)) => return Ok(req),
                _ => continue, // a cancel/bad frame before any turn is meaningless
            },
            Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                Ok(Incoming::Turn(req)) => return Ok(req),
                _ => continue,
            },
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return Err(None),
            Some(Ok(_)) => continue,
        }
    }
}

fn parse_incoming(text: &str) -> Result<Incoming, String> {
    if text.trim() == "cancel" {
        return Ok(Incoming::Cancel);
    }
    let payload: ClientMessage =
        serde_json::from_str(text).map_err(|e| format!("bad client frame: {e}"))?;
    Ok(Incoming::Turn(into_turn(payload)))
}

fn parse_incoming_bytes(bytes: &[u8]) -> Result<Incoming, String> {
    // `pie-client` sends `ClientMessage` as MessagePack over binary frames
    // (`rmp_serde::encode::to_vec_named`). Decode with the same codec — a JSON
    // decode here silently fails every frame and the turn never dispatches.
    let payload: ClientMessage =
        rmp_serde::from_slice(bytes).map_err(|e| format!("bad client frame: {e}"))?;
    Ok(Incoming::Turn(into_turn(payload)))
}

/// Wrap a client payload into per-turn content. `req_id`/`session` are minted
/// by `Session` in `turn`/`create`. Blob ingest (if any) attaches `blobs` here.
fn into_turn(payload: ClientMessage) -> TurnInput {
    TurnInput {
        message: payload,
        blobs: Vec::new(),
        priority: Priority::Normal,
    }
}

/// Encode a `ServerMessage` as MessagePack for a binary WS frame — the codec
/// `pie-client`'s reader expects (`rmp_serde::decode::from_slice`). `None` on
/// an encode failure (logged), so the caller drops the frame rather than
/// sending an undecodable empty binary.
fn encode(msg: &client_api::ServerMessage) -> Option<Vec<u8>> {
    match rmp_serde::to_vec_named(msg) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::error!(error = %e, "ServerMessage msgpack encode failed; dropping frame");
            None
        }
    }
}

fn turn_done_json() -> String {
    "{\"type\":\"turn_done\"}".to_string()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "type": "error", "message": msg }).to_string()
}

/// A turn the cluster would not take, answered to the call that asked. A
/// frame that carried a correlation id gets a `response { ok: false }` under
/// that id — the client's pending call fails by name, and the other calls
/// this socket multiplexes are untouched. A frame with none (or one the
/// encoder cannot serialise) gets the bare `error` text frame, which is all
/// there is to say about it.
fn refusal(corr_id: Option<u32>, why: &str) -> Message {
    match corr_id.and_then(|corr_id| {
        encode(&client_api::ServerMessage::Response {
            corr_id,
            ok: false,
            result: why.to_string(),
        })
    }) {
        Some(bytes) => Message::Binary(bytes.into()),
        None => Message::Text(error_json(why).into()),
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;

    #[test]
    fn a_correlated_call_is_refused_under_its_own_id() {
        let frame = refusal(Some(7), "cluster saturated");
        let Message::Binary(bytes) = frame else {
            panic!("a refusal of a correlated call is a binary response frame");
        };
        let decoded: client_api::ServerMessage =
            rmp_serde::from_slice(&bytes).expect("the codec pie-client reads");
        match decoded {
            client_api::ServerMessage::Response {
                corr_id,
                ok,
                result,
            } => {
                assert_eq!(corr_id, 7);
                assert!(!ok);
                assert_eq!(result, "cluster saturated");
            }
            other => panic!("expected a response frame, got {other:?}"),
        }
    }

    #[test]
    fn an_uncorrelated_frame_gets_the_bare_error() {
        let Message::Text(text) = refusal(None, "no route") else {
            panic!("a refusal with no id to answer under is the error text frame");
        };
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["message"], "no route");
    }

    #[test]
    fn every_call_that_expects_a_response_carries_its_id() {
        let ping: client_api::ClientMessage =
            serde_json::from_str(r#"{"type":"ping","corr_id":3}"#).unwrap();
        assert_eq!(ping.corr_id(), Some(3));
        let signal: client_api::ClientMessage = serde_json::from_str(
            r#"{"type":"signal_process","process_id":"p","message":"m"}"#,
        )
        .unwrap();
        assert_eq!(signal.corr_id(), None);
    }
}
