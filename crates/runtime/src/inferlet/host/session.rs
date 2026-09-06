//! pie:core/session - User <-> Process remote communication.
//!
//! `receive`/`receive-file` are now native `async func`: the host awaits the
//! next message/file directly (no fabricated future resource). Per the
//! wasmtime-46 component-model-async model, async-func imports are generated on
//! the `HostWithStore` trait taking an `Accessor` rather than `&mut self`.

use crate::inferlet::host::frames::{Frames, Pcm, audio_extension, image_extension};
use crate::inferlet::host::pie;
use crate::inferlet::host::pie::inferlet::frames::{AudioFormat, ImageFormat};
use crate::inferlet::process;
use crate::inferlet::{ProcessCtx, ProcessEvent};
use crate::server;
use anyhow::{Context, Result};
use wasmtime::component::{Accessor, HasSelf, Resource};
use wasmtime_wasi::WasiView;

/// Give a suggested file name an extension when the guest left it bare, and
/// keep it a NAME rather than a path: the client writes it into a directory it
/// chose, and a guest that says `../../etc/passwd` gets `passwd`.
///
/// The sanitisation is repeated client-side (`pie run -o` does its own), on
/// purpose: this one is a courtesy so the wire carries something sane, that
/// one is the check that matters, because the wire is not trusted.
fn suggested_name(name: &str, extension: &str) -> String {
    let base = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('.');
    let base = if base.is_empty() { "output" } else { base };
    if std::path::Path::new(base)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(extension))
    {
        base.to_string()
    } else {
        format!("{base}.{extension}")
    }
}

/// The shared tail of `send-frames` and `send-pcm`: hand the encoded bytes to
/// the client without them ever having been a guest value.
async fn stream_out(ctx: &mut ProcessCtx, bytes: Vec<u8>, name: String) -> Result<()> {
    let process_id = ctx.id();
    if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
        server::send_file(client_id, process_id, bytes.into(), Some(name))?;
    }
    Ok(())
}

impl pie::inferlet::session::Host for ProcessCtx {
    async fn send(&mut self, message: String) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await
            && let Err(err) =
                server::send_event(client_id, process_id, &ProcessEvent::Message(message))
        {
            tracing::warn!(
                client_id,
                process_id = %process_id,
                error = %err,
                "session.send delivery failed"
            );
        }
        Ok(())
    }

    async fn send_file(&mut self, data: Vec<u8>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into(), None)?;
        }
        Ok(())
    }

    /// Encode a `frames` handle and stream it out. The encoded bytes are
    /// built here and moved into the transport; they are never lowered into
    /// the guest's linear memory, which is the whole difference between this
    /// and `send-file(f.encode(format))`.
    async fn send_frames(
        &mut self,
        f: Resource<Frames>,
        format: ImageFormat,
        name: String,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let encoded = match self.ctx().table.get(&f)?.encode(format) {
            Ok(bytes) => bytes,
            Err(e) => return Ok(Err(e)),
        };
        let name = suggested_name(&name, image_extension(format));
        stream_out(self, encoded, name).await?;
        Ok(Ok(()))
    }

    /// `send-frames` for audio.
    async fn send_pcm(
        &mut self,
        p: Resource<Pcm>,
        format: AudioFormat,
        name: String,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let encoded = self.ctx().table.get(&p)?.encode(format);
        let name = suggested_name(&name, audio_extension(format));
        stream_out(self, encoded, name).await?;
        Ok(Ok(()))
    }

    /// `receive` for a guest that cannot lower an `async func` (see
    /// `channel.take-blocking`): the guest's task is suspended in the call.
    async fn receive_blocking(&mut self) -> Result<Option<String>> {
        let process_id = self.id();
        crate::server::inbox::receive(process_id.to_string())
            .await
            .with_context(|| format!("session.receive-blocking failed for process {process_id}"))
            .map(Some)
    }

    /// `receive-file` for the same guests.
    async fn receive_file_blocking(&mut self) -> Result<Option<Vec<u8>>> {
        let process_id = self.id();
        let Some(client_id) = process::get_client_id(process_id).await.ok().flatten() else {
            return Ok(None);
        };
        match crate::server::receive_file(client_id, process_id).await {
            Ok(data) => Ok(Some(data.to_vec())),
            Err(error) => {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    %error,
                    "session.receive_file_blocking delivery failed"
                );
                Ok(None)
            }
        }
    }
}

impl pie::inferlet::session::HostWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    async fn receive(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<String>> {
        let process_id = accessor.with(|mut access| access.get().id());
        // A plain await: an idle receive holds no pooled state, so the
        // planner can evict around it freely — no standing safe point
        // needed. The residency gate re-engages on the next pooled call.
        crate::server::inbox::receive(process_id.to_string())
            .await
            .with_context(|| format!("session.receive failed for process {process_id}"))
            .map(Some)
    }

    async fn receive_file(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<Vec<u8>>> {
        let process_id = accessor.with(|mut access| access.get().id());
        let Some(client_id) = process::get_client_id(process_id).await.ok().flatten() else {
            return Ok(None);
        };
        match crate::server::receive_file(client_id, process_id).await {
            Ok(data) => Ok(Some(data.to_vec())),
            Err(error) => {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    %error,
                    "session.receive_file delivery failed"
                );
                Ok(None)
            }
        }
    }
}
