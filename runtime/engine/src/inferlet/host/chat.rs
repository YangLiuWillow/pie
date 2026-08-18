//! pie:instruct/chat — Conversation management
//!
//! Imported by inferlets that support chat-style interaction.
//! Delegates to the model's `Instruct` implementation.

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use pie_model::instruct::{ChatDecoder, ChatEvent};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Chat decoder resource — wraps a model-specific ChatDecoder trait object.
pub struct Decoder {
    inner: Box<dyn ChatDecoder>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("chat::Decoder").finish()
    }
}

impl pie::inferlet::chat::Host for ProcessCtx {
    async fn system(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().system(&message))
    }

    async fn user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().user(&message))
    }

    async fn first_user(&mut self, message: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().first_user(&message))
    }

    async fn system_user(&mut self, system: String, user: String) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().system_user(&system, &user))
    }

    async fn assistant_call(
        &mut self,
        content: Option<String>,
        calls: Vec<pie::inferlet::tools::ToolCall>,
        after_query: bool,
        is_last: bool,
    ) -> Result<Vec<u32>> {
        let pairs: Vec<(String, String)> = calls
            .into_iter()
            .map(|c| (c.name, c.arguments_json))
            .collect();
        Ok(crate::model::model().instruct().assistant_with_tool_calls_at(
            content.as_deref(),
            &pairs,
            after_query,
            is_last,
        ))
    }

    async fn assistant(
        &mut self,
        message: String,
        after_query: bool,
        is_last: bool,
    ) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().assistant_at(&message, after_query, is_last))
    }

    /// The template's `enable_thinking`, applied here rather than chosen by
    /// which function the guest called.
    async fn cue(&mut self, thinking: bool) -> Result<Vec<u32>> {
        let model = crate::model::model();
        let instruct = model.instruct();
        Ok(if thinking { instruct.cue() } else { instruct.cue_no_think() })
    }

    async fn seal(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().seal())
    }

    async fn stop_tokens(&mut self) -> Result<Vec<u32>> {
        Ok(crate::model::model().instruct().seal())
    }

}

impl pie::inferlet::chat::HostDecoder for ProcessCtx {
    async fn new(&mut self) -> Result<Resource<Decoder>> {
        let inner = crate::model::model().instruct().chat_decoder();
        let decoder = Decoder { inner };
        Ok(self.ctx().table.push(decoder)?)
    }

    async fn feed(
        &mut self,
        this: Resource<Decoder>,
        tokens: Vec<u32>,
    ) -> Result<Result<pie::inferlet::chat::Event, pie::inferlet::types::Error>> {
        let decoder = self.ctx().table.get_mut(&this)?;
        let event = decoder.inner.feed(&tokens);
        Ok(Ok(match event {
            ChatEvent::Delta(s) => pie::inferlet::chat::Event::Delta(s),
            ChatEvent::Interrupt(id) => pie::inferlet::chat::Event::Interrupt(id),
            ChatEvent::Done(s) => pie::inferlet::chat::Event::Done(s),
        }))
    }

    async fn reset(&mut self, this: Resource<Decoder>) -> Result<()> {
        let decoder = self.ctx().table.get_mut(&this)?;
        decoder.inner.reset();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Decoder>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
