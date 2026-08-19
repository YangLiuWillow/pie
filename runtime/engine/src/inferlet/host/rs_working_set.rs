//! `pie:inferlet/working-set` — RS working-set host resource.
//!
//! The WASM resource type is [`crate::store::rs::working_set::RsWorkingSet`],
//! a thin handle (model, driver, RsWorkingSetId, cached geometry); every
//! substantive operation delegates to the per-(model, driver) [`RsStore`]
//! resolved through `store::registry`.

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use crate::inferlet::host::pipeline::Pipeline;
use crate::store::registry as store_registry;
use crate::store::rs::RsGeometry;
use crate::store::rs::working_set::RsWorkingSet;

type WitRange = pie::inferlet::working_set::PageRange;

impl pie::inferlet::working_set::HostRsWorkingSet for ProcessCtx {
    /// Fresh, empty RS working set bound to the single bound model (model 0),
    /// driver 0. Geometry comes from the model's RS caps (0/0/1 for
    /// pure-attention models).
    async fn new(&mut self) -> Result<Resource<RsWorkingSet>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let model = 0;
        let caps = crate::model::model().rs_caps();
        let geom = RsGeometry {
            state_size: caps.state_size,
            buffer_page_tokens: caps.buffer_page_size,
            fold_granularity: caps.fold_granularity,
        };
        let stores = store_registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom);
        let ws = RsWorkingSet::new(model, 0, id, geom);
        self.register_rs_working_set(model, 0, id);
        Ok(self.ctx().table.push(ws)?)
    }


    async fn buffer_size(&mut self, this: Resource<RsWorkingSet>) -> Result<u32> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let size = stores.rs.lock().unwrap().buffer_size(ws.id);
        Ok(size.map_err(anyhow::Error::from)?)
    }


    async fn alloc_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        n: u32,
    ) -> Result<Result<WitRange, String>> {
        // Strict admission: RS buffer slots are scarce pooled resources.
        crate::inferlet::process::ensure_bind_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let range = stores.rs.lock().unwrap().alloc_buffer(ws.id, n);
        Ok(range
            .map(|r| WitRange {
                start: r.start,
                len: r.len,
            })
            .map_err(|e| e.to_string()))
    }

    async fn free_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        indices: Vec<u32>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let mut rs = stores.rs.lock().unwrap();
        let epoch = rs.current_epoch();
        let out = rs
            .free_buffer(ws.id, &indices, epoch)
            .map_err(|e| e.to_string());
        rs.retire_idle();
        Ok(out)
    }

    async fn discard_buffered(
        &mut self,
        this: Resource<RsWorkingSet>,
        count: u32,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let out = stores
            .rs
            .lock()
            .unwrap()
            .discard_buffered(ws.id, count)
            .map_err(|e| e.to_string());
        Ok(out)
    }

    async fn reorder_buffer(
        &mut self,
        this: Resource<RsWorkingSet>,
        perm: Vec<u32>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let ws = self.ctx().table.get(&this)?.clone();
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let out = stores
            .rs
            .lock()
            .unwrap()
            .reorder_buffer(ws.id, &perm)
            .map_err(|e| e.to_string());
        Ok(out)
    }

    async fn fork(
        &mut self,
        this: Resource<RsWorkingSet>,
        on: Resource<Pipeline>,
    ) -> Result<Result<Resource<RsWorkingSet>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        // No drain: RS mappings publish at prepare, in submission order, so
        // the committed mapping already carries every fire submitted on `on`
        // before this call. Fork therefore snapshots exactly the state the
        // guest's submission order asks for, and the shared slot's CONTENTS
        // are ordered by the stream — a later CoW copy is issued behind the
        // fires that wrote the parent. `KvWorkingSet::fork` takes the same
        // lock-only path for the same reason. The pipeline argument remains
        // the scope and ordering declaration.
        let (failure, scope) = {
            let pipeline = self.ctx().table.get(&on)?;
            (pipeline.failure.clone(), pipeline.scope.clone())
        };
        let ws = self.ctx().table.get(&this)?.clone();
        if let Err(owner) = ws.claim_pipeline_scope(&scope) {
            return Ok(Err(format!(
                "rs working set fork: parent is scoped to pipeline {owner:#x}, \
                 not supplied pipeline {:#x}",
                scope.id()
            )));
        }
        if let Some(reason) = failure.lock().unwrap().clone() {
            return Ok(Err(format!(
                "rs working set fork: pipeline failed: {reason}"
            )));
        }

        let stores = store_registry::get(ws.model, ws.driver as usize);
        let forked = stores.rs.lock().unwrap().fork(ws.id);
        match forked {
            Ok(id) => {
                // A distinct working-set id gets its OWN fresh lifecycle —
                // never a clone of the parent's (see the KV working-set
                // `fork`/`slice` for the identical rationale).
                let child = RsWorkingSet::new(ws.model, ws.driver, id, ws.geom);
                self.register_rs_working_set(ws.model, ws.driver, id);
                Ok(Ok(self.ctx().table.push(child)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    /// Park this working set's fold under `key`. Same scope/failure
    /// discipline as `fork` (the store-level park IS a fork), same no-drain
    /// rationale: RS mappings publish at prepare, in submission order.
    async fn update_index(
        &mut self,
        this: Resource<RsWorkingSet>,
        on: Resource<Pipeline>,
        key: Vec<u8>,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let (failure, scope) = {
            let pipeline = self.ctx().table.get(&on)?;
            (pipeline.failure.clone(), pipeline.scope.clone())
        };
        let ws = self.ctx().table.get(&this)?.clone();
        if let Err(owner) = ws.claim_pipeline_scope(&scope) {
            return Ok(Err(format!(
                "rs working set update-index: parent is scoped to pipeline {owner:#x}, \
                 not supplied pipeline {:#x}",
                scope.id()
            )));
        }
        if let Some(reason) = failure.lock().unwrap().clone() {
            return Ok(Err(format!(
                "rs working set update-index: pipeline failed: {reason}"
            )));
        }
        let stores = store_registry::get(ws.model, ws.driver as usize);
        let parked = stores.rs.lock().unwrap().update_index(key, ws.id);
        match parked {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    /// Fork the parked fold back out (see the KV twin for the model-0
    /// convention on statics). The caller owns the result; the entry keeps
    /// its own fork.
    async fn from_index(
        &mut self,
        key: Vec<u8>,
    ) -> Result<Result<Option<Resource<RsWorkingSet>>, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let model = 0;
        let caps = crate::model::model().rs_caps();
        let geom = RsGeometry {
            state_size: caps.state_size,
            buffer_page_tokens: caps.buffer_page_size,
            fold_granularity: caps.fold_granularity,
        };
        let stores = store_registry::get(model, 0);
        let resumed = stores.rs.lock().unwrap().from_index(&key);
        match resumed {
            Ok(Some(id)) => {
                let ws = RsWorkingSet::new(model, 0, id, geom);
                self.register_rs_working_set(model, 0, id);
                Ok(Ok(Some(self.ctx().table.push(ws)?)))
            }
            Ok(None) => Ok(Ok(None)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn remove_index(&mut self, key: Vec<u8>) -> Result<Result<bool, String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let stores = store_registry::get(0, 0);
        let removed = stores.rs.lock().unwrap().remove_index(&key);
        Ok(Ok(removed))
    }

    async fn drop(&mut self, this: Resource<RsWorkingSet>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        // `release` performs the exact `release_working_set` /
        // `retire_idle` sequence and marks the shared lifecycle done; `ws`'s
        // own drop just below (and the fallback `RsLifecycle::drop` it would
        // otherwise trigger) is then a no-op.
        let ws = self.ctx().table.delete(this)?;
        self.unregister_rs_working_set(ws.model, ws.driver, ws.id);
        ws.release();
        Ok(())
    }
}
