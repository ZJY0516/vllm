// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::block_pool::{BlockPool, CacheKey};

#[derive(Default)]
struct RequestState {
    full_blocks: Vec<u32>,
    mamba_blocks: Vec<u32>,
    full_cached_blocks: usize,
    mamba_cached_blocks: usize,
    mamba_allocated: bool,
    last_state_block_idx: Option<usize>,
}

/// Owns a shared block pool and request tables for one FullAttention/Mamba pair.
#[pyclass(module = "vllm._rust_kv_cache")]
pub(crate) struct HybridMambaKVCacheManager {
    block_size: usize,
    enable_caching: bool,
    full_group_id: usize,
    mamba_group_id: usize,
    pool: BlockPool,
    requests: FxHashMap<String, RequestState>,
    mamba_cached_this_step: FxHashSet<CacheKey>,
    new_full_block_ids: Vec<u32>,
}

impl HybridMambaKVCacheManager {
    fn split_groups<'a, T>(&self, groups: &'a [Vec<T>]) -> PyResult<(&'a [T], &'a [T])> {
        if groups.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "the hybrid manager requires two block groups, got {}",
                groups.len()
            )));
        }
        Ok((
            groups[self.full_group_id].as_slice(),
            groups[self.mamba_group_id].as_slice(),
        ))
    }

    fn order_groups<T>(&self, full: Vec<T>, mamba: Vec<T>) -> Vec<Vec<T>> {
        let mut groups = vec![Vec::new(), Vec::new()];
        groups[self.full_group_id] = full;
        groups[self.mamba_group_id] = mamba;
        groups
    }

    fn check_hash_count(&self, num_hashes: usize, num_tokens: usize) -> PyResult<()> {
        let num_full_blocks = num_tokens / self.block_size;
        if num_hashes < num_full_blocks {
            return Err(PyValueError::new_err(format!(
                "{num_full_blocks} full blocks are cacheable, but only {} hashes exist",
                num_hashes
            )));
        }
        Ok(())
    }

    fn blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        computed_groups: &[Vec<u32>],
    ) -> PyResult<usize> {
        let (computed_full, computed_mamba) = self.split_groups(computed_groups)?;
        let full_required = num_tokens.div_ceil(self.block_size);
        let mamba_required = num_tokens_main_model.div_ceil(self.block_size);
        if let Some(state) = self.requests.get(request_id) {
            if computed_groups.iter().any(|blocks| !blocks.is_empty()) {
                return Err(PyAssertionError::new_err(
                    "a running request cannot add prefix-cache hits",
                ));
            }
            let full_new = full_required.saturating_sub(state.full_blocks.len());
            let mamba_new = usize::from(mamba_required > state.mamba_blocks.len());
            return Ok(full_new + mamba_new);
        }

        if let Some(&block_id) = computed_mamba.iter().rfind(|&&block_id| block_id != 0)
            && let Some(cache_key) = self.pool.cache_key(block_id)
            && self.mamba_cached_this_step.contains(cache_key)
        {
            return Ok(self.pool.num_blocks() + 1);
        }

        let full_new = full_required.saturating_sub(computed_full.len());
        let mamba_new = usize::from(mamba_required > computed_mamba.len());
        Ok(full_new
            + mamba_new
            + self.pool.count_evictable(computed_full)?
            + self.pool.count_evictable(computed_mamba)?)
    }

    fn remove_skipped(&mut self, request_id: &str, processed_tokens: usize) -> PyResult<()> {
        let Some(state) = self.requests.get_mut(request_id) else {
            return Ok(());
        };
        let Some(block_idx) = state.last_state_block_idx else {
            return Ok(());
        };
        let first_required_block = processed_tokens.div_ceil(self.block_size).saturating_sub(1);
        if block_idx >= first_required_block || block_idx >= state.mamba_blocks.len() {
            return Ok(());
        }
        let block_id = state.mamba_blocks[block_idx];
        if block_id != 0 {
            state.mamba_blocks[block_idx] = 0;
            self.pool.release(std::iter::once(block_id))?;
        }
        Ok(())
    }

    fn allocate_request_blocks(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        let full_required = num_tokens.div_ceil(self.block_size);
        let mamba_required = num_tokens_main_model.div_ceil(self.block_size);

        let full_current = self.requests[request_id].full_blocks.len();
        let mut new_full = Vec::with_capacity(full_required.saturating_sub(full_current));
        for _ in full_current..full_required {
            new_full.push(self.pool.allocate()?);
        }
        self.new_full_block_ids.extend_from_slice(&new_full);
        self.requests
            .get_mut(request_id)
            .expect("request state exists")
            .full_blocks
            .extend_from_slice(&new_full);

        let state = self.requests.get_mut(request_id).expect("request state exists");
        if mamba_required <= state.mamba_blocks.len() {
            state.mamba_allocated = true;
            return Ok(self.order_groups(new_full, Vec::new()));
        }

        let previous_len = state.mamba_blocks.len();
        if state.mamba_allocated {
            state.last_state_block_idx = previous_len.checked_sub(1);
        } else if previous_len > 0 {
            state.last_state_block_idx = Some(previous_len - 1);
        }
        let num_skipped_blocks = mamba_required.saturating_sub(1);
        state.mamba_blocks.resize(num_skipped_blocks, 0);
        let new_mamba_block = self.pool.allocate()?;
        state.mamba_blocks.push(new_mamba_block);
        state.mamba_allocated = true;
        let returned_mamba = state.mamba_blocks[previous_len..].to_vec();
        Ok(self.order_groups(new_full, returned_mamba))
    }

    fn cache_request_blocks(
        &mut self,
        request_id: &str,
        block_hashes: &Bound<'_, PyList>,
        num_tokens: usize,
    ) -> PyResult<()> {
        if !self.enable_caching {
            return Ok(());
        }
        self.check_hash_count(block_hashes.len(), num_tokens)?;
        let num_full_blocks = num_tokens / self.block_size;
        let state = self.requests.get(request_id).ok_or_else(|| {
            PyKeyError::new_err(format!("request {request_id:?} has no allocated blocks"))
        })?;
        let full_start = state.full_cached_blocks;
        let mamba_start = state.mamba_cached_blocks;
        if full_start >= num_full_blocks && mamba_start >= num_full_blocks {
            return Ok(());
        }
        if state.full_blocks.len() < num_full_blocks || state.mamba_blocks.len() < num_full_blocks {
            return Err(PyAssertionError::new_err(format!(
                "request {request_id:?} does not have {num_full_blocks} cacheable blocks"
            )));
        }
        let full_blocks = if full_start < num_full_blocks {
            state.full_blocks[full_start..num_full_blocks].to_vec()
        } else {
            Vec::new()
        };
        let mamba_blocks = if mamba_start < num_full_blocks {
            state.mamba_blocks[mamba_start..num_full_blocks].to_vec()
        } else {
            Vec::new()
        };

        for (index, block_id) in (full_start..).zip(full_blocks) {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            self.pool.cache(
                block_id,
                CacheKey::new(block_hash, self.full_group_id),
                (index + 1) * self.block_size,
            )?;
        }
        for (index, block_id) in (mamba_start..).zip(mamba_blocks) {
            if block_id == 0 {
                continue;
            }
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let cache_key = CacheKey::new(block_hash, self.mamba_group_id);
            self.pool.cache(block_id, cache_key.clone(), (index + 1) * self.block_size)?;
            self.mamba_cached_this_step.insert(cache_key);
        }
        let state = self.requests.get_mut(request_id).expect("request state exists");
        state.full_cached_blocks = state.full_cached_blocks.max(num_full_blocks);
        state.mamba_cached_blocks = state.mamba_cached_blocks.max(num_full_blocks);
        Ok(())
    }

    fn get_request(&self, request_id: &str) -> Option<&RequestState> {
        self.requests.get(request_id)
    }
}

#[pymethods]
impl HybridMambaKVCacheManager {
    #[new]
    fn new(
        num_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        full_group_id: usize,
        mamba_group_id: usize,
    ) -> PyResult<Self> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be positive"));
        }
        if full_group_id > 1 || mamba_group_id > 1 || full_group_id == mamba_group_id {
            return Err(PyValueError::new_err(
                "full_group_id and mamba_group_id must identify two distinct groups",
            ));
        }
        Ok(Self {
            block_size,
            enable_caching,
            full_group_id,
            mamba_group_id,
            pool: BlockPool::new(num_blocks, enable_caching)?,
            requests: FxHashMap::default(),
            mamba_cached_this_step: FxHashSet::default(),
            new_full_block_ids: Vec::new(),
        })
    }

    fn find_longest_cache_hit(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize, usize)> {
        if !self.enable_caching {
            return Ok((self.order_groups(Vec::new(), Vec::new()), 0, 0));
        }
        let max_blocks = (max_cache_hit_length / self.block_size).min(block_hashes.len()?);
        let mut full_blocks = Vec::with_capacity(max_blocks);
        for index in 0..max_blocks {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let Some(block_id) = self.pool.find_cached(block_hash.clone(), self.full_group_id)
            else {
                break;
            };
            full_blocks.push(block_id);
        }
        let full_hit_tokens = full_blocks.len() * self.block_size;
        for index in (0..full_blocks.len()).rev() {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            if let Some(block_id) = self.pool.find_cached(block_hash, self.mamba_group_id) {
                full_blocks.truncate(index + 1);
                let mut mamba_blocks = vec![0; index];
                mamba_blocks.push(block_id);
                return Ok((
                    self.order_groups(full_blocks, mamba_blocks),
                    (index + 1) * self.block_size,
                    full_hit_tokens - (index + 1) * self.block_size,
                ));
            }
        }
        Ok((
            self.order_groups(Vec::new(), Vec::new()),
            0,
            full_hit_tokens,
        ))
    }

    #[pyo3(signature = (
        request_id,
        num_tokens,
        num_tokens_main_model,
        computed_groups,
        block_hashes,
        num_tokens_to_cache,
        processed_computed_tokens,
        reserved_blocks=0,
        watermark_blocks=0,
        full_num_tokens=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn allocate_slots(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        computed_groups: Vec<Vec<u32>>,
        block_hashes: &Bound<'_, PyList>,
        num_tokens_to_cache: usize,
        processed_computed_tokens: usize,
        reserved_blocks: usize,
        watermark_blocks: usize,
        full_num_tokens: Option<usize>,
    ) -> PyResult<Option<Vec<Vec<u32>>>> {
        self.check_hash_count(block_hashes.len(), num_tokens_to_cache)?;
        if let Some(full_num_tokens) = full_num_tokens {
            let full_required = self.blocks_to_allocate(
                request_id,
                full_num_tokens,
                full_num_tokens,
                &computed_groups,
            )?;
            if full_required + reserved_blocks + watermark_blocks > self.pool.num_free_blocks() {
                return Ok(None);
            }
        }

        self.remove_skipped(request_id, processed_computed_tokens)?;
        let blocks_to_allocate = self.blocks_to_allocate(
            request_id,
            num_tokens,
            num_tokens_main_model,
            &computed_groups,
        )?;
        if blocks_to_allocate + reserved_blocks + watermark_blocks > self.pool.num_free_blocks() {
            return Ok(None);
        }

        if !self.requests.contains_key(request_id) {
            let (computed_full, computed_mamba) = self.split_groups(&computed_groups)?;
            self.pool.touch(computed_full)?;
            self.pool.touch(computed_mamba)?;
            let state = RequestState {
                full_blocks: computed_full.to_vec(),
                mamba_blocks: computed_mamba.to_vec(),
                full_cached_blocks: computed_full.len(),
                mamba_cached_blocks: computed_mamba.len(),
                ..RequestState::default()
            };
            self.requests.insert(request_id.to_owned(), state);
        }

        let new_blocks =
            self.allocate_request_blocks(request_id, num_tokens, num_tokens_main_model)?;
        self.cache_request_blocks(request_id, block_hashes, num_tokens_to_cache)?;
        Ok(Some(new_blocks))
    }

    fn cache_blocks(
        &mut self,
        request_id: &str,
        block_hashes: &Bound<'_, PyList>,
        num_tokens: usize,
    ) -> PyResult<()> {
        self.cache_request_blocks(request_id, block_hashes, num_tokens)
    }

    fn remove_skipped_blocks(
        &mut self,
        request_id: &str,
        processed_computed_tokens: usize,
    ) -> PyResult<()> {
        self.remove_skipped(request_id, processed_computed_tokens)
    }

    fn free(&mut self, request_id: &str) -> PyResult<()> {
        let Some(state) = self.requests.remove(request_id) else {
            return Ok(());
        };
        if self.full_group_id < self.mamba_group_id {
            self.pool.release(state.full_blocks.into_iter().rev())?;
            self.pool.release(state.mamba_blocks.into_iter().rev())
        } else {
            self.pool.release(state.mamba_blocks.into_iter().rev())?;
            self.pool.release(state.full_blocks.into_iter().rev())
        }
    }

    fn get_block_ids(&self, request_id: &str) -> Vec<Vec<u32>> {
        let Some(state) = self.get_request(request_id) else {
            return self.order_groups(Vec::new(), Vec::new());
        };
        self.order_groups(state.full_blocks.clone(), state.mamba_blocks.clone())
    }

    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        computed_groups: Vec<Vec<u32>>,
    ) -> PyResult<usize> {
        self.blocks_to_allocate(
            request_id,
            num_tokens,
            num_tokens_main_model,
            &computed_groups,
        )
    }

    fn get_num_common_prefix_blocks(&self, running_request_id: &str) -> PyResult<Vec<usize>> {
        let state = self.get_request(running_request_id).ok_or_else(|| {
            PyKeyError::new_err(format!("request {running_request_id:?} has no blocks"))
        })?;
        let request_count = self.requests.len() as u32;
        let full_common = state
            .full_blocks
            .iter()
            .take_while(|&&block_id| self.pool.ref_count(block_id) == request_count)
            .count();
        Ok(self.order_groups(vec![full_common], vec![0]).into_iter().flatten().collect())
    }

    fn estimate_cached_tokens(&self, request_id: &str) -> usize {
        let Some(state) = self.get_request(request_id) else {
            return 0;
        };
        let full_cached = state
            .full_blocks
            .iter()
            .map(|&block_id| self.pool.hash_num_tokens(block_id))
            .max()
            .unwrap_or(0);
        let mamba_cached = state
            .mamba_blocks
            .iter()
            .filter(|&&block_id| block_id != 0)
            .map(|&block_id| self.pool.hash_num_tokens(block_id))
            .max()
            .unwrap_or(0);
        full_cached.min(mamba_cached)
    }

    fn evict_blocks(&mut self, block_ids: Vec<u32>) -> PyResult<()> {
        self.pool.evict(block_ids)
    }

    fn reset_prefix_cache(&mut self) -> bool {
        let reset = self.pool.reset_prefix_cache();
        if reset {
            self.mamba_cached_this_step.clear();
        }
        reset
    }

    fn new_step_starts(&mut self) {
        self.mamba_cached_this_step.clear();
    }

    fn take_new_block_ids(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.new_full_block_ids)
    }

    fn get_zeroing_block_ids_in_range(
        &self,
        request_id: &str,
        start_token: usize,
        end_token: usize,
    ) -> Vec<u32> {
        let Some(state) = self.get_request(request_id) else {
            return Vec::new();
        };
        let start_block = start_token / self.block_size;
        let end_block = end_token.div_ceil(self.block_size);
        state.full_blocks
            [start_block.min(state.full_blocks.len())..end_block.min(state.full_blocks.len())]
            .to_vec()
    }

    fn record_blocks_for_zeroing(&mut self, request_id: &str, start_token: usize) -> PyResult<()> {
        if !start_token.is_multiple_of(self.block_size) {
            return Err(PyValueError::new_err(
                "start_token must be block aligned for KV cache zeroing",
            ));
        }
        let state = self
            .get_request(request_id)
            .ok_or_else(|| PyKeyError::new_err(format!("request {request_id:?} has no blocks")))?;
        let start_block = start_token / self.block_size;
        let block_ids = state.full_blocks[start_block.min(state.full_blocks.len())..].to_vec();
        self.new_full_block_ids.extend(block_ids);
        Ok(())
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.pool.num_free_blocks()
    }

    #[getter]
    fn usage(&self) -> f64 {
        self.pool.usage()
    }
}
