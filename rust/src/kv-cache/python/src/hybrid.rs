// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::block_pool::{BlockPool, CacheKey};

#[derive(Default)]
struct RequestState {
    blocks: Vec<Vec<u32>>,
    cached_blocks: Vec<usize>,
    mamba_allocated: Vec<bool>,
    last_state_block_idx: Vec<Option<usize>>,
}

/// Owns a shared block pool and request tables for FullAttention/Mamba groups.
#[pyclass(module = "vllm._rust_kv_cache")]
pub(crate) struct HybridMambaKVCacheManager {
    block_size: usize,
    enable_caching: bool,
    group_is_mamba: Vec<bool>,
    pool: BlockPool,
    requests: FxHashMap<String, RequestState>,
    mamba_cached_this_step: FxHashSet<CacheKey>,
    new_attention_block_ids: Vec<u32>,
}

impl HybridMambaKVCacheManager {
    fn validate_groups<T>(&self, groups: &[Vec<T>]) -> PyResult<()> {
        if groups.len() != self.group_is_mamba.len() {
            return Err(PyValueError::new_err(format!(
                "the hybrid manager requires {} block groups, got {}",
                self.group_is_mamba.len(),
                groups.len(),
            )));
        }
        Ok(())
    }

    fn empty_groups<T>(&self) -> Vec<Vec<T>> {
        (0..self.group_is_mamba.len()).map(|_| Vec::new()).collect()
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
        self.validate_groups(computed_groups)?;
        let full_required = num_tokens.div_ceil(self.block_size);
        let mamba_required = num_tokens_main_model.div_ceil(self.block_size);
        if let Some(state) = self.requests.get(request_id) {
            if computed_groups.iter().any(|blocks| !blocks.is_empty()) {
                return Err(PyAssertionError::new_err(
                    "a running request cannot add prefix-cache hits",
                ));
            }
            return Ok(state
                .blocks
                .iter()
                .zip(&self.group_is_mamba)
                .map(|(blocks, &is_mamba)| {
                    if is_mamba {
                        usize::from(mamba_required > blocks.len())
                    } else {
                        full_required.saturating_sub(blocks.len())
                    }
                })
                .sum());
        }

        for (group_id, blocks) in computed_groups.iter().enumerate() {
            if self.group_is_mamba[group_id]
                && let Some(&block_id) = blocks.iter().rfind(|&&block_id| block_id != 0)
                && let Some(cache_key) = self.pool.cache_key(block_id)
                && self.mamba_cached_this_step.contains(cache_key)
            {
                return Ok(self.pool.num_blocks() + 1);
            }
        }

        let mut blocks_to_allocate = 0;
        for (group_id, blocks) in computed_groups.iter().enumerate() {
            blocks_to_allocate += if self.group_is_mamba[group_id] {
                usize::from(mamba_required > blocks.len())
            } else {
                full_required.saturating_sub(blocks.len())
            };
            blocks_to_allocate += self.pool.count_evictable(blocks)?;
        }
        Ok(blocks_to_allocate)
    }

    fn remove_skipped(&mut self, request_id: &str, processed_tokens: usize) -> PyResult<()> {
        let Some(state) = self.requests.get_mut(request_id) else {
            return Ok(());
        };
        let first_required_block = processed_tokens.div_ceil(self.block_size).saturating_sub(1);
        let mut released = Vec::new();
        for group_id in 0..self.group_is_mamba.len() {
            if !self.group_is_mamba[group_id] {
                continue;
            }
            let Some(block_idx) = state.last_state_block_idx[group_id] else {
                continue;
            };
            if block_idx >= first_required_block || block_idx >= state.blocks[group_id].len() {
                continue;
            }
            let block_id = state.blocks[group_id][block_idx];
            if block_id != 0 {
                state.blocks[group_id][block_idx] = 0;
                released.push(block_id);
            }
        }
        if !released.is_empty() {
            self.pool.release(released.into_iter())?;
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

        let mut new_groups = self.empty_groups();
        for group_id in 0..self.group_is_mamba.len() {
            if self.group_is_mamba[group_id] {
                continue;
            }
            let current = self.requests[request_id].blocks[group_id].len();
            let new_blocks = &mut new_groups[group_id];
            new_blocks.reserve(full_required.saturating_sub(current));
            for _ in current..full_required {
                new_blocks.push(self.pool.allocate()?);
            }
            self.new_attention_block_ids.extend_from_slice(new_blocks);
            self.requests.get_mut(request_id).expect("request state exists").blocks[group_id]
                .extend_from_slice(new_blocks);
        }

        for group_id in 0..self.group_is_mamba.len() {
            if !self.group_is_mamba[group_id] {
                continue;
            }
            let state = self.requests.get_mut(request_id).expect("request state exists");
            if mamba_required <= state.blocks[group_id].len() {
                state.mamba_allocated[group_id] = true;
                continue;
            }
            let previous_len = state.blocks[group_id].len();
            if state.mamba_allocated[group_id] || previous_len > 0 {
                state.last_state_block_idx[group_id] = previous_len.checked_sub(1);
            }
            state.blocks[group_id].resize(mamba_required.saturating_sub(1), 0);
            let new_mamba_block = self.pool.allocate()?;
            state.blocks[group_id].push(new_mamba_block);
            state.mamba_allocated[group_id] = true;
            new_groups[group_id] = state.blocks[group_id][previous_len..].to_vec();
        }
        Ok(new_groups)
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
        for group_id in 0..self.group_is_mamba.len() {
            let (start, blocks) = {
                let state = self.requests.get(request_id).ok_or_else(|| {
                    PyKeyError::new_err(format!("request {request_id:?} has no allocated blocks"))
                })?;
                if state.blocks[group_id].len() < num_full_blocks {
                    return Err(PyAssertionError::new_err(format!(
                        "request {request_id:?} group {group_id} does not have \
                         {num_full_blocks} cacheable blocks"
                    )));
                }
                let start = state.cached_blocks[group_id];
                if start >= num_full_blocks {
                    continue;
                }
                (
                    start,
                    state.blocks[group_id][start..num_full_blocks].to_vec(),
                )
            };
            for (index, block_id) in (start..).zip(blocks) {
                if block_id == 0 {
                    continue;
                }
                let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
                let cache_key = CacheKey::new(block_hash, group_id);
                let parent_block_id = (!self.group_is_mamba[group_id])
                    .then(|| index.checked_sub(1))
                    .flatten()
                    .map(|index| self.requests[request_id].blocks[group_id][index]);
                self.pool.cache(
                    block_id,
                    cache_key.clone(),
                    (index + 1) * self.block_size,
                    parent_block_id,
                )?;
                if self.group_is_mamba[group_id] {
                    self.mamba_cached_this_step.insert(cache_key);
                }
            }
            let cached_blocks =
                &mut self.requests.get_mut(request_id).expect("request state exists").cached_blocks
                    [group_id];
            *cached_blocks = (*cached_blocks).max(num_full_blocks);
        }
        Ok(())
    }

    fn get_request(&self, request_id: &str) -> Option<&RequestState> {
        self.requests.get(request_id)
    }

    fn find_full_attention_hits(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_blocks: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        let mut hit_groups = self.empty_groups();
        if max_blocks == 0 {
            return Ok(hit_groups);
        }

        let first_hash = block_hashes.get_item(0)?.extract::<Vec<u8>>()?;
        let mut terminal_blocks = Vec::new();
        for (group_id, &is_mamba) in self.group_is_mamba.iter().enumerate() {
            if is_mamba {
                continue;
            }
            let Some(block_id) = self.pool.find_cached(first_hash.clone(), group_id) else {
                return Ok(hit_groups);
            };
            terminal_blocks.push((group_id, block_id));
        }

        let mut low = 0;
        let mut high = max_blocks;
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            let block_hash = block_hashes.get_item(middle)?.extract::<Vec<u8>>()?;
            let mut middle_blocks = Vec::with_capacity(terminal_blocks.len());
            for &(group_id, _) in &terminal_blocks {
                let Some(block_id) = self.pool.find_cached(block_hash.clone(), group_id) else {
                    middle_blocks.clear();
                    break;
                };
                middle_blocks.push((group_id, block_id));
            }
            if middle_blocks.is_empty() {
                high = middle;
            } else {
                low = middle;
                terminal_blocks = middle_blocks;
            }
        }

        let mut paths = Vec::with_capacity(terminal_blocks.len());
        for &(group_id, block_id) in &terminal_blocks {
            let Some(path) = self.pool.find_cached_path(block_id, group_id, low + 1) else {
                return self.find_full_attention_hits_scalar(block_hashes, max_blocks);
            };
            paths.push((group_id, path));
        }
        for (group_id, path) in paths {
            hit_groups[group_id] = path;
        }
        Ok(hit_groups)
    }

    fn find_full_attention_hits_scalar(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_blocks: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        let mut hit_groups = self.empty_groups();
        for index in 0..max_blocks {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let mut block_ids = Vec::new();
            for (group_id, &is_mamba) in self.group_is_mamba.iter().enumerate() {
                if is_mamba {
                    continue;
                }
                let Some(block_id) = self.pool.find_cached(block_hash.clone(), group_id) else {
                    block_ids.clear();
                    break;
                };
                block_ids.push((group_id, block_id));
            }
            if block_ids.is_empty() {
                break;
            }
            for (group_id, block_id) in block_ids {
                hit_groups[group_id].push(block_id);
            }
        }
        Ok(hit_groups)
    }
}

#[pymethods]
impl HybridMambaKVCacheManager {
    #[new]
    fn new(
        num_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        group_is_mamba: Vec<bool>,
    ) -> PyResult<Self> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be positive"));
        }
        if group_is_mamba.len() < 2
            || !group_is_mamba.iter().any(|&is_mamba| is_mamba)
            || !group_is_mamba.iter().any(|&is_mamba| !is_mamba)
        {
            return Err(PyValueError::new_err(
                "group_is_mamba must identify at least one FullAttention and one Mamba group",
            ));
        }
        Ok(Self {
            block_size,
            enable_caching,
            group_is_mamba,
            pool: BlockPool::new(num_blocks, enable_caching)?,
            requests: FxHashMap::default(),
            mamba_cached_this_step: FxHashSet::default(),
            new_attention_block_ids: Vec::new(),
        })
    }

    fn find_longest_cache_hit(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize, usize)> {
        if !self.enable_caching {
            return Ok((self.empty_groups(), 0, 0));
        }
        let max_blocks = (max_cache_hit_length / self.block_size).min(block_hashes.len()?);
        let mut hit_groups = self.find_full_attention_hits(block_hashes, max_blocks)?;
        let full_hit_blocks = self
            .group_is_mamba
            .iter()
            .position(|&is_mamba| !is_mamba)
            .map(|group_id| hit_groups[group_id].len())
            .unwrap_or(0);
        let full_hit_tokens = full_hit_blocks * self.block_size;
        for index in (0..full_hit_blocks).rev() {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let mut mamba_block_ids = Vec::new();
            for (group_id, &is_mamba) in self.group_is_mamba.iter().enumerate() {
                if !is_mamba {
                    continue;
                }
                let Some(block_id) = self.pool.find_cached(block_hash.clone(), group_id) else {
                    mamba_block_ids.clear();
                    break;
                };
                mamba_block_ids.push((group_id, block_id));
            }
            if mamba_block_ids.is_empty() {
                continue;
            }
            for (group_id, blocks) in hit_groups.iter_mut().enumerate() {
                if !self.group_is_mamba[group_id] {
                    blocks.truncate(index + 1);
                }
            }
            for (group_id, block_id) in mamba_block_ids {
                hit_groups[group_id] = vec![0; index];
                hit_groups[group_id].push(block_id);
            }
            return Ok((
                hit_groups,
                (index + 1) * self.block_size,
                full_hit_tokens - (index + 1) * self.block_size,
            ));
        }
        Ok((self.empty_groups(), 0, full_hit_tokens))
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
            self.validate_groups(&computed_groups)?;
            for blocks in &computed_groups {
                self.pool.touch(blocks)?;
            }
            let group_count = self.group_is_mamba.len();
            let state = RequestState {
                cached_blocks: computed_groups.iter().map(Vec::len).collect(),
                blocks: computed_groups,
                mamba_allocated: vec![false; group_count],
                last_state_block_idx: vec![None; group_count],
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
        for blocks in state.blocks {
            self.pool.release(blocks.into_iter().rev())?;
        }
        Ok(())
    }

    fn get_block_ids(&self, request_id: &str) -> Vec<Vec<u32>> {
        let Some(state) = self.get_request(request_id) else {
            return self.empty_groups();
        };
        state.blocks.clone()
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
        Ok(state
            .blocks
            .iter()
            .zip(&self.group_is_mamba)
            .map(|(blocks, &is_mamba)| {
                if is_mamba {
                    0
                } else {
                    blocks
                        .iter()
                        .take_while(|&&block_id| self.pool.ref_count(block_id) == request_count)
                        .count()
                }
            })
            .collect())
    }

    fn estimate_cached_tokens(&self, request_id: &str) -> usize {
        let Some(state) = self.get_request(request_id) else {
            return 0;
        };
        state
            .blocks
            .iter()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|&&block_id| block_id != 0)
                    .map(|&block_id| self.pool.hash_num_tokens(block_id))
                    .max()
                    .unwrap_or(0)
            })
            .min()
            .unwrap_or(0)
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
        std::mem::take(&mut self.new_attention_block_ids)
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
        state
            .blocks
            .iter()
            .zip(&self.group_is_mamba)
            .filter(|(_, is_mamba)| !**is_mamba)
            .flat_map(|(blocks, _)| {
                blocks[start_block.min(blocks.len())..end_block.min(blocks.len())].to_vec()
            })
            .collect()
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
        let block_ids = state
            .blocks
            .iter()
            .zip(&self.group_is_mamba)
            .filter(|(_, is_mamba)| !**is_mamba)
            .flat_map(|(blocks, _)| blocks[start_block.min(blocks.len())..].iter().copied())
            .collect::<Vec<_>>();
        self.new_attention_block_ids.extend(block_ids);
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
