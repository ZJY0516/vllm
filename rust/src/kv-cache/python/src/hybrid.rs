// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::block_pool::{BlockPool, CacheKey};

const FULL_ATTENTION: u8 = 0;
const MAMBA_ALIGN: u8 = 1;
const SLIDING_WINDOW: u8 = 2;

#[derive(Clone, Copy, Eq, PartialEq)]
enum GroupPolicy {
    FullAttention,
    MambaAlign,
    SlidingWindow {
        window_size: usize,
        extra_retained_tokens: usize,
    },
}

impl GroupPolicy {
    fn from_descriptor(
        kind: u8,
        window_size: usize,
        extra_retained_tokens: usize,
    ) -> PyResult<Self> {
        match kind {
            FULL_ATTENTION => Ok(Self::FullAttention),
            MAMBA_ALIGN => Ok(Self::MambaAlign),
            SLIDING_WINDOW if window_size > 0 => Ok(Self::SlidingWindow {
                window_size,
                extra_retained_tokens,
            }),
            SLIDING_WINDOW => Err(PyValueError::new_err(
                "sliding-window groups require a positive window size",
            )),
            _ => Err(PyValueError::new_err(format!(
                "unknown KV cache group policy {kind}"
            ))),
        }
    }

    fn is_full_attention(self) -> bool {
        matches!(self, Self::FullAttention)
    }

    fn is_mamba(self) -> bool {
        matches!(self, Self::MambaAlign)
    }

    fn is_sliding_window(self) -> bool {
        matches!(self, Self::SlidingWindow { .. })
    }

    fn skipped_tokens(self, processed_tokens: usize) -> usize {
        match self {
            Self::SlidingWindow {
                window_size,
                extra_retained_tokens,
            } => processed_tokens
                .saturating_sub(window_size.saturating_sub(1))
                .saturating_sub(extra_retained_tokens),
            _ => 0,
        }
    }
}

#[derive(Default)]
struct RequestState {
    blocks: Vec<Vec<u32>>,
    cached_blocks: Vec<usize>,
    released_prefix_blocks: Vec<usize>,
    mamba_allocated: Vec<bool>,
    last_state_block_idx: Vec<Option<usize>>,
}

/// Owns a shared block pool and request tables for heterogeneous cache groups.
#[pyclass(module = "vllm._rust_kv_cache")]
pub(crate) struct HybridKVCacheManager {
    scheduler_block_size: usize,
    hash_block_size: usize,
    enable_caching: bool,
    group_policies: Vec<GroupPolicy>,
    group_block_sizes: Vec<usize>,
    eagle_groups: Vec<bool>,
    max_admission_blocks: Vec<usize>,
    pool: BlockPool,
    requests: FxHashMap<String, RequestState>,
    pending_hits: FxHashMap<String, Vec<Vec<u32>>>,
    mamba_cached_this_step: FxHashSet<CacheKey>,
    new_attention_block_ids: Vec<u32>,
}

impl HybridKVCacheManager {
    fn validate_groups<T>(&self, groups: &[Vec<T>]) -> PyResult<()> {
        if groups.len() != self.group_policies.len() {
            return Err(PyValueError::new_err(format!(
                "the hybrid manager requires {} block groups, got {}",
                self.group_policies.len(),
                groups.len(),
            )));
        }
        Ok(())
    }

    fn empty_groups<T>(&self) -> Vec<Vec<T>> {
        (0..self.group_policies.len()).map(|_| Vec::new()).collect()
    }

    fn group_hash_index(&self, group_id: usize, block_index: usize) -> usize {
        (block_index + 1) * self.group_block_sizes[group_id] / self.hash_block_size - 1
    }

    fn extract_group_hash(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        group_id: usize,
        block_index: usize,
    ) -> PyResult<Vec<u8>> {
        block_hashes
            .get_item(self.group_hash_index(group_id, block_index))?
            .extract::<Vec<u8>>()
    }

    fn extract_boundary_hash(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        boundary_tokens: usize,
    ) -> PyResult<Vec<u8>> {
        block_hashes
            .get_item(boundary_tokens / self.hash_block_size - 1)?
            .extract::<Vec<u8>>()
    }

    fn check_hash_count(&self, block_hashes: &Bound<'_, PyAny>, num_tokens: usize) -> PyResult<()> {
        let num_hashes = block_hashes.len()?;
        for (group_id, &block_size) in self.group_block_sizes.iter().enumerate() {
            let num_full_blocks = num_tokens / block_size;
            if num_full_blocks == 0 {
                continue;
            }
            let required_hashes = self.group_hash_index(group_id, num_full_blocks - 1) + 1;
            if num_hashes < required_hashes {
                return Err(PyValueError::new_err(format!(
                    "group {group_id} needs {required_hashes} hashes to cache {num_tokens} tokens, got {num_hashes}"
                )));
            }
        }
        Ok(())
    }

    fn required_blocks(
        &self,
        group_id: usize,
        num_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
    ) -> usize {
        let tokens = if self.group_policies[group_id].is_mamba() {
            num_tokens_main_model
        } else {
            num_tokens
        };
        let mut required = tokens.div_ceil(self.group_block_sizes[group_id]);
        if apply_admission_cap && self.max_admission_blocks[group_id] > 0 {
            required = required.min(self.max_admission_blocks[group_id]);
        }
        required
    }

    fn blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        computed_groups: &[Vec<u32>],
        apply_admission_cap: bool,
    ) -> PyResult<usize> {
        self.validate_groups(computed_groups)?;
        if let Some(state) = self.requests.get(request_id) {
            if computed_groups.iter().any(|blocks| !blocks.is_empty()) {
                return Err(PyAssertionError::new_err(
                    "a running request cannot add prefix-cache hits",
                ));
            }
            return Ok(state
                .blocks
                .iter()
                .enumerate()
                .map(|(group_id, blocks)| {
                    self.required_blocks(
                        group_id,
                        num_tokens,
                        num_tokens_main_model,
                        apply_admission_cap,
                    )
                    .saturating_sub(blocks.len())
                })
                .sum());
        }

        for (group_id, blocks) in computed_groups.iter().enumerate() {
            if self.group_policies[group_id].is_mamba()
                && let Some(&block_id) = blocks.iter().rfind(|&&block_id| block_id != 0)
                && let Some(cache_key) = self.pool.cache_key(block_id)
                && self.mamba_cached_this_step.contains(cache_key)
            {
                return Ok(self.pool.num_blocks() + 1);
            }
        }

        let mut blocks_to_allocate = 0;
        for (group_id, blocks) in computed_groups.iter().enumerate() {
            let required = self.required_blocks(
                group_id,
                num_tokens,
                num_tokens_main_model,
                apply_admission_cap,
            );
            blocks_to_allocate += required.saturating_sub(blocks.len());
            blocks_to_allocate += self.pool.count_evictable(blocks)?;
        }
        Ok(blocks_to_allocate)
    }

    fn release_block_range(
        &mut self,
        request_id: &str,
        group_id: usize,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let Some(state) = self.requests.get_mut(request_id) else {
            return Ok(());
        };
        let end = end.min(state.blocks[group_id].len());
        let mut released = Vec::new();
        for index in (start.min(end)..end).rev() {
            let block_id = state.blocks[group_id][index];
            if block_id != 0 {
                state.blocks[group_id][index] = 0;
                released.push(block_id);
            }
        }
        if !released.is_empty() {
            self.pool.release(released.into_iter())?;
        }
        Ok(())
    }

    fn remove_skipped(&mut self, request_id: &str, processed_tokens: usize) -> PyResult<()> {
        for group_id in 0..self.group_policies.len() {
            let policy = self.group_policies[group_id];
            if policy.is_sliding_window() {
                let skipped_blocks =
                    policy.skipped_tokens(processed_tokens) / self.group_block_sizes[group_id];
                let Some(released_prefix_blocks) = self
                    .requests
                    .get(request_id)
                    .map(|state| state.released_prefix_blocks[group_id])
                else {
                    continue;
                };
                if skipped_blocks > released_prefix_blocks {
                    self.release_block_range(
                        request_id,
                        group_id,
                        released_prefix_blocks,
                        skipped_blocks,
                    )?;
                    self.requests
                        .get_mut(request_id)
                        .expect("request state exists")
                        .released_prefix_blocks[group_id] = skipped_blocks;
                }
                continue;
            }
            if !policy.is_mamba() {
                continue;
            }
            let Some(state) = self.requests.get(request_id) else {
                continue;
            };
            let first_required_block =
                processed_tokens.div_ceil(self.group_block_sizes[group_id]).saturating_sub(1);
            let Some(block_idx) = state.last_state_block_idx[group_id] else {
                continue;
            };
            if block_idx < first_required_block {
                self.release_block_range(request_id, group_id, block_idx, block_idx + 1)?;
            }
        }
        Ok(())
    }

    fn allocate_request_blocks(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        let mut new_groups = self.empty_groups();
        for (group_id, new_blocks) in new_groups.iter_mut().enumerate() {
            if self.group_policies[group_id].is_mamba() {
                continue;
            }
            let required = num_tokens.div_ceil(self.group_block_sizes[group_id]);
            let current = self.requests[request_id].blocks[group_id].len();
            new_blocks.reserve(required.saturating_sub(current));
            for _ in current..required {
                new_blocks.push(self.pool.allocate()?);
            }
            self.new_attention_block_ids.extend_from_slice(new_blocks);
            self.requests.get_mut(request_id).expect("request state exists").blocks[group_id]
                .extend_from_slice(new_blocks);
        }

        for (group_id, new_group) in new_groups.iter_mut().enumerate() {
            if !self.group_policies[group_id].is_mamba() {
                continue;
            }
            let required = num_tokens_main_model.div_ceil(self.group_block_sizes[group_id]);
            let state = self.requests.get_mut(request_id).expect("request state exists");
            if required <= state.blocks[group_id].len() {
                state.mamba_allocated[group_id] = true;
                continue;
            }
            let previous_len = state.blocks[group_id].len();
            if state.mamba_allocated[group_id] || previous_len > 0 {
                state.last_state_block_idx[group_id] = previous_len.checked_sub(1);
            }
            state.blocks[group_id].resize(required.saturating_sub(1), 0);
            let new_mamba_block = self.pool.allocate()?;
            state.blocks[group_id].push(new_mamba_block);
            state.mamba_allocated[group_id] = true;
            *new_group = state.blocks[group_id][previous_len..].to_vec();
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
        self.check_hash_count(block_hashes.as_any(), num_tokens)?;
        for group_id in 0..self.group_policies.len() {
            let block_size = self.group_block_sizes[group_id];
            let num_full_blocks = num_tokens / block_size;
            let (start, blocks) = {
                let state = self.requests.get(request_id).ok_or_else(|| {
                    PyKeyError::new_err(format!("request {request_id:?} has no allocated blocks"))
                })?;
                if state.blocks[group_id].len() < num_full_blocks {
                    return Err(PyAssertionError::new_err(format!(
                        "request {request_id:?} group {group_id} does not have {num_full_blocks} cacheable blocks"
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
                let block_hash = self.extract_group_hash(block_hashes.as_any(), group_id, index)?;
                let cache_key = CacheKey::new(block_hash, group_id);
                let parent_block_id = self.group_policies[group_id]
                    .is_full_attention()
                    .then(|| index.checked_sub(1))
                    .flatten()
                    .map(|index| self.requests[request_id].blocks[group_id][index]);
                self.pool.cache(
                    block_id,
                    cache_key.clone(),
                    (index + 1) * block_size,
                    parent_block_id,
                )?;
                if self.group_policies[group_id].is_mamba() {
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

    fn full_group_ids(&self) -> Vec<usize> {
        self.group_policies
            .iter()
            .enumerate()
            .filter_map(|(group_id, policy)| policy.is_full_attention().then_some(group_id))
            .collect()
    }

    fn equivalent_group_ids(&self, group_id: usize) -> Vec<usize> {
        self.group_policies
            .iter()
            .enumerate()
            .filter_map(|(candidate_id, &policy)| {
                (policy == self.group_policies[group_id]
                    && self.group_block_sizes[candidate_id] == self.group_block_sizes[group_id])
                    .then_some(candidate_id)
            })
            .collect()
    }

    fn drop_eagle_full_attention_tail(
        &self,
        mut hit_groups: Vec<Vec<u32>>,
        hit_tokens: usize,
    ) -> (Vec<Vec<u32>>, usize) {
        if !self.full_group_ids().iter().any(|&group_id| self.eagle_groups[group_id]) {
            return (hit_groups, hit_tokens);
        }
        let hit_tokens = hit_tokens.saturating_sub(self.scheduler_block_size);
        for group_id in self.full_group_ids() {
            hit_groups[group_id].truncate(hit_tokens / self.group_block_sizes[group_id]);
        }
        (hit_groups, hit_tokens)
    }

    fn find_full_attention_hits(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_tokens: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize)> {
        let full_groups = self.full_group_ids();
        let mut hit_groups = self.empty_groups();
        if full_groups.is_empty() {
            return Ok((
                hit_groups,
                max_tokens / self.scheduler_block_size * self.scheduler_block_size,
            ));
        }
        let max_boundaries = (max_tokens / self.scheduler_block_size)
            .min(block_hashes.len()? * self.hash_block_size / self.scheduler_block_size);
        if max_boundaries == 0 {
            return Ok((hit_groups, 0));
        }

        let first_boundary = self.scheduler_block_size;
        let first_hash = self.extract_boundary_hash(block_hashes, first_boundary)?;
        let mut terminal_blocks = Vec::with_capacity(full_groups.len());
        for &group_id in &full_groups {
            let Some(block_id) = self.pool.find_cached(first_hash.clone(), group_id) else {
                return Ok((hit_groups, 0));
            };
            terminal_blocks.push((group_id, block_id));
        }

        let mut low = 1;
        let mut high = max_boundaries + 1;
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            let boundary = middle * self.scheduler_block_size;
            let block_hash = self.extract_boundary_hash(block_hashes, boundary)?;
            let mut middle_blocks = Vec::with_capacity(full_groups.len());
            for &group_id in &full_groups {
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

        let hit_tokens = low * self.scheduler_block_size;
        for &(group_id, block_id) in &terminal_blocks {
            let expected_len = hit_tokens / self.group_block_sizes[group_id];
            let Some(path) = self.pool.find_cached_path(block_id, group_id, expected_len) else {
                return self.find_full_attention_hits_scalar(
                    block_hashes,
                    max_tokens,
                    &full_groups,
                );
            };
            hit_groups[group_id] = path;
        }
        Ok(self.drop_eagle_full_attention_tail(hit_groups, hit_tokens))
    }

    fn find_full_attention_hits_scalar(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_tokens: usize,
        full_groups: &[usize],
    ) -> PyResult<(Vec<Vec<u32>>, usize)> {
        let mut hit_groups = self.empty_groups();
        let mut common_hit_tokens = max_tokens;
        for &group_id in full_groups {
            let block_size = self.group_block_sizes[group_id];
            let max_blocks = (max_tokens / block_size)
                .min(block_hashes.len()? * self.hash_block_size / block_size);
            for index in 0..max_blocks {
                let block_hash = self.extract_group_hash(block_hashes, group_id, index)?;
                let Some(block_id) = self.pool.find_cached(block_hash, group_id) else {
                    break;
                };
                hit_groups[group_id].push(block_id);
            }
            common_hit_tokens = common_hit_tokens.min(hit_groups[group_id].len() * block_size);
        }
        common_hit_tokens =
            common_hit_tokens / self.scheduler_block_size * self.scheduler_block_size;
        for &group_id in full_groups {
            hit_groups[group_id].truncate(common_hit_tokens / self.group_block_sizes[group_id]);
        }
        Ok(self.drop_eagle_full_attention_tail(hit_groups, common_hit_tokens))
    }

    fn find_sliding_window_hits(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        group_ids: &[usize],
        max_tokens: usize,
        drop_eagle_block: bool,
    ) -> PyResult<(Vec<(usize, Vec<u32>)>, usize)> {
        let group_id = group_ids[0];
        let GroupPolicy::SlidingWindow { window_size, .. } = self.group_policies[group_id] else {
            return Err(PyAssertionError::new_err(
                "sliding-window lookup requires a sliding-window group",
            ));
        };
        let block_size = self.group_block_sizes[group_id];
        let max_blocks =
            (max_tokens / block_size).min(block_hashes.len()? * self.hash_block_size / block_size);
        if max_blocks == 0 {
            return Ok((
                group_ids.iter().map(|&group_id| (group_id, Vec::new())).collect(),
                0,
            ));
        }
        let required_contiguous =
            (window_size - 1).div_ceil(block_size).max(1) + usize::from(drop_eagle_block);
        let mut computed = vec![vec![0; max_blocks]; group_ids.len()];
        let mut num_contiguous = 0;
        let mut match_found = false;
        for index in (0..max_blocks).rev() {
            if num_contiguous == 0 {
                let post_pop_blocks = if drop_eagle_block { index } else { index + 1 };
                if !(post_pop_blocks * block_size).is_multiple_of(self.scheduler_block_size) {
                    continue;
                }
            }
            let block_hash = self.extract_group_hash(block_hashes, group_id, index)?;
            let mut cached = Vec::with_capacity(group_ids.len());
            for &candidate_id in group_ids {
                let Some(block_id) = self.pool.find_cached(block_hash.clone(), candidate_id) else {
                    cached.clear();
                    break;
                };
                cached.push(block_id);
            }
            if cached.is_empty() {
                num_contiguous = 0;
                continue;
            }
            for (group_blocks, block_id) in computed.iter_mut().zip(cached) {
                group_blocks[index] = block_id;
            }
            num_contiguous += 1;
            if num_contiguous >= required_contiguous {
                for group_blocks in &mut computed {
                    group_blocks.truncate(index + num_contiguous);
                }
                match_found = true;
                break;
            }
        }
        if !match_found {
            for group_blocks in &mut computed {
                group_blocks.truncate(num_contiguous);
            }
        }
        if drop_eagle_block && !computed[0].is_empty() {
            for group_blocks in &mut computed {
                group_blocks.pop();
            }
        }
        while !computed[0].is_empty()
            && !(computed[0].len() * block_size).is_multiple_of(self.scheduler_block_size)
        {
            for group_blocks in &mut computed {
                group_blocks.pop();
            }
        }
        let hit_tokens = computed[0].len() * block_size;
        Ok((
            group_ids.iter().copied().zip(computed).collect(),
            hit_tokens,
        ))
    }

    fn find_attention_hits(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_tokens: usize,
        lookup_cap: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize, usize)> {
        let (mut hit_groups, mut candidate) =
            self.find_full_attention_hits(block_hashes, max_tokens)?;
        let has_full_attention =
            self.group_policies.iter().any(|policy| policy.is_full_attention());
        let mut longest_hit = if has_full_attention { candidate } else { 0 };
        let mut eagle_verified = FxHashSet::default();

        loop {
            let mut next_candidate = candidate;
            let mut sliding_hits = Vec::new();
            for (group_id, policy) in self.group_policies.iter().enumerate() {
                if !policy.is_sliding_window() {
                    continue;
                }
                let group_ids = self.equivalent_group_ids(group_id);
                if group_ids[0] != group_id {
                    continue;
                }
                let use_eagle =
                    group_ids.iter().any(|&candidate_id| self.eagle_groups[candidate_id]);
                let drop_eagle_block = use_eagle && !eagle_verified.contains(&group_id);
                let lookup_limit = if drop_eagle_block {
                    (next_candidate + self.group_block_sizes[group_id]).min(lookup_cap)
                } else {
                    next_candidate
                };
                let (blocks, hit_tokens) = self.find_sliding_window_hits(
                    block_hashes,
                    &group_ids,
                    lookup_limit,
                    drop_eagle_block,
                )?;
                longest_hit = longest_hit.max(hit_tokens);
                if drop_eagle_block {
                    eagle_verified.insert(group_id);
                } else if hit_tokens < next_candidate {
                    eagle_verified.clear();
                }
                next_candidate = next_candidate.min(hit_tokens);
                sliding_hits.extend(blocks);
            }
            if next_candidate == candidate {
                for (group_id, blocks) in sliding_hits {
                    hit_groups[group_id] = blocks;
                }
                break;
            }
            candidate = next_candidate;
        }

        for (group_id, policy) in self.group_policies.iter().enumerate() {
            if policy.is_full_attention() {
                hit_groups[group_id].truncate(candidate / self.group_block_sizes[group_id]);
            }
        }
        Ok((hit_groups, candidate, longest_hit))
    }

    fn find_mamba_hit(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        mut hit_groups: Vec<Vec<u32>>,
        attention_hit_tokens: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize)> {
        let mamba_groups = self
            .group_policies
            .iter()
            .enumerate()
            .filter_map(|(group_id, policy)| policy.is_mamba().then_some(group_id))
            .collect::<Vec<_>>();
        if mamba_groups.is_empty() {
            return Ok((hit_groups, attention_hit_tokens));
        }
        let max_blocks = attention_hit_tokens / self.scheduler_block_size;
        for index in (0..max_blocks).rev() {
            let boundary = (index + 1) * self.scheduler_block_size;
            let block_hash = self.extract_boundary_hash(block_hashes, boundary)?;
            let mut mamba_blocks = Vec::with_capacity(mamba_groups.len());
            for &group_id in &mamba_groups {
                let Some(block_id) = self.pool.find_cached(block_hash.clone(), group_id) else {
                    mamba_blocks.clear();
                    break;
                };
                mamba_blocks.push((group_id, block_id));
            }
            if mamba_blocks.is_empty() {
                continue;
            }
            for (group_id, policy) in self.group_policies.iter().enumerate() {
                if !policy.is_mamba() {
                    hit_groups[group_id].truncate(boundary / self.group_block_sizes[group_id]);
                }
            }
            for (group_id, block_id) in mamba_blocks {
                hit_groups[group_id] = vec![0; index];
                hit_groups[group_id].push(block_id);
            }
            return Ok((hit_groups, boundary));
        }
        Ok((self.empty_groups(), 0))
    }
}

#[pymethods]
impl HybridKVCacheManager {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_blocks: usize,
        scheduler_block_size: usize,
        hash_block_size: usize,
        enable_caching: bool,
        group_kinds: Vec<u8>,
        group_block_sizes: Vec<usize>,
        sliding_windows: Vec<usize>,
        extra_retained_tokens: Vec<usize>,
        max_admission_blocks: Vec<usize>,
        eagle_groups: Vec<bool>,
    ) -> PyResult<Self> {
        let group_count = group_kinds.len();
        if group_count == 0
            || group_block_sizes.len() != group_count
            || sliding_windows.len() != group_count
            || extra_retained_tokens.len() != group_count
            || max_admission_blocks.len() != group_count
            || eagle_groups.len() != group_count
        {
            return Err(PyValueError::new_err(
                "all hybrid group descriptor vectors must have the same non-zero length",
            ));
        }
        if scheduler_block_size == 0 || hash_block_size == 0 {
            return Err(PyValueError::new_err(
                "scheduler and hash block sizes must be positive",
            ));
        }
        let mut group_policies = Vec::with_capacity(group_count);
        for group_id in 0..group_count {
            let block_size = group_block_sizes[group_id];
            if block_size == 0
                || !scheduler_block_size.is_multiple_of(block_size)
                || !block_size.is_multiple_of(hash_block_size)
            {
                return Err(PyValueError::new_err(format!(
                    "group {group_id} block size {block_size} must divide scheduler block size {scheduler_block_size} and be divisible by hash block size {hash_block_size}"
                )));
            }
            group_policies.push(GroupPolicy::from_descriptor(
                group_kinds[group_id],
                sliding_windows[group_id],
                extra_retained_tokens[group_id],
            )?);
        }
        let has_mamba = group_policies.iter().any(|policy| policy.is_mamba());
        let has_sliding = group_policies.iter().any(|policy| policy.is_sliding_window());
        if has_mamba && has_sliding {
            return Err(PyValueError::new_err(
                "combining Mamba and sliding-window groups is not supported",
            ));
        }
        if eagle_groups.iter().enumerate().any(|(group_id, &is_eagle)| {
            is_eagle
                && (group_policies[group_id].is_mamba()
                    || (group_policies[group_id].is_full_attention()
                        && group_block_sizes[group_id] != scheduler_block_size))
        }) {
            return Err(PyValueError::new_err(
                "EAGLE/DSpark groups must use sliding-window attention or scheduler-sized full attention",
            ));
        }
        if has_mamba
            && (!group_policies.iter().any(|policy| policy.is_full_attention())
                || group_block_sizes.iter().any(|&block_size| block_size != scheduler_block_size)
                || hash_block_size != scheduler_block_size)
        {
            return Err(PyValueError::new_err(
                "Mamba align currently requires a full-attention group and identical scheduler, hash, and group block sizes",
            ));
        }
        Ok(Self {
            scheduler_block_size,
            hash_block_size,
            enable_caching,
            group_policies,
            group_block_sizes,
            eagle_groups,
            max_admission_blocks,
            pool: BlockPool::new(num_blocks, enable_caching)?,
            requests: FxHashMap::default(),
            pending_hits: FxHashMap::default(),
            mamba_cached_this_step: FxHashSet::default(),
            new_attention_block_ids: Vec::new(),
        })
    }

    fn find_longest_cache_hit(
        &mut self,
        request_id: &str,
        block_hashes: &Bound<'_, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Vec<Vec<u32>>, usize, usize)> {
        if !self.enable_caching {
            return Ok((self.empty_groups(), 0, 0));
        }
        let max_tokens =
            max_cache_hit_length / self.scheduler_block_size * self.scheduler_block_size;
        let (attention_groups, attention_hit, longest_hit) =
            self.find_attention_hits(block_hashes, max_tokens, max_cache_hit_length)?;
        let (hit_groups, hit_tokens) =
            self.find_mamba_hit(block_hashes, attention_groups, attention_hit)?;
        self.pending_hits.insert(request_id.to_owned(), hit_groups.clone());
        Ok((
            hit_groups,
            hit_tokens,
            longest_hit.saturating_sub(hit_tokens),
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
        computed_groups: Option<Vec<Vec<u32>>>,
        block_hashes: &Bound<'_, PyList>,
        num_tokens_to_cache: usize,
        processed_computed_tokens: usize,
        reserved_blocks: usize,
        watermark_blocks: usize,
        full_num_tokens: Option<usize>,
    ) -> PyResult<Option<Vec<Vec<u32>>>> {
        let computed_groups = match computed_groups {
            Some(computed_groups) => {
                self.pending_hits.remove(request_id);
                computed_groups
            }
            None => self.pending_hits.remove(request_id).ok_or_else(|| {
                PyAssertionError::new_err(format!(
                    "request {request_id:?} has no pending native prefix-cache hit"
                ))
            })?,
        };
        self.check_hash_count(block_hashes.as_any(), num_tokens_to_cache)?;
        if let Some(full_num_tokens) = full_num_tokens {
            let full_required = self.blocks_to_allocate(
                request_id,
                full_num_tokens,
                full_num_tokens,
                &computed_groups,
                true,
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
            false,
        )?;
        if blocks_to_allocate + reserved_blocks + watermark_blocks > self.pool.num_free_blocks() {
            return Ok(None);
        }

        if !self.requests.contains_key(request_id) {
            self.validate_groups(&computed_groups)?;
            for blocks in &computed_groups {
                self.pool.touch(blocks)?;
            }
            let group_count = self.group_policies.len();
            let state = RequestState {
                cached_blocks: computed_groups.iter().map(Vec::len).collect(),
                released_prefix_blocks: computed_groups
                    .iter()
                    .map(|blocks| blocks.iter().take_while(|&&block_id| block_id == 0).count())
                    .collect(),
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
        self.pending_hits.remove(request_id);
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

    #[pyo3(signature = (
        request_id,
        num_tokens,
        num_tokens_main_model,
        computed_groups,
        apply_admission_cap=false,
    ))]
    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        computed_groups: Vec<Vec<u32>>,
        apply_admission_cap: bool,
    ) -> PyResult<usize> {
        self.blocks_to_allocate(
            request_id,
            num_tokens,
            num_tokens_main_model,
            &computed_groups,
            apply_admission_cap,
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
            .zip(&self.group_policies)
            .map(|(blocks, policy)| {
                if policy.is_full_attention() {
                    blocks
                        .iter()
                        .take_while(|&&block_id| self.pool.ref_count(block_id) == request_count)
                        .count()
                } else {
                    0
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
            self.pending_hits.clear();
            self.mamba_cached_this_step.clear();
        }
        reset
    }

    fn new_step_starts(&mut self) {
        self.pending_hits.clear();
        self.mamba_cached_this_step.clear();
    }

    fn discard_pending_hit(&mut self, request_id: &str) {
        self.pending_hits.remove(request_id);
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
        state
            .blocks
            .iter()
            .zip(&self.group_policies)
            .zip(&self.group_block_sizes)
            .filter(|((_, policy), _)| !policy.is_mamba())
            .flat_map(|((blocks, _), &block_size)| {
                let start_block = start_token / block_size;
                let end_block = end_token.div_ceil(block_size);
                blocks[start_block.min(blocks.len())..end_block.min(blocks.len())]
                    .iter()
                    .copied()
                    .filter(|&block_id| block_id != 0)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn record_blocks_for_zeroing(&mut self, request_id: &str, start_token: usize) -> PyResult<()> {
        if !start_token.is_multiple_of(self.scheduler_block_size) {
            return Err(PyValueError::new_err(
                "start_token must be scheduler-block aligned for KV cache zeroing",
            ));
        }
        let state = self
            .get_request(request_id)
            .ok_or_else(|| PyKeyError::new_err(format!("request {request_id:?} has no blocks")))?;
        let block_ids = state
            .blocks
            .iter()
            .zip(&self.group_policies)
            .zip(&self.group_block_sizes)
            .filter(|((_, policy), _)| !policy.is_mamba())
            .flat_map(|((blocks, _), &block_size)| {
                blocks[start_token / block_size..]
                    .iter()
                    .copied()
                    .filter(|&block_id| block_id != 0)
            })
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
