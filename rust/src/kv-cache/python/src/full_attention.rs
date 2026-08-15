// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::FxHashMap;

#[derive(Default)]
struct BlockState {
    ref_count: u32,
    block_hash: Option<Vec<u8>>,
    block_hash_num_tokens: usize,
    previous_free: Option<u32>,
    next_free: Option<u32>,
    in_free_queue: bool,
}

/// Owns the complete metadata state for one full-attention KV cache group.
#[pyclass(module = "vllm._rust_kv_cache")]
pub(crate) struct FullAttentionKVCacheManager {
    block_size: usize,
    enable_caching: bool,
    blocks: Vec<BlockState>,
    free_head: Option<u32>,
    free_tail: Option<u32>,
    num_free_blocks: usize,
    cached_blocks: FxHashMap<Vec<u8>, Vec<u32>>,
    request_blocks: FxHashMap<String, Vec<u32>>,
    request_cached_blocks: FxHashMap<String, usize>,
}

impl FullAttentionKVCacheManager {
    fn cdiv(value: usize, divisor: usize) -> usize {
        value.div_ceil(divisor)
    }

    fn state(&self, block_id: u32) -> PyResult<&BlockState> {
        self.blocks.get(block_id as usize).ok_or_else(|| {
            PyValueError::new_err(format!(
                "block ID {block_id} is outside a {}-block pool",
                self.blocks.len()
            ))
        })
    }

    fn state_mut(&mut self, block_id: u32) -> PyResult<&mut BlockState> {
        let num_blocks = self.blocks.len();
        self.blocks.get_mut(block_id as usize).ok_or_else(|| {
            PyValueError::new_err(format!(
                "block ID {block_id} is outside a {num_blocks}-block pool"
            ))
        })
    }

    fn remove_free(&mut self, block_id: u32) -> PyResult<()> {
        let state = self.state(block_id)?;
        if !state.in_free_queue {
            return Err(PyAssertionError::new_err(format!(
                "block {block_id} is not in the free queue"
            )));
        }
        let previous = state.previous_free;
        let next = state.next_free;
        if let Some(previous) = previous {
            self.blocks[previous as usize].next_free = next;
        } else {
            self.free_head = next;
        }
        if let Some(next) = next {
            self.blocks[next as usize].previous_free = previous;
        } else {
            self.free_tail = previous;
        }
        let state = &mut self.blocks[block_id as usize];
        state.previous_free = None;
        state.next_free = None;
        state.in_free_queue = false;
        self.num_free_blocks -= 1;
        Ok(())
    }

    fn pop_free(&mut self) -> PyResult<u32> {
        let block_id =
            self.free_head.ok_or_else(|| PyValueError::new_err("no free KV cache blocks"))?;
        self.remove_free(block_id)?;
        Ok(block_id)
    }

    fn prepend_free(&mut self, block_ids: &[u32]) {
        if block_ids.is_empty() {
            return;
        }
        let old_head = self.free_head;
        let mut previous = None;
        for &block_id in block_ids {
            let state = &mut self.blocks[block_id as usize];
            state.previous_free = previous;
            state.next_free = None;
            state.in_free_queue = true;
            if let Some(previous) = previous {
                self.blocks[previous as usize].next_free = Some(block_id);
            }
            previous = Some(block_id);
        }
        let first = block_ids[0];
        let last = *block_ids.last().expect("non-empty block list");
        self.blocks[last as usize].next_free = old_head;
        if let Some(old_head) = old_head {
            self.blocks[old_head as usize].previous_free = Some(last);
        } else {
            self.free_tail = Some(last);
        }
        self.free_head = Some(first);
        self.num_free_blocks += block_ids.len();
    }

    fn append_free(&mut self, block_ids: &[u32]) {
        if block_ids.is_empty() {
            return;
        }
        let old_tail = self.free_tail;
        let mut previous = old_tail;
        for &block_id in block_ids {
            let state = &mut self.blocks[block_id as usize];
            state.previous_free = previous;
            state.next_free = None;
            state.in_free_queue = true;
            if let Some(previous) = previous {
                self.blocks[previous as usize].next_free = Some(block_id);
            }
            previous = Some(block_id);
        }
        if old_tail.is_none() {
            self.free_head = Some(block_ids[0]);
        }
        self.free_tail = previous;
        self.num_free_blocks += block_ids.len();
    }

    fn remove_cached_hash(&mut self, block_id: u32) {
        let Some(block_hash) = self.blocks[block_id as usize].block_hash.take() else {
            return;
        };
        let remove_entry = if let Some(block_ids) = self.cached_blocks.get_mut(&block_hash) {
            if let Some(index) = block_ids.iter().position(|&id| id == block_id) {
                block_ids.swap_remove(index);
            }
            block_ids.is_empty()
        } else {
            false
        };
        if remove_entry {
            self.cached_blocks.remove(&block_hash);
        }
        self.blocks[block_id as usize].block_hash_num_tokens = 0;
    }

    fn allocate_block(&mut self) -> PyResult<u32> {
        let block_id = self.pop_free()?;
        self.remove_cached_hash(block_id);
        let state = &mut self.blocks[block_id as usize];
        if state.ref_count != 0 {
            return Err(PyAssertionError::new_err(format!(
                "free block {block_id} has ref_count {}",
                state.ref_count
            )));
        }
        state.ref_count = 1;
        Ok(block_id)
    }

    fn touch_blocks(&mut self, block_ids: &[u32]) -> PyResult<()> {
        for &block_id in block_ids {
            let state = self.state(block_id)?;
            if state.ref_count == 0 {
                self.remove_free(block_id)?;
            }
            self.blocks[block_id as usize].ref_count += 1;
        }
        Ok(())
    }

    fn count_evictable(&self, block_ids: &[u32]) -> PyResult<usize> {
        block_ids.iter().try_fold(0, |count, &block_id| {
            Ok(count + usize::from(self.state(block_id)?.ref_count == 0))
        })
    }

    fn blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        computed_block_ids: &[u32],
    ) -> PyResult<usize> {
        let required_blocks = Self::cdiv(num_tokens, self.block_size);
        if let Some(request_blocks) = self.request_blocks.get(request_id) {
            if !computed_block_ids.is_empty() {
                return Err(PyAssertionError::new_err(
                    "a running request cannot add prefix-cache hits",
                ));
            }
            return Ok(required_blocks.saturating_sub(request_blocks.len()));
        }
        let new_blocks = required_blocks.saturating_sub(computed_block_ids.len());
        Ok(new_blocks + self.count_evictable(computed_block_ids)?)
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
        let num_full_blocks = num_tokens / self.block_size;
        let num_cached_blocks = self.request_cached_blocks.get(request_id).copied().unwrap_or(0);
        if num_cached_blocks >= num_full_blocks {
            return Ok(());
        }
        if block_hashes.len() < num_full_blocks {
            return Err(PyValueError::new_err(format!(
                "{num_full_blocks} full blocks require at least {num_full_blocks} hashes, got {}",
                block_hashes.len()
            )));
        }
        let request_blocks = self.request_blocks.get(request_id).ok_or_else(|| {
            PyKeyError::new_err(format!("request {request_id:?} has no allocated blocks"))
        })?;
        if request_blocks.len() < num_full_blocks {
            return Err(PyAssertionError::new_err(format!(
                "request {request_id:?} has {} blocks but {num_full_blocks} are cacheable",
                request_blocks.len()
            )));
        }
        let block_ids = request_blocks[num_cached_blocks..num_full_blocks].to_vec();
        for (index, block_id) in (num_cached_blocks..num_full_blocks).zip(block_ids) {
            if self.blocks[block_id as usize].block_hash.is_some() {
                return Err(PyAssertionError::new_err(format!(
                    "uncached request block {block_id} already has a hash"
                )));
            }
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let state = &mut self.blocks[block_id as usize];
            state.block_hash = Some(block_hash.clone());
            state.block_hash_num_tokens = (index + 1) * self.block_size;
            self.cached_blocks.entry(block_hash).or_default().push(block_id);
        }
        self.request_cached_blocks.insert(request_id.to_owned(), num_full_blocks);
        Ok(())
    }

    fn release_blocks(&mut self, block_ids: impl Iterator<Item = u32>) -> PyResult<()> {
        let mut reuse_first = Vec::new();
        let mut reuse_last = Vec::new();
        let enable_caching = self.enable_caching;
        for block_id in block_ids {
            let state = self.state_mut(block_id)?;
            if state.ref_count == 0 {
                return Err(PyAssertionError::new_err(format!(
                    "cannot free unreferenced block {block_id}"
                )));
            }
            state.ref_count -= 1;
            if state.ref_count == 0 {
                if enable_caching && state.block_hash.is_some() {
                    reuse_last.push(block_id);
                } else {
                    reuse_first.push(block_id);
                }
            }
        }
        self.prepend_free(&reuse_first);
        self.append_free(&reuse_last);
        Ok(())
    }
}

#[pymethods]
impl FullAttentionKVCacheManager {
    #[new]
    fn new(num_blocks: usize, block_size: usize, enable_caching: bool) -> PyResult<Self> {
        if num_blocks < 2 {
            return Err(PyValueError::new_err(
                "the KV cache pool must include a null block and one usable block",
            ));
        }
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be positive"));
        }
        let mut blocks = (0..num_blocks).map(|_| BlockState::default()).collect::<Vec<_>>();
        for block_id in 1..num_blocks {
            let state = &mut blocks[block_id];
            state.previous_free = (block_id > 1).then_some((block_id - 1) as u32);
            state.next_free = (block_id + 1 < num_blocks).then_some((block_id + 1) as u32);
            state.in_free_queue = true;
        }
        Ok(Self {
            block_size,
            enable_caching,
            blocks,
            free_head: Some(1),
            free_tail: Some((num_blocks - 1) as u32),
            num_free_blocks: num_blocks - 1,
            cached_blocks: FxHashMap::default(),
            request_blocks: FxHashMap::default(),
            request_cached_blocks: FxHashMap::default(),
        })
    }

    fn find_longest_cache_hit(
        &self,
        block_hashes: &Bound<'_, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Vec<u32>, usize)> {
        if !self.enable_caching {
            return Ok((Vec::new(), 0));
        }
        let max_blocks = max_cache_hit_length / self.block_size;
        let mut block_ids = Vec::with_capacity(max_blocks);
        for item in block_hashes.try_iter()?.take(max_blocks) {
            let block_hash = item?.extract::<Vec<u8>>()?;
            let Some(cached) = self.cached_blocks.get(&block_hash) else {
                break;
            };
            let Some(&block_id) = cached.first() else {
                return Err(PyAssertionError::new_err(
                    "prefix-cache index contains an empty block list",
                ));
            };
            block_ids.push(block_id);
        }
        let hit_tokens = block_ids.len() * self.block_size;
        Ok((block_ids, hit_tokens))
    }

    #[pyo3(signature = (
        request_id,
        num_tokens,
        computed_block_ids,
        block_hashes,
        num_tokens_to_cache,
        reserved_blocks=0,
        watermark_blocks=0,
        full_num_tokens=None,
    ))]
    fn allocate_slots(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        computed_block_ids: Vec<u32>,
        block_hashes: &Bound<'_, PyList>,
        num_tokens_to_cache: usize,
        reserved_blocks: usize,
        watermark_blocks: usize,
        full_num_tokens: Option<usize>,
    ) -> PyResult<Option<Vec<u32>>> {
        let num_full_blocks_to_cache = num_tokens_to_cache / self.block_size;
        if block_hashes.len() < num_full_blocks_to_cache {
            return Err(PyValueError::new_err(format!(
                "{num_full_blocks_to_cache} full blocks are cacheable, but only {} hashes exist",
                block_hashes.len()
            )));
        }
        if let Some(full_num_tokens) = full_num_tokens {
            let full_required =
                self.blocks_to_allocate(request_id, full_num_tokens, &computed_block_ids)?;
            if full_required + reserved_blocks + watermark_blocks > self.num_free_blocks {
                return Ok(None);
            }
        }
        let blocks_to_allocate =
            self.blocks_to_allocate(request_id, num_tokens, &computed_block_ids)?;
        if blocks_to_allocate + reserved_blocks + watermark_blocks > self.num_free_blocks {
            return Ok(None);
        }

        if !self.request_blocks.contains_key(request_id) {
            self.touch_blocks(&computed_block_ids)?;
            self.request_blocks.insert(request_id.to_owned(), computed_block_ids.clone());
            self.request_cached_blocks
                .insert(request_id.to_owned(), computed_block_ids.len());
        }

        let required_blocks = Self::cdiv(num_tokens, self.block_size);
        let current_blocks = self.request_blocks[request_id].len();
        let num_new_blocks = required_blocks.saturating_sub(current_blocks);
        let mut new_block_ids = Vec::with_capacity(num_new_blocks);
        for _ in 0..num_new_blocks {
            new_block_ids.push(self.allocate_block()?);
        }
        self.request_blocks
            .get_mut(request_id)
            .expect("request table entry was inserted")
            .extend_from_slice(&new_block_ids);
        self.cache_request_blocks(request_id, block_hashes, num_tokens_to_cache)?;
        Ok(Some(new_block_ids))
    }

    fn cache_blocks(
        &mut self,
        request_id: &str,
        block_hashes: &Bound<'_, PyList>,
        num_tokens: usize,
    ) -> PyResult<()> {
        self.cache_request_blocks(request_id, block_hashes, num_tokens)
    }

    fn free(&mut self, request_id: &str) -> PyResult<()> {
        let Some(block_ids) = self.request_blocks.remove(request_id) else {
            return Ok(());
        };
        self.request_cached_blocks.remove(request_id);
        self.release_blocks(block_ids.into_iter().rev())
    }

    fn get_block_ids(&self, request_id: &str) -> Vec<u32> {
        self.request_blocks.get(request_id).cloned().unwrap_or_default()
    }

    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        computed_block_ids: Vec<u32>,
    ) -> PyResult<usize> {
        self.blocks_to_allocate(request_id, num_tokens, &computed_block_ids)
    }

    fn get_num_common_prefix_blocks(&self, running_request_id: &str) -> PyResult<usize> {
        let request_blocks = self.request_blocks.get(running_request_id).ok_or_else(|| {
            PyKeyError::new_err(format!("request {running_request_id:?} has no blocks"))
        })?;
        let request_count = self.request_blocks.len() as u32;
        Ok(request_blocks
            .iter()
            .take_while(|&&block_id| self.blocks[block_id as usize].ref_count == request_count)
            .count())
    }

    fn estimate_cached_tokens(&self, request_id: &str) -> usize {
        self.request_blocks
            .get(request_id)
            .into_iter()
            .flatten()
            .map(|&block_id| self.blocks[block_id as usize].block_hash_num_tokens)
            .max()
            .unwrap_or(0)
    }

    fn evict_blocks(&mut self, block_ids: Vec<u32>) -> PyResult<()> {
        for block_id in block_ids {
            self.state(block_id)?;
            self.remove_cached_hash(block_id);
        }
        Ok(())
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if self.num_free_blocks != self.blocks.len() - 1 {
            return false;
        }
        self.cached_blocks.clear();
        for state in &mut self.blocks {
            state.block_hash = None;
            state.block_hash_num_tokens = 0;
        }
        true
    }

    #[getter]
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks
    }

    #[getter]
    fn usage(&self) -> f64 {
        let usable_blocks = self.blocks.len() - 1;
        1.0 - self.num_free_blocks as f64 / usable_blocks as f64
    }
}
