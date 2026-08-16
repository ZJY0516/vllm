// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::FxHashMap;

use crate::block_pool::{BlockPool, CacheKey};

/// Owns the complete metadata state for one full-attention KV cache group.
#[pyclass(module = "vllm._rust_kv_cache")]
pub(crate) struct FullAttentionKVCacheManager {
    block_size: usize,
    enable_caching: bool,
    pool: BlockPool,
    request_blocks: FxHashMap<String, Vec<u32>>,
    request_cached_blocks: FxHashMap<String, usize>,
}

impl FullAttentionKVCacheManager {
    fn blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        computed_block_ids: &[u32],
    ) -> PyResult<usize> {
        let required_blocks = num_tokens.div_ceil(self.block_size);
        if let Some(request_blocks) = self.request_blocks.get(request_id) {
            if !computed_block_ids.is_empty() {
                return Err(PyAssertionError::new_err(
                    "a running request cannot add prefix-cache hits",
                ));
            }
            return Ok(required_blocks.saturating_sub(request_blocks.len()));
        }
        let new_blocks = required_blocks.saturating_sub(computed_block_ids.len());
        Ok(new_blocks + self.pool.count_evictable(computed_block_ids)?)
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
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let parent_block_id = index.checked_sub(1).map(|index| request_blocks[index]);
            self.pool.cache(
                block_id,
                CacheKey::new(block_hash, 0),
                (index + 1) * self.block_size,
                parent_block_id,
            )?;
        }
        self.request_cached_blocks.insert(request_id.to_owned(), num_full_blocks);
        Ok(())
    }
}

#[pymethods]
impl FullAttentionKVCacheManager {
    #[new]
    fn new(num_blocks: usize, block_size: usize, enable_caching: bool) -> PyResult<Self> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be positive"));
        }
        Ok(Self {
            block_size,
            enable_caching,
            pool: BlockPool::new(num_blocks, enable_caching)?,
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
        let max_blocks = (max_cache_hit_length / self.block_size).min(block_hashes.len()?);
        if max_blocks == 0 {
            return Ok((Vec::new(), 0));
        }

        let first_hash = block_hashes.get_item(0)?.extract::<Vec<u8>>()?;
        let Some(mut terminal_block_id) = self.pool.find_cached(first_hash, 0) else {
            return Ok((Vec::new(), 0));
        };
        let mut low = 0;
        let mut high = max_blocks;
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            let block_hash = block_hashes.get_item(middle)?.extract::<Vec<u8>>()?;
            if let Some(block_id) = self.pool.find_cached(block_hash, 0) {
                low = middle;
                terminal_block_id = block_id;
            } else {
                high = middle;
            }
        }
        if let Some(block_ids) = self.pool.find_cached_path(terminal_block_id, 0, low + 1) {
            return Ok((block_ids, (low + 1) * self.block_size));
        }

        let mut block_ids = Vec::with_capacity(max_blocks);
        for index in 0..max_blocks {
            let block_hash = block_hashes.get_item(index)?.extract::<Vec<u8>>()?;
            let Some(block_id) = self.pool.find_cached(block_hash, 0) else {
                break;
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
            if full_required + reserved_blocks + watermark_blocks > self.pool.num_free_blocks() {
                return Ok(None);
            }
        }
        let blocks_to_allocate =
            self.blocks_to_allocate(request_id, num_tokens, &computed_block_ids)?;
        if blocks_to_allocate + reserved_blocks + watermark_blocks > self.pool.num_free_blocks() {
            return Ok(None);
        }

        if !self.request_blocks.contains_key(request_id) {
            self.pool.touch(&computed_block_ids)?;
            let num_computed_blocks = computed_block_ids.len();
            self.request_blocks.insert(request_id.to_owned(), computed_block_ids);
            self.request_cached_blocks.insert(request_id.to_owned(), num_computed_blocks);
        }

        let required_blocks = num_tokens.div_ceil(self.block_size);
        let current_blocks = self.request_blocks[request_id].len();
        let num_new_blocks = required_blocks.saturating_sub(current_blocks);
        let mut new_block_ids = Vec::with_capacity(num_new_blocks);
        for _ in 0..num_new_blocks {
            new_block_ids.push(self.pool.allocate()?);
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
        self.pool.release(block_ids.into_iter().rev())
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
            .take_while(|&&block_id| self.pool.ref_count(block_id) == request_count)
            .count())
    }

    fn estimate_cached_tokens(&self, request_id: &str) -> usize {
        self.request_blocks
            .get(request_id)
            .into_iter()
            .flatten()
            .map(|&block_id| self.pool.hash_num_tokens(block_id))
            .max()
            .unwrap_or(0)
    }

    fn evict_blocks(&mut self, block_ids: Vec<u32>) -> PyResult<()> {
        self.pool.evict(block_ids)
    }

    fn reset_prefix_cache(&mut self) -> bool {
        self.pool.reset_prefix_cache()
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
