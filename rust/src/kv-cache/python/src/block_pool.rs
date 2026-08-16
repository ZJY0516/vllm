// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CacheKey {
    pub(crate) block_hash: Vec<u8>,
    pub(crate) group_id: usize,
}

impl CacheKey {
    pub(crate) fn new(block_hash: Vec<u8>, group_id: usize) -> Self {
        Self {
            block_hash,
            group_id,
        }
    }
}

#[derive(Default)]
struct BlockState {
    ref_count: u32,
    cache_key: Option<CacheKey>,
    block_hash_num_tokens: usize,
    parent: Option<BlockReference>,
    generation: u32,
    previous_free: Option<u32>,
    next_free: Option<u32>,
    in_free_queue: bool,
}

#[derive(Clone, Copy, Debug)]
struct BlockReference {
    block_id: u32,
    generation: u32,
}

pub(crate) struct BlockPool {
    enable_caching: bool,
    blocks: Vec<BlockState>,
    free_head: Option<u32>,
    free_tail: Option<u32>,
    num_free_blocks: usize,
    cached_blocks: FxHashMap<CacheKey, Vec<u32>>,
}

impl BlockPool {
    pub(crate) fn new(num_blocks: usize, enable_caching: bool) -> PyResult<Self> {
        if num_blocks < 2 {
            return Err(PyValueError::new_err(
                "the KV cache pool must include a null block and one usable block",
            ));
        }
        let mut blocks = (0..num_blocks).map(|_| BlockState::default()).collect::<Vec<_>>();
        for block_id in 1..num_blocks {
            let state = &mut blocks[block_id];
            state.previous_free = (block_id > 1).then_some((block_id - 1) as u32);
            state.next_free = (block_id + 1 < num_blocks).then_some((block_id + 1) as u32);
            state.in_free_queue = true;
        }
        Ok(Self {
            enable_caching,
            blocks,
            free_head: Some(1),
            free_tail: Some((num_blocks - 1) as u32),
            num_free_blocks: num_blocks - 1,
            cached_blocks: FxHashMap::default(),
        })
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
        let Some(cache_key) = self.blocks[block_id as usize].cache_key.take() else {
            return;
        };
        let remove_entry = if let Some(block_ids) = self.cached_blocks.get_mut(&cache_key) {
            if let Some(index) = block_ids.iter().position(|&id| id == block_id) {
                block_ids.swap_remove(index);
            }
            block_ids.is_empty()
        } else {
            false
        };
        if remove_entry {
            self.cached_blocks.remove(&cache_key);
        }
        self.blocks[block_id as usize].block_hash_num_tokens = 0;
        self.blocks[block_id as usize].parent = None;
    }

    pub(crate) fn allocate(&mut self) -> PyResult<u32> {
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
        state.generation = state.generation.wrapping_add(1);
        Ok(block_id)
    }

    pub(crate) fn touch(&mut self, block_ids: &[u32]) -> PyResult<()> {
        for &block_id in block_ids {
            if block_id == 0 {
                continue;
            }
            let state = self.state(block_id)?;
            if state.ref_count == 0 {
                self.remove_free(block_id)?;
            }
            self.blocks[block_id as usize].ref_count += 1;
        }
        Ok(())
    }

    pub(crate) fn count_evictable(&self, block_ids: &[u32]) -> PyResult<usize> {
        block_ids.iter().try_fold(0, |count, &block_id| {
            if block_id == 0 {
                return Ok(count);
            }
            Ok(count + usize::from(self.state(block_id)?.ref_count == 0))
        })
    }

    pub(crate) fn release<I>(&mut self, block_ids: I) -> PyResult<()>
    where
        I: Iterator<Item = u32>,
    {
        let mut reuse_first = Vec::new();
        let mut reuse_last = Vec::new();
        for block_id in block_ids {
            if block_id == 0 {
                continue;
            }
            let enable_caching = self.enable_caching;
            let state = self.state_mut(block_id)?;
            if state.ref_count == 0 {
                return Err(PyAssertionError::new_err(format!(
                    "cannot free unreferenced block {block_id}"
                )));
            }
            state.ref_count -= 1;
            if state.ref_count == 0 {
                if enable_caching && state.cache_key.is_some() {
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

    pub(crate) fn find_cached(&self, block_hash: Vec<u8>, group_id: usize) -> Option<u32> {
        self.cached_blocks
            .get(&CacheKey::new(block_hash, group_id))
            .and_then(|block_ids| block_ids.first().copied())
    }

    pub(crate) fn cache(
        &mut self,
        block_id: u32,
        cache_key: CacheKey,
        num_tokens: usize,
        parent_block_id: Option<u32>,
    ) -> PyResult<()> {
        if block_id == 0 {
            return Ok(());
        }
        let state = self.state(block_id)?;
        if state.cache_key.is_some() {
            return Err(PyAssertionError::new_err(format!(
                "uncached request block {block_id} already has a hash"
            )));
        }
        let parent = parent_block_id
            .map(|parent_block_id| {
                let parent_state = self.state(parent_block_id)?;
                let parent_cache_key = parent_state.cache_key.as_ref().ok_or_else(|| {
                    PyAssertionError::new_err(format!(
                        "parent block {parent_block_id} is not cached"
                    ))
                })?;
                if parent_cache_key.group_id != cache_key.group_id {
                    return Err(PyAssertionError::new_err(format!(
                        "parent block {parent_block_id} belongs to cache group {}, not {}",
                        parent_cache_key.group_id, cache_key.group_id
                    )));
                }
                Ok(BlockReference {
                    block_id: parent_block_id,
                    generation: parent_state.generation,
                })
            })
            .transpose()?;
        self.cached_blocks.entry(cache_key.clone()).or_default().push(block_id);
        let state = &mut self.blocks[block_id as usize];
        state.cache_key = Some(cache_key);
        state.block_hash_num_tokens = num_tokens;
        state.parent = parent;
        Ok(())
    }

    pub(crate) fn find_cached_path(
        &self,
        terminal_block_id: u32,
        group_id: usize,
        expected_len: usize,
    ) -> Option<Vec<u32>> {
        let mut block_ids = Vec::with_capacity(expected_len);
        let mut block_id = terminal_block_id;
        for index in (0..expected_len).rev() {
            let state = self.blocks.get(block_id as usize)?;
            if state.cache_key.as_ref()?.group_id != group_id {
                return None;
            }
            block_ids.push(block_id);
            match (index, state.parent) {
                (0, None) => {}
                (0, Some(_)) | (_, None) => return None,
                (_, Some(parent)) => {
                    let parent_state = self.blocks.get(parent.block_id as usize)?;
                    if parent_state.generation != parent.generation {
                        return None;
                    }
                    block_id = parent.block_id;
                }
            }
        }
        block_ids.reverse();
        Some(block_ids)
    }

    pub(crate) fn ref_count(&self, block_id: u32) -> u32 {
        self.blocks[block_id as usize].ref_count
    }

    pub(crate) fn cache_key(&self, block_id: u32) -> Option<&CacheKey> {
        self.blocks.get(block_id as usize).and_then(|state| state.cache_key.as_ref())
    }

    pub(crate) fn hash_num_tokens(&self, block_id: u32) -> usize {
        self.blocks[block_id as usize].block_hash_num_tokens
    }

    pub(crate) fn evict(&mut self, block_ids: Vec<u32>) -> PyResult<()> {
        for block_id in block_ids {
            self.state(block_id)?;
            self.remove_cached_hash(block_id);
        }
        Ok(())
    }

    pub(crate) fn reset_prefix_cache(&mut self) -> bool {
        if self.num_free_blocks != self.blocks.len() - 1 {
            return false;
        }
        self.cached_blocks.clear();
        for state in &mut self.blocks {
            state.cache_key = None;
            state.block_hash_num_tokens = 0;
            state.parent = None;
        }
        true
    }

    pub(crate) fn num_free_blocks(&self) -> usize {
        self.num_free_blocks
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn usage(&self) -> f64 {
        let usable_blocks = self.blocks.len() - 1;
        1.0 - self.num_free_blocks as f64 / usable_blocks as f64
    }
}
