// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Self-contained KV-cache metadata managers exposed to Python.

mod block_pool;
mod full_attention;
mod hybrid;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn _rust_kv_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<full_attention::FullAttentionKVCacheManager>()?;
    m.add_class::<hybrid::HybridMambaKVCacheManager>()?;
    Ok(())
}
