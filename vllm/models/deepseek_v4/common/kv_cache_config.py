# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _annotate_eagle_groups,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


class DeepseekV4KVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Like MTP, the DSpark drafter reuses the target's decoder layer and
    registers its KV layers last without a spec marker, so its group can only
    be found positionally."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        groups = super().get_kv_cache_groups(vllm_config, kv_cache_spec)
        _annotate_eagle_groups(
            vllm_config, kv_cache_spec, groups, use_trailing_layer_fallback=True
        )
        return groups
