# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sparse selection and attention using FlashInfer and PyTorch."""

from typing import Any

import torch
import torch.nn.functional as F


class Step5FlashInferSparse:
    def __init__(self, device: torch.device):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        from flashinfer.attn_scores import fp8_paged_mqa_logits

        self._logits = fp8_paged_mqa_logits
        self._workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
        self._attention = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, "NHD", backend="fa2"
        )
        self._decode_plans: dict[tuple, tuple[Any, torch.Tensor]] = {}

    def run_decode_attention(
        self, query, kv_cache, regions, block_table, positions, active, out, scale
    ):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        batch, heads, dim = query.shape
        capacity = regions.shape[1] * 8
        plan_key = (batch, heads, dim, capacity, query.dtype, kv_cache.dtype, scale)
        if plan_key not in self._decode_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Step5 decode attention must be warmed up before capture"
                )
            cu_q = torch.arange(batch + 1, dtype=torch.int32)
            cu_k = cu_q * capacity
            mask = torch.zeros(
                batch * capacity // 8, dtype=torch.uint8, device=query.device
            )
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=cu_q.to(query.device),
                kv_indptr_buf=cu_k.to(query.device),
                custom_mask_buf=mask,
                mask_indptr_buf=torch.arange(
                    batch + 1, device=query.device, dtype=torch.int32
                )
                * (capacity // 8),
                backend="fa2",
            )
            wrapper.plan(
                cu_q,
                cu_k,
                heads,
                1,
                dim,
                head_dim_vo=dim,
                custom_mask=torch.zeros(
                    batch * capacity, device=query.device, dtype=torch.bool
                ),
                causal=False,
                sm_scale=scale,
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                disable_split_kv=True,
            )
            self._decode_plans[plan_key] = wrapper, mask
        wrapper, mask = self._decode_plans[plan_key]
        offsets = torch.arange(8, device=query.device)
        logical = regions[..., None] * 8 + offsets
        valid = (
            (regions[..., None] >= 0)
            & (logical <= positions[:, None, None])
            & active[:, None, None]
        )
        # FlashInfer consumes little-endian packed masks, one byte per region.
        mask.copy_((valid.to(torch.uint8) << offsets.to(torch.uint8)).sum(-1).flatten())
        tokens = logical.clamp_min(0).reshape(batch, capacity)
        block_size = kv_cache.shape[2]
        physical = block_table.gather(1, tokens // block_size).long()
        key_cache, value_cache = kv_cache.transpose(1, 2).split(dim, -1)
        k = key_cache[physical, tokens % block_size].reshape(batch * capacity, 1, dim)
        v = value_cache[physical, tokens % block_size].reshape(batch * capacity, 1, dim)
        wrapper.run(query.contiguous(), k, v, out=out)
        out.masked_fill_(~active[:, None, None], 0)

    def pack_summaries(self, mean, block_table, regions):
        capacity = (regions + 127) // 128 * 128
        pages = capacity // 32
        requests = block_table.shape[0]
        logical = torch.arange(capacity, device=mean.device)
        columns = logical // mean.shape[1]
        physical = block_table[:, columns.clamp_max(block_table.shape[1] - 1)]
        valid = (columns < block_table.shape[1]) & (logical < regions)
        valid = valid[None, :] & (physical >= 0) & (physical < mean.shape[0])
        values = mean[
            physical.clamp(0, mean.shape[0] - 1).long(), logical % mean.shape[1]
        ]
        values = torch.where(valid[..., None], values, 0)
        packed = torch.empty(
            (requests * pages, 32 * 260), device=mean.device, dtype=torch.uint8
        )
        packed[:, : 32 * 256] = values.reshape(requests * pages, 32 * 256)
        packed[:, 32 * 256 :].view(torch.float32).fill_(1)
        return packed.view(-1, 32, 1, 260), pages

    def logits(self, query, weights, packed, pages, requests, lengths, regions):
        q = query[:, 0].float()
        w = weights[:, 0].float()
        heads = q.shape[1]
        if heads < 8:
            q = F.pad(q, (0, 0, 0, 8 - heads))
            w = F.pad(w, (0, 8 - heads))
        q = q.to(torch.float8_e4m3fn).unsqueeze(1)
        table = (requests[:, None] * pages + torch.arange(pages, device=q.device)).int()
        return self._logits(
            q=q,
            kv_fused=packed,
            weights=w.contiguous(),
            context_lens=lengths.int().contiguous(),
            block_table=table,
            max_context_len=regions,
        )

    def run_attention(
        self, query, kv_cache, regions, block_table, requests, positions, out, scale
    ):
        block_size = kv_cache.shape[2]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(query.shape[-1], -1)
        offsets = torch.arange(8, device=query.device)
        for start in range(0, query.shape[0], 32):
            end = min(start + 32, query.shape[0])
            selected = regions[start:end]
            logical = selected[..., None] * 8 + offsets
            valid = (selected[..., None] >= 0) & (
                logical <= positions[start:end, None, None]
            )
            tokens = logical[valid]
            owners = requests[start:end, None, None].expand_as(logical)[valid]
            physical = block_table[owners.long(), tokens // block_size].long()
            k = key_cache[physical, tokens % block_size].contiguous()
            v = value_cache[physical, tokens % block_size].contiguous()
            counts = valid.sum((1, 2), dtype=torch.int32)
            cu_k = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
            cu_q = torch.arange(end - start + 1, device=query.device, dtype=torch.int32)
            self._attention.plan(
                cu_q,
                cu_k,
                query.shape[1],
                1,
                query.shape[2],
                head_dim_vo=query.shape[2],
                causal=False,
                sm_scale=scale,
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                disable_split_kv=True,
            )
            out[start:end] = self._attention.run(query[start:end].contiguous(), k, v)
