# SPDX-License-Identifier: Apache-2.0

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionLayer,
                                              AttentionMetadata,
                                              MLAAttentionImpl, T)
from vllm.attention.backends.utils import get_flash_attn_version
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce,
                              get_tensor_model_parallel_rank)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_fp8_linear_generic, current_platform_fp8_dtype, is_fp8)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize)
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    from flash_attn import flash_attn_varlen_func
import csv
from vllm.logger import init_logger
import os
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity
import math
import triton
import triton.language as tl

logger = init_logger(__name__)


@triton.jit
def _get_kv_cache_kernel(
    KV_cache,  #kv_cache [num_blocks, BLOCK_SIZE, kv_dim]
    Block_tables,   #block_table [bsz, max_seq_len]
    Seq_lens,    #seq_lens_tensor [bsz]
    K_buffer,  #kv_buffer [bsz, max_seq_len, 512]
    V_buffer,  #v_buffer [bsz, max_seq_len, 64]
    BLOCK_SIZE,
    stride_block_tables,
    stride_cache_b,
    stride_cache_s,
    stride_k_buffer_b,
    stride_k_buffer_s,
    stride_v_buffer_b,
    stride_v_buffer_s,
    BLOCK_DV: tl.constexpr,
    BLOCK_DPE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    split_id = tl.program_id(1)
    
    cur_batch_seq_len = tl.load(Seq_lens + cur_batch)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, BLOCK_SIZE)
        
    for offs_n in range(0, kv_len_per_split):
        cur_len_id = offs_n * BLOCK_SIZE + split_id
        if(cur_len_id < cur_batch_seq_len):
            page_number = tl.load(Block_tables + cur_batch * stride_block_tables + offs_n)
            kv_loc = page_number * stride_cache_b + split_id * stride_cache_s
            offs_buf_k = kv_loc[None, :] + tl.arange(0, BLOCK_DV)
            offs_buf_v = kv_loc[None, :] + tl.arange(0, BLOCK_DPE) + BLOCK_DV
            k = tl.load(KV_cache + offs_buf_k)
            v = tl.load(KV_cache + offs_buf_v)

            k_buffer_loc = cur_batch * stride_k_buffer_b + (offs_n * BLOCK_SIZE + split_id) * stride_k_buffer_s 
            v_buffer_loc = cur_batch * stride_v_buffer_b + (offs_n * BLOCK_SIZE + split_id) * stride_v_buffer_s
            off_k_buffer_loc = k_buffer_loc[None, :] + tl.arange(0, BLOCK_DV)
            off_v_buffer_loc = v_buffer_loc[None, :] + tl.arange(0, BLOCK_DPE)
            tl.store(K_buffer + off_k_buffer_loc, k)
            tl.store(V_buffer + off_v_buffer_loc, v)

@triton.jit
def _select_kv_cache_kernel(
    cache,  #v_c_cache [bsz, max_seq_lens, dim]
    Top_k,    #[bsz]
    Indices,   #[bsz, max_seq_len]
    Select_cache,  #kv_buffer [bsz, max_seq_len, dim]
    BLOCK_SIZE,
    stride_Indices,
    stride_cache_b,
    stride_cache_s,
    stride_select_cache_b,
    stride_select_cache_s,
    BLOCK_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    split_id = tl.program_id(1)
    
    cur_batch_seq_len = tl.load(Top_k + cur_batch)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, BLOCK_SIZE)
    
    for offs_n in range(0, kv_len_per_split):
        cur_len_id = offs_n * BLOCK_SIZE + split_id
        # tl.device_print("cur_len_id", cur_len_id)
        if(cur_len_id < cur_batch_seq_len):
            index = tl.load(Indices + cur_batch * stride_Indices + cur_len_id)
            cache_loc = cur_batch*stride_cache_b + index*stride_cache_s
            offs_cache = cache_loc[None, :] + tl.arange(0, BLOCK_DV)
            select_tensor = tl.load(cache + offs_cache)

            select_cache_loc = cur_batch*stride_select_cache_b + cur_len_id*stride_select_cache_s
            select_offs_cache = select_cache_loc[None, :] + tl.arange(0, BLOCK_DV)
            tl.store(Select_cache + select_offs_cache, select_tensor)

@dataclass
class MLACommonMetadata(AttentionMetadata):
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    Common class for implementing repeated parts

    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    Deepseek's MLA attention works the following way:
    * Use a single latent vector to represent the entire KV cache.
    * The attention "simulates" a multi-head attention, while the compute is
      similar to multi-query attention.
    * The dataflow is as follows,

        * B: batch/sequence length
        * H: hidden size
        * N: number of attention heads
        * Lq: latent dimension for Q
        * Lkv: latent dimension for K/V
        * P: nope dimension, P+R is the actual head_dim in common attention.
        * R: rope dimension, this slide of the head_dim goes through rope.
        * V: V head dim.
        * kv_c: latent/compressed KV
        * q_c: latent/compressed Q

        #
        # Outside the MLA attention backend
        #

        1. The hidden states (B, H) are projected down into cq (B, Lq) and
           kv_c_k_pe (B, Lkv+R).
        2. The kv_c_k_pe is split into kv_c (B, Lkv) and k_pe (B, R). cq
           and kv_c are normalized.

        #
        # Inside the MLA attention backend
        #

        * if prefill:

        3. The q_c is then projected up into the multi-head version.
           * q_c goes from (B, Lq) to (B, N, (P+R)), which is split into q_nope
             (B, N, P) and q_pe (B, N, R).
        4. q_pe, k_pe are then passed through rotary embeddings.
        5. kv_c and k_pe are concatenated and inserted into the cache
        6. The kv_c is then projected up into the multi-head version.
           * kv_c goes from (B, Lkv) to (B, N, (P+V)) which has the nope
             dimensions for K and V, which is split into k_nope (B, N, P)
             and v (B, N, V).
        7. q (B, N, (P+R)) and k (B, N, (P+R)) matrices are assembled from
           q_nope, q_pe, k_nope, k_pe.
        8. Attention is computued with q, k, v.
        9. The attention computation returns (B, N, V), which is projected back
           to (B, H) using out projection.

        * if decode:

        3. Here's the change, we do not perform up the full up projection for
           q_c, and there is no up projection at all for kv_c. This is
           achieved by the technique of "weight absorption". The paper says
           "Fortunately, due to the associative law of matrix multiplication,
           we can absorb WUK into WUQ, and WUV into WO"
           * The q up projection turns (B, Lq) into (B, N, (P+R)), we split it
             into W_UQ (Lq, N, P) and W_QR (Lq, N, R).
           * The kv_c up projection turns (B, Lkv) into (B, N, (P+V)), we split
             it into W_UK (Lkv, N, P) and W_UV (Lkv, N, V).
           * The out projection shape W_O (N*V, H) turns (B, N, V) into (B, H).
           * We can precompute the product of W_UQ and W_UK into
             W_UQ_UK (Lq, N, Lkv), which is possible due to QK^T operation in
             attention.
           * We can precompute the product of W_UV and W_O into
             W_UV_O (N, Lkv, H), which is possible due to V@O as the
             "epilogue" of attention
        4. We still need to compute q_pe (B, N, R) by applying W_QR to q_latent.
        5. q_pe, k_pe are then passed through rotary embeddings.
        6. kv_c and k_pe are concatenated and inserted into the cache
        7. By applying W_UQ_UK to q_latent, we have the new q_nope of shape
           (B, N, Lkv).
        8. q (B, N, (Lkv+R)), k (B, (Lkv+R)) are assembled from q_nope, q_pe,
           kv_a, k_pe. v (B, Lkv) is exactly the same vector as kv_a.
        9. The attention is computed with q, k, v. Note that we just performed
           a MQA attention with (LKv+R) as our head dim.
        10. The KV cache is updated using the new entries k (B, N, (Lkv+R)),
           which included the v and rope values.
        11. The attention computation returns (B, N, Lkv), which is projected
           back to (B, H) using W_UV_O.

    From @tsu-bin's calculation, we only want to use the absorption technique
    for decode. The prefill algorithm should still use the up-projected MHA
    for less flops and memory usage.

    """
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
        layer_idx: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        self.rotary_emb = rotary_emb
        self.use_yarn_rope = isinstance(rotary_emb,
                                        DeepseekScalingRotaryEmbedding)
        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.sparsity = os.getenv('SPARSITY')
        if (self.sparsity != None ):
            print(f"--->sparsity: {self.sparsity}")
        self.top_k = os.getenv('TOP_K')
        # print(f"--->top_k: {self.top_k}")
        self.layer_idx = layer_idx
        # print(f"--->layer_idx: {self.layer_idx}")
        # self.save_score = 1
        self.max_bsz = 64
        self.max_seq_len = 4000
        self.kv_c_buffer = None
        self.k_pe_buffer = None
        self.select_kv_c_buffer = None
        self.select_k_pe_buffer = None

    # @torch.compile
    # def new_get_kv_cache(
    #     self,
    #     kv_cache: torch.Tensor,
    #     attn_metadata:T
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # 获取解码序列的batch大小
    #     bsz = attn_metadata.num_decode_tokens
    #     max_seq_len = max(attn_metadata.seq_lens)
    #     local_kv_c_cache = self.kv_c_buffer[:bsz*max_seq_len, :].view(bsz, max_seq_len, self.kv_lora_rank)
    #     local_k_pe_cache = self.k_pe_buffer[:bsz*max_seq_len, :].view(bsz, max_seq_len, self.qk_rope_head_dim)
    #     # kv_c_cache = torch.zeros((bsz, max_seq_len, self.kv_lora_rank), device=kv_cache.device, dtype=kv_cache.dtype)
    #     # k_pe_cache = torch.zeros((bsz, max_seq_len, 64), device=kv_cache.device, dtype=kv_cache.dtype)

    #     for i in range(bsz):
    #         # kv_c_and_k_pe_cache = torch.zeros(0, kv_cache.shape[2]).to(kv_cache.device)
    #         pos = 0
    #         seq_len = attn_metadata.seq_lens[i]
    #         block_table = attn_metadata.block_tables[i]
    #         slot_mapping = attn_metadata.slot_mapping[i]

    #         block_size = kv_cache.shape[1]
    #         num_blocks = (seq_len + block_size - 1) // block_size
    #         for index in range(num_blocks):
    #             j = block_table[index]
    #             if index == num_blocks - 1:
    #                 end_slot = slot_mapping - j*kv_cache.shape[1] + 1
    #                 local_kv_c_cache[i][pos:pos+end_slot, :] = kv_cache[j][:end_slot, :self.kv_lora_rank]
    #                 local_k_pe_cache[i][pos:pos+end_slot, :] = kv_cache[j][:end_slot, self.kv_lora_rank:]
    #                 break
    #             local_kv_c_cache[i][pos:pos+block_size, :] = kv_cache[j][:, :self.kv_lora_rank]
    #             local_k_pe_cache[i][pos:pos+block_size, :] = kv_cache[j][:, self.kv_lora_rank:]
    #             pos += block_size
    #     return local_kv_c_cache, local_k_pe_cache

    def new_get_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: T
    ) -> torch.Tensor:
        # 获取解码序列的batch大小
        bsz = attn_metadata.num_decode_tokens
        max_seq_len = max(attn_metadata.seq_lens)
        BLOCK_SIZE = kv_cache.shape[1]
        BLOCK_DV = self.kv_lora_rank
        BLOCK_DPE = kv_cache.shape[2] - self.kv_lora_rank
        k_buffer = torch.empty((bsz, max_seq_len, BLOCK_DV), device=kv_cache.device, dtype=kv_cache.dtype)
        v_buffer = torch.empty((bsz, max_seq_len, BLOCK_DPE), device=kv_cache.device, dtype=kv_cache.dtype)
        grid = (
            bsz,
            BLOCK_SIZE
        )

        _get_kv_cache_kernel[grid](
            kv_cache,
            attn_metadata.block_tables,
            attn_metadata.seq_lens_tensor,
            k_buffer,
            v_buffer,
            BLOCK_SIZE,
            attn_metadata.block_tables.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            BLOCK_DV=BLOCK_DV,
            BLOCK_DPE=BLOCK_DPE
        )

        return k_buffer, v_buffer

    def get_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: T
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取解码序列的batch大小
        bsz = attn_metadata.num_decode_tokens
        max_seq_len = max(attn_metadata.seq_lens)
        BLOCK_SIZE = kv_cache.shape[1]
        BLOCK_DV = self.kv_lora_rank
        BLOCK_DPE = self.qk_rope_head_dim
        k_buffer = torch.empty((bsz, max_seq_len, BLOCK_DV), device=kv_cache.device, dtype=kv_cache.dtype)
        pe_buffer = torch.empty((bsz, max_seq_len, BLOCK_DPE), device=kv_cache.device, dtype=kv_cache.dtype)
        
        grid = (
            bsz,
            BLOCK_SIZE
        )

        _get_kv_cache_kernel[grid](
            kv_cache,
            attn_metadata.block_tables,
            attn_metadata.seq_lens_tensor,
            k_buffer,
            pe_buffer,
            BLOCK_SIZE,
            attn_metadata.block_tables.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            pe_buffer.stride(0),
            pe_buffer.stride(1),
            BLOCK_DV,
            BLOCK_DPE
        )

        return k_buffer, pe_buffer

    def _v_up_proj_and_o_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_UV_O):
                output_parallel = apply_fp8_linear_generic(
                    x.flatten(start_dim=1), self.W_UV_O, self.W_UV_O_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape)
            else:
                output_parallel = torch.matmul(x.flatten(start_dim=1),
                                               self.W_UV_O)
            if self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel
            return output
        else:
            x = torch.einsum("bnl,lnv->bnv", x, self.W_UV)
            return self.o_proj(x.reshape(-1,
                                         self.num_heads * self.v_head_dim))[0]

    def _q_proj_and_k_up_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_Q_UK):
                return apply_fp8_linear_generic(
                    x, self.W_Q_UK, self.W_Q_UK_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape).view(
                        -1, self.num_heads, self.kv_lora_rank)
            return torch.matmul(x, self.W_Q_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)
        else:
            x = torch.matmul(x, self.W_Q)\
                .view(-1, self.num_heads, self.qk_nope_head_dim)
            return torch.einsum("bnp,lnp->bnl", x, self.W_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # TODO(lucas) This is very gross, we need a more wide scale refactor of
        # all the FP8 code with a more standard way of
        # defining schemes/group-shapes, we should also potentially force
        # quant_methods to support a decompress function
        #
        # returns input_group_shape, weight_group_shape
        def get_scale_group_shapes_for_fp8(layer: LinearBase) -> \
            Tuple[Tuple[int, int], Tuple[int, int]]:
            if isinstance(layer.quant_method, Fp8LinearMethod):
                if layer.quant_method.block_quant:
                    weight_block_size = \
                        layer.quant_method.quant_config.weight_block_size
                    # per-token-group (1, X), block-quantized (X, Y)
                    return (1, weight_block_size[-1]), weight_block_size
                else:
                    return (-1, -1), (-1, -1)  # per-tensor, per-tensor
            elif isinstance(layer.quant_method, CompressedTensorsLinearMethod)\
                and isinstance(layer.scheme, CompressedTensorsW8A8Fp8):
                # this is hacky but we always assume the for
                # CompressedTensorsW8A8Fp8 the input is dynamic per-token
                # we ignore if it is static-per-tensor since we are going to
                # requantize after later anyways
                strategy = layer.scheme.strategy
                if strategy == QuantizationStrategy.TENSOR:
                    return (1, -1), (-1, -1)  # per-token, per-tensor
                elif strategy == QuantizationStrategy.CHANNEL:
                    return (1, -1), (-1, 1)  # per-token, per-channel
                else:
                    raise NotImplementedError(
                        f"QuantizationStrategy.{strategy} is not supported for "
                        "fp8 MLA, please run with VLLM_MLA_DISABLE=1")
            else:
                raise NotImplementedError(
                    "Can't determine scale group shapes for "
                    f"{layer.quant_method}, please run with VLLM_MLA_DISABLE=1"
                )

        def get_layer_weight(layer):
            if hasattr(layer, "weight"):
                return layer.weight
            elif hasattr(layer, "qweight"):
                return layer.qweight
            else:
                raise AttributeError(
                    f"Layer '{layer}' has neither weight nor qweight")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        weight_dtype = get_layer_weight(self.kv_b_proj).dtype
        assert get_layer_weight(self.o_proj).dtype == weight_dtype
        assert get_layer_weight(self.q_proj).dtype == weight_dtype

        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_proj_weight = get_and_maybe_dequant_weights(self.q_proj).T\
                .view(-1, self.num_heads, self.qk_head_dim)

        # can be W_Q or W_UQ depending q_lora_rank, the former if
        # q_lora_rank is None, the latter otherwise. From the Attention backend
        # perspective though we call these both W_Q and rely on the layer
        # to pass in the correct matrix
        W_Q = q_proj_weight[..., :self.qk_nope_head_dim]
        self.W_QR = q_proj_weight[..., self.qk_nope_head_dim:]\
            .flatten(start_dim=1).contiguous()

        # W_QR is small so for simplicity we dont bother requantizing it
        self.W_QR = self.W_QR.to(act_dtype)

        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            requantization_enabled = not envs.VLLM_MLA_DISABLE_REQUANTIZATION
            if is_fp8(weight_dtype) and requantization_enabled:
                # This assumes it wise to requantize using the same group shapes
                # (i.e. strategy, per-tensor, per-channel, block etc.) that the
                # weights were originally quantized
                requant_input_group_shape, requant_weight_group_shape = \
                    get_scale_group_shapes_for_fp8(self.q_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.kv_b_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.o_proj)
                self.reqaunt_input_group_shape = requant_input_group_shape
                self.reqaunt_weight_group_shape = requant_weight_group_shape

            #
            # Perform matrix-absorption following
            #     https://github.com/flashinfer-ai/flashinfer/pull/551
            # for decode, as a result we end up with absorbed weights for decode
            # and another copy of raw weights for prefill.
            #
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # We absorb `W_UK` into `W_Q` resulting in either W_Q_UK or W_UQ_UK
            # depending q_lora_rank, the former if q_lora_rank is None, the
            # latter otherwise
            # basically if q_lora_rank is none we are absorbing into q_proj
            # instead of UQ
            W_Q_UK = torch.einsum("qnd,lnd -> qnl", W_Q, W_UK)\
                .flatten(start_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_Q_UK, W_Q_UK_scales = scaled_quantize(
                    W_Q_UK,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_Q_UK = W_Q_UK.T.contiguous()
                self.W_Q_UK_scales = W_Q_UK_scales.T.contiguous()
            else:
                self.W_Q_UK = W_Q_UK.to(act_dtype)

            W_O = get_and_maybe_dequant_weights(self.o_proj)\
                .view(-1, self.num_heads, self.v_head_dim)
            W_UV_O = torch.einsum("lnd,hnd -> nlh", W_UV, W_O)\
                .flatten(start_dim=0, end_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_UV_O, W_UV_O_scales = scaled_quantize(
                    W_UV_O,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_UV_O = W_UV_O.T.contiguous()
                self.W_UV_O_scales = W_UV_O_scales.T.contiguous()
            else:
                self.W_UV_O = W_UV_O.to(act_dtype)

            self.tp_size = get_tensor_model_parallel_world_size()
            self.rank = get_tensor_model_parallel_rank()
        else:
            if is_fp8(weight_dtype):
                raise NotImplementedError(
                    "Currently fp8 requires matrix absorption")

            self.W_UV = W_UV
            self.W_UK = W_UK
            self.W_Q = W_Q.flatten(start_dim=1)

    @abstractmethod
    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError


    def choose_top_k(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        mean = torch.mean(logits, dim=1)
        _, indices = torch.topk(mean, top_k)
        return indices
    
    def compute_attn_score(
        self,
        q_pe: torch.Tensor, 
        q_nope: torch.Tensor,
        k_pe_cache: torch.Tensor,
        kv_c_cache: torch.Tensor,
        seq_lens: List,
    ) -> torch.Tensor:
        attn_weights_pe = torch.matmul(q_pe, k_pe_cache.transpose(1, 2))
        attn_weights_nope = torch.matmul(q_nope, kv_c_cache.transpose(1, 2))
        logits = (attn_weights_pe + attn_weights_nope) * self.scale
        seq_lens = torch.tensor(seq_lens, device=logits.device)
        mask = torch.arange(logits.size(2), device=logits.device) >= seq_lens.view(-1, 1, 1)
        logits.masked_fill_(mask, float('-inf'))
        attn_weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(q_nope.dtype)
        return attn_weights
    
    def flash_attn_score(
        self,
        q_pe: torch.Tensor, 
        q_nope: torch.Tensor,
        k_pe_cache: torch.Tensor,
        kv_c_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1) #[bsz, num_heads, head_dim]
        k =  torch.cat([kv_c_cache, k_pe_cache], dim=-1).unsqueeze(2) #[bsz, num_heads, head_dim]
        s_q = 1
        h_q = q.shape[-2]
        h_kv = 1
        dv = self.kv_lora_rank #self.kv_lora_rank
        d=self.kv_lora_rank + self.qk_rope_head_dim
        b = q.shape[0]

        max_seqlen = cache_seqlens.max().item()
        import triton
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

        block_size = 64
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32, device=q.device
        ).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=k.dtype, device=q.device)
        for i in range(b):
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, :cache_seqlens[i].item()] = k[i][:cache_seqlens[i],:,:]
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
        flash_out, flash_lse = flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=True,
            softmax_scale=self.scale
        )
        return flash_out.squeeze(1), flash_lse.squeeze(-1)
    
    def get_folder(self, folder):
        # 定义时间格式解析器
        time_format = "%Y-%m-%d-%H-%M-%S"
        datetime_objects = []
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir():    
                    # 将字符串转换为datetime对象
                    datetime_objects.append(datetime.strptime(os.path.basename(entry.path), time_format))
                    
                    # 找到最晚时间
                    latest_datetime = max(datetime_objects)
        return f"{folder}/{latest_datetime.strftime(time_format)}"


    def select_kv_cache(
        self,
        kv_c_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        attn_metadata: T,
        indices: torch.Tensor,
        sparsity:float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取解码序列的batch大小
        bsz = attn_metadata.num_decode_tokens
        top_k = [math.ceil(x * sparsity) for x in attn_metadata.seq_lens]
        max_len = max(top_k)
        top_k = torch.tensor(top_k, device=kv_c_cache.device, dtype=attn_metadata.seq_lens_tensor.dtype)
        selected_k_pe_cache = torch.empty((bsz, max_len, k_pe_cache.size(2)), device=k_pe_cache.device, dtype=k_pe_cache.dtype)
        selected_kv_c_cache = torch.empty((bsz, max_len, kv_c_cache.size(2)), device=kv_c_cache.device, dtype=kv_c_cache.dtype)
        BLOCK_SIZE = 16
        BLOCK_DV = self.kv_lora_rank
        BLOCK_DPE = self.qk_rope_head_dim

        grid = (
            bsz,
            BLOCK_SIZE
        )
        _select_kv_cache_kernel[grid](
            kv_c_cache,
            top_k,
            indices,
            selected_kv_c_cache,
            BLOCK_SIZE,
            indices.stride(0),
            kv_c_cache.stride(0),
            kv_c_cache.stride(1),
            selected_kv_c_cache.stride(0),
            selected_kv_c_cache.stride(1),
            BLOCK_DV,
        )

        _select_kv_cache_kernel[grid](
            k_pe_cache,
            top_k,
            indices,
            selected_k_pe_cache,
            BLOCK_SIZE,
            indices.stride(0),
            k_pe_cache.stride(0),
            k_pe_cache.stride(1),
            selected_k_pe_cache.stride(0),
            selected_k_pe_cache.stride(1),
            BLOCK_DPE,
        )
        return selected_kv_c_cache, selected_k_pe_cache

    @torch.compile
    def get_index(self, attn_weights, top_k):
        index = []
        for i in range(len(top_k)):
            index.append(torch.sort(torch.topk(torch.mean(attn_weights[i],dim=0), top_k[i]).indices).values)
        return index

    def new_absorbed_forward(
        self,
        q_nope: torch.Tensor, 
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        top_k: str = None,
        sparsity: str = None,
    ):
        if(attn_metadata.seq_lens != None):
            kv_c_cache, k_pe_cache = self.new_get_kv_cache(kv_cache, attn_metadata)
            attn_weights = self.compute_attn_score(q_pe, q_nope, k_pe_cache, kv_c_cache, attn_metadata.seq_lens) 
            top_k = [math.ceil(x * float(sparsity)) for x in attn_metadata.seq_lens]
            index = self.get_index(attn_weights, top_k)
            selected_k_pe_cache, selected_kv_c_cache = self.select_kv_cache(kv_c_cache, k_pe_cache, index)
            selected_attn_weights = self.compute_attn_score(q_pe, q_nope, selected_k_pe_cache, selected_kv_c_cache, top_k)
            attn_output = torch.matmul(selected_attn_weights, selected_kv_c_cache)
            return self._v_up_proj_and_o_proj(attn_output)
        else:
            return torch.zeros((attn_metadata.num_decode_tokens, self.v_head_dim*self.num_heads), device=self.W_UV_O.device, dtype=self.W_UV_O.dtype)

    def absorbed_forward(
        self,
        q_nope: torch.Tensor, 
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        top_k: str = None,
        sparsity: str = None,
    ):
        if(attn_metadata.seq_lens != None):
            bsz = attn_metadata.num_decode_tokens
            # kv_c_cache, k_pe_cache = self.new_get_kv_cache(kv_cache, attn_metadata)
            kv_c_cache, k_pe_cache = self.get_kv_cache(kv_cache, attn_metadata)
            attn_weights = self.compute_attn_score(q_pe, q_nope, k_pe_cache, kv_c_cache, attn_metadata.seq_lens) 

            # if(self.layer_idx != None and self.save_score):
            #     cur_dir = os.path.abspath(os.getcwd())
            #     main_folder = f"{cur_dir}/attn_scores"
            #     folder = self.get_folder(main_folder)
            #     file_name = f"{folder}/layer_{self.layer_idx+1}/tp_{self.rank}.csv"
            #     if os.path.exists(file_name):
            #         with open(file_name, mode='a', newline='') as file:
            #             writer = csv.writer(file)
            #             for i in range(attn_weights.size(1)):
            #                 writer.writerow(attn_weights[0][i].to(torch.float32).cpu().numpy())
            #     else:
            #         with open(file_name, mode='w', newline='') as file:
            #             writer = csv.writer(file)
            #             for i in range(attn_weights.size(1)):
            #                 writer.writerow(attn_weights[0][i].to(torch.float32).cpu().numpy())
            # return

            if(top_k != None):
                #sparse top-k
                top_k = int(top_k)
                index = self.choose_top_k(attn_weights, top_k)
                index = torch.sort(index, dim=1).values
                selected_k_pe_cache = k_pe_cache[torch.arange(k_pe_cache.size(0)).unsqueeze(1), index]
                selected_kv_c_cache = kv_c_cache[torch.arange(kv_c_cache.size(0)).unsqueeze(1), index]
                selected_attn_weights = self.compute_attn_score(q_pe, q_nope, selected_k_pe_cache, selected_kv_c_cache, [top_k]*len(attn_metadata.seq_lens))
                attn_output = torch.matmul(selected_attn_weights, selected_kv_c_cache)   
            elif (sparsity != None):
                # sparsity = float(sparsity)
                # import math
                # top_k = [math.ceil(x * sparsity) for x in attn_metadata.seq_lens]
                # max_len = max(top_k)
                # selected_k_pe_cache = torch.zeros((bsz, max_len, k_pe_cache.size(2)), device=k_pe_cache.device, dtype=k_pe_cache.dtype)
                # selected_kv_c_cache = torch.zeros((bsz, max_len, kv_c_cache.size(2)), device=kv_c_cache.device, dtype=kv_c_cache.dtype)
                # for i in range(len(attn_metadata.seq_lens)):
                #     index = torch.topk(torch.mean(attn_weights[i],dim=0), top_k[i]).indices
                #     index = torch.sort(index).values
                #     selected_k_pe_cache[i][:top_k[i], :] = k_pe_cache[i][index]
                #     selected_kv_c_cache[i][:top_k[i], :] = kv_c_cache[i][index]
                # selected_attn_weights = self.compute_attn_score(q_pe, q_nope, selected_k_pe_cache, selected_kv_c_cache, top_k)
                # attn_output = torch.matmul(selected_attn_weights, selected_kv_c_cache)
                # logger.info("=====sparse!!!!!=====")
                sparsity = float(sparsity)
                top_k = [math.ceil(x * sparsity) for x in attn_metadata.seq_lens]
                mean_attn_weights = torch.mean(attn_weights,dim=1)
                _, indices = torch.sort(mean_attn_weights, dim=1,descending=True)
                selected_kv_c_cache, selected_k_pe_cache = self.select_kv_cache(kv_c_cache, k_pe_cache, attn_metadata, indices, sparsity)
                selected_attn_weights = self.compute_attn_score(q_pe, q_nope, selected_k_pe_cache, selected_kv_c_cache, top_k)
                attn_output = torch.matmul(selected_attn_weights, selected_kv_c_cache)
            else:
                attn_output = torch.matmul(attn_weights, kv_c_cache)
        else:
            attn_output = torch.zeros((attn_metadata.num_decode_tokens, self.v_head_dim*self.num_heads), device=self.W_UV_O.device, dtype=self.W_UV_O.dtype)
        return self._v_up_proj_and_o_proj(attn_output) 
    
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")

        is_decode = attn_metadata.decode_metadata is not None
        is_prefill = attn_metadata.prefill_metadata is not None

        if (is_decode and is_prefill):
            raise NotImplementedError(
                "chunked prefill is not supported for MLAImplBase")

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions")

        if is_decode:
            q_nope = self._q_proj_and_k_up_proj(hidden_states_or_q_c)
            q_pe = torch.matmul(hidden_states_or_q_c, self.W_QR)\
                .view(-1, self.num_heads, self.qk_rope_head_dim)
            q_pe, k_pe = self.rotary_emb(attn_metadata.input_positions, q_pe,
                                         k_pe)
        else:
            assert is_prefill
            q = self.q_proj(hidden_states_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)

            # TODO(lucas): there must be a nicer way to write this line
            q[..., self.qk_nope_head_dim:], k_pe = \
                self.rotary_emb(
                    attn_metadata.input_positions,
                    q[..., self.qk_nope_head_dim:], k_pe)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if attn_metadata.prefill_metadata is not None:
            return self._forward_prefill(q, k_c_normed, k_pe, attn_metadata)

        # if (self.sparsity != None and self.kv_c_buffer == None):
        #     self.kv_c_buffer = torch.empty((self.max_bsz*self.max_seq_len, self.kv_lora_rank), device=kv_cache.device, dtype=kv_cache.dtype)
        #     self.k_pe_buffer = torch.empty((self.max_bsz*self.max_seq_len, kv_cache.shape[2] - self.kv_lora_rank), device=kv_cache.device, dtype=kv_cache.dtype)
        #     self.select_kv_c_buffer = torch.empty((math.ceil(self.max_bsz*self.max_seq_len*float(self.sparsity)), self.kv_lora_rank), device=kv_cache.device, dtype=kv_cache.dtype)
        #     self.select_k_pe_buffer = torch.empty((math.ceil(self.max_bsz*self.max_seq_len*float(self.sparsity)), kv_cache.shape[2] - self.kv_lora_rank), device=kv_cache.device, dtype=kv_cache.dtype)

        if attn_metadata.decode_metadata is not None:
            # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if(self.sparsity != None):
                # return self.new_absorbed_forward(q_nope, q_pe, kv_cache, attn_metadata, top_k=self.top_k, sparsity=self.sparsity)
                return self.absorbed_forward(q_nope, q_pe, kv_cache, attn_metadata, top_k=self.top_k, sparsity=self.sparsity)
            else:
                return self._forward_decode(q_nope, q_pe, kv_cache, attn_metadata)

    # Optional common flash-attn based prefill
    def _forward_prefill_flash(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        seq_start_loc: torch.Tensor,
        max_prefill_seq_len: int,
    ) -> torch.Tensor:

        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=seq_start_loc,
            cu_seqlens_k=seq_start_loc,
            max_seqlen_q=max_prefill_seq_len,
            max_seqlen_k=max_prefill_seq_len,
            softmax_scale=self.scale,
            causal=True,
        )
        attn_output = attn_output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(attn_output)[0]