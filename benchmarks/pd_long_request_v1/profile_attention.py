# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile one layer of paged FlashAttention for equal-length prefill chunks.

Each query attends to an existing context and its causal share of the new
chunk. K/V already contain both parts. The timed region reuses these buffers
and excludes model projections, KV writes, TP, and inter-engine transfer.
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple


class Shape(NamedTuple):
    context_length: int
    chunk_size: int
    batch_size: int

    @property
    def kv_length(self) -> int:
        return self.context_length + self.chunk_size

    def pages_per_sequence(self, page_size: int) -> int:
        return (self.kv_length + page_size - 1) // page_size


def shape_record(shape: Shape, args: argparse.Namespace) -> dict:
    context, chunk, batch = shape
    pages = shape.pages_per_sequence(args.page_size) * batch
    queries = chunk * batch
    pairs = batch * (context * chunk + chunk * (chunk + 1) // 2)
    # Q and out, plus both K and V. FP16 and BF16 are both two bytes.
    buffer_bytes = 4 * args.head_dim * (
        queries * args.query_heads + pages * args.page_size * args.kv_heads
    )
    return {
        **shape._asdict(),
        "kv_length": shape.kv_length,
        "batch_query_tokens": queries,
        "batch_causal_qk_pairs_per_head": pairs,
        "pages": pages,
        "page_size": args.page_size,
        "layout": args.layout,
        "dtype": args.dtype,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "q_out_kv_bytes": buffer_bytes,
    }


def make_call(torch, flash_attn, scheduler_metadata_fn, shape, args, fa_version):
    context, chunk, batch = shape
    device = "cuda:0"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pages_per_sequence = shape.pages_per_sequence(args.page_size)
    pages = batch * pages_per_sequence
    q = torch.randn(
        batch * chunk, args.query_heads, args.head_dim, device=device, dtype=dtype
    )
    out = torch.empty_like(q)

    # vLLM exposes [page, 2, token, head, dim] with layout-specific strides.
    if args.layout == "HND":
        cache = torch.randn(
            pages, 2, args.kv_heads, args.page_size, args.head_dim,
            device=device, dtype=dtype,
        ).permute(0, 1, 3, 2, 4)
    else:
        cache = torch.randn(
            pages, 2, args.page_size, args.kv_heads, args.head_dim,
            device=device, dtype=dtype,
        )
    k, v = cache.unbind(1)
    table = torch.arange(pages, dtype=torch.int32, device=device)
    table = table.view(pages_per_sequence, batch).T.contiguous()
    kv_lengths = torch.full((batch,), context + chunk, dtype=torch.int32, device=device)
    q_offsets = torch.arange(
        0, (batch + 1) * chunk, chunk, dtype=torch.int32, device=device
    )
    metadata = None
    num_splits = 1 if args.batch_invariant else 0
    if fa_version == 3 and not args.batch_invariant:
        metadata = scheduler_metadata_fn(
            batch_size=batch,
            max_seqlen_q=chunk,
            max_seqlen_k=shape.kv_length,
            num_heads_q=args.query_heads,
            num_heads_kv=args.kv_heads,
            headdim=args.head_dim,
            cache_seqlens=kv_lengths,
            qkv_dtype=dtype,
            cu_seqlens_q=q_offsets,
            page_size=args.page_size,
            causal=True,
            num_splits=num_splits,
        )

    def invoke():
        return flash_attn(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=q_offsets,
            max_seqlen_q=chunk,
            seqused_k=kv_lengths,
            max_seqlen_k=shape.kv_length,
            block_table=table,
            causal=True,
            scheduler_metadata=metadata,
            fa_version=fa_version,
            num_splits=num_splits,
        )

    return invoke, (q, k, v, table)


def check_paged_causal(torch, invoke, tensors, shape):
    context, chunk, batch = shape
    q, k, v, table = tensors
    query = q.view(batch, chunk, q.shape[1], q.shape[2]).float()
    blocks = table.long()
    keys = k[blocks].flatten(1, 2)[:, :shape.kv_length].float()
    values = v[blocks].flatten(1, 2)[:, :shape.kv_length].float()
    groups = q.shape[1] // k.shape[-2]
    keys = keys.repeat_interleave(groups, dim=2)
    values = values.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, keys) * q.shape[-1] ** -0.5
    query_pos = context + torch.arange(chunk, device=q.device)
    key_pos = torch.arange(shape.kv_length, device=q.device)
    future = key_pos[None, :] > query_pos[:, None]
    probabilities = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    reference = torch.einsum("bhqk,bkhd->bqhd", probabilities, values)
    actual = invoke().view_as(reference).float()
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)


def measure_ms(torch, invoke, args, label):
    for _ in range(args.warmup):
        invoke()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.repeats)
    ]
    # CUDA events are initialized on their first record.
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    annotation = torch.cuda.nvtx.range(label) if args.nvtx else nullcontext()
    with annotation:
        for start, end in events:
            start.record()
            invoke()
            end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def profile(args, shapes, records):
    import torch

    if not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA vLLM is required for timing. Use --dry-run on CPU.")

    from vllm import __version__, envs
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        is_fa_version_supported,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    fa_version = (
        get_flash_attn_version(head_size=args.head_dim)
        if args.fa_version == "auto"
        else int(args.fa_version)
    )
    if fa_version is None:
        raise RuntimeError("vLLM could not select a supported FlashAttention version.")
    if not is_fa_version_supported(fa_version):
        reason = fa_version_unsupported_reason(fa_version)
        raise RuntimeError(f"FlashAttention {fa_version} is unavailable: {reason}")
    args.batch_invariant = envs.VLLM_BATCH_INVARIANT

    def make(shape):
        return make_call(
            torch, flash_attn_varlen_func, get_scheduler_metadata,
            shape, args, fa_version,
        )

    with torch.inference_mode():
        if args.check:
            check_shape = Shape(context_length=35, chunk_size=5, batch_size=2)
            invoke, tensors = make(check_shape)
            check_paged_causal(torch, invoke, tensors, check_shape)
            del invoke, tensors
        for shape, record in zip(shapes, records, strict=True):
            invoke, tensors = make(shape)
            label = "paged_flash_attention/L={}/c={}/B={}".format(*shape)
            samples = measure_ms(torch, invoke, args, label)
            record.update(
                fa_version=fa_version,
                elapsed_ms_median=statistics.median(samples),
                elapsed_ms_p95=sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                elapsed_ms_samples=samples,
            )
            del invoke, tensors

    return {
        "vllm_version": __version__,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "fa_version": fa_version,
        "fa3_aot_metadata": fa_version == 3 and not args.batch_invariant,
        "batch_invariant": args.batch_invariant,
        "causal_reference_checked": args.check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[0, 512, 4096, 8192, 16384, 28672],
    )
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layout", choices=("HND", "NHD"), default="HND")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--fa-version", choices=("auto", "2", "3", "4"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    parser.add_argument("--csv", type=Path, help="Also write a summary by shape")
    parser.add_argument("--check", action="store_true", help="Check a GPU reference")
    parser.add_argument("--nvtx", action="store_true", help="Mark each shape in Nsight")
    parser.add_argument("--dry-run", action="store_true", help="Print shape sizes only")
    args = parser.parse_args()

    if min(args.context_lengths) < 0 or min(args.chunk_sizes + args.batch_sizes) < 1:
        parser.error("Contexts must be nonnegative, and chunks and batches positive.")
    if min(args.query_heads, args.kv_heads) < 1 or args.query_heads % args.kv_heads:
        parser.error("Heads must be positive, with query heads divisible by KV heads.")
    if args.head_dim < 8 or args.head_dim % 8:
        parser.error("The head dimension must be a positive multiple of 8.")
    if args.page_size < 16 or args.page_size % 16:
        parser.error("The page size must be a positive multiple of 16.")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("Warm-up must be nonnegative and repeats must be positive.")
    if args.output and args.csv and args.output.resolve() == args.csv.resolve():
        parser.error("JSON and CSV output paths must differ.")

    shapes = [
        Shape(*values)
        for values in itertools.product(
            args.context_lengths, args.chunk_sizes, args.batch_sizes
        )
    ]
    records = [shape_record(shape, args) for shape in shapes]
    if any(
        max(row["kv_length"], row["pages"], row["batch_query_tokens"]) >= 2**31
        for row in records
    ):
        parser.error("Shape exceeds the int32 range of FlashAttention index tensors.")
    metadata = {
        "backend": "vllm_flash_attn_varlen_paged",
        "causal_alignment": "bottom_right",
        "page_order": "interleaved_no_shared_pages",
        "scope": "single_layer_reused_kv_no_append",
        "dry_run": args.dry_run,
        "requested_fa_version": args.fa_version,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    if not args.dry_run:
        metadata.update(profile(args, shapes, records))

    output = json.dumps({"metadata": metadata, "results": records}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            fields = [key for key in records[0] if key != "elapsed_ms_samples"]
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
