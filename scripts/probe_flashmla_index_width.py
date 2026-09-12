"""Does the FlashMLA kv-cache kernel really require an index width of index_topk?

SGLang asserts ``indices.shape[-1] == nsa_index_topk`` and calls the requirement
a kernel constraint.  The dense-warmup design hinges on whether a wider index
table is admissible, so ask the kernel directly for a few widths.
"""

import torch
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

PAGE = 64
KV_DIM = 656  # 512 nope + 64 rope in FP8 with per-chunk scales
D_QK = 576
D_V = 512
HEADS = 64


def try_width(width: int, valid: int, rows: int = 8, pages: int = 256) -> str:
    device = "cuda"
    q = torch.randn(rows, 1, HEADS, D_QK, device=device, dtype=torch.bfloat16)
    kv = torch.zeros(pages, PAGE, 1, KV_DIM, device=device, dtype=torch.uint8)
    # Valid prefix then -1 padding, exactly the layout per_head_topk_paged emits.
    indices = torch.full((rows, 1, width), -1, device=device, dtype=torch.int32)
    indices[:, :, :valid] = torch.arange(valid, device=device, dtype=torch.int32)
    seqlens = torch.full((rows,), valid, device=device, dtype=torch.int32)
    try:
        meta, splits = get_mla_metadata(
            cache_seqlens=seqlens,
            num_q_tokens_per_head_k=HEADS,
            num_heads_k=1,
            num_heads_q=HEADS,
            is_fp8_kvcache=True,
            topk=width,
        )
        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            cache_seqlens=seqlens,
            head_dim_v=D_V,
            tile_scheduler_metadata=meta,
            num_splits=splits,
            softmax_scale=1.0,
            indices=indices,
            block_table=torch.empty((rows, 0), dtype=torch.int32, device=device),
            is_fp8_kvcache=True,
        )
        torch.cuda.synchronize()
        return f"OK out={tuple(out.shape)}"
    except Exception as error:  # noqa: BLE001 - reporting the kernel's own message
        return f"FAIL {type(error).__name__}: {str(error).splitlines()[0][:200]}"


if __name__ == "__main__":
    # The kernel asserts topk % TOPK_BLOCK_SIZE == 0; find the block size.
    for width in (64, 128, 192, 256, 512, 768, 1024, 2048, 2304, 3072, 4096, 8192):
        valid = min(width, 700)
        print(f"width={width:5d} valid={valid:5d} -> {try_width(width, valid)}", flush=True)
