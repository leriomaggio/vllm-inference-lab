# L2 — PagedAttention from scratch

A decoding model keeps a KV cache that grows by one token per step. Storing each
sequence's cache contiguously wastes memory: you must reserve the maximum possible
length up front, and you cannot share identical prefixes between sequences.
**PagedAttention** borrows the operating-system trick of virtual memory — store
the cache in fixed-size **blocks (pages)** from a shared pool, and keep a
per-sequence **block table** mapping logical blocks to physical ones.

This layer implements that data structure for a single sequence and validates it
against the L0 reference cache, then deliberately breaks it to show the failure
mode.

## The pieces

| File | Role |
|------|------|
| `vllab/paged/block_table.py` | `BlockTable`: logical→physical mapping over a free pool; `ensure_capacity`, `locate`. |
| `vllab/paged/paged_attention.py` | `PagedKVCache` (block pool + append + gather + attend) and `paged_decode`. |
| `vllab/paged/faults.py` | `FaultyBlockTable` (a `BlockTable` subclass adding `corrupt`), `block_table_fault_demo`, `FaultReport`: repoint one logical block and show the output corrupts silently. |

### Storage layout

- Pools are `(num_blocks, H, block_size, D)` tensors for K and V.
- Appending token `t` computes `(logical, slot) = divmod(t, block_size)`, allocates
  a physical block for `logical` if needed, and writes `(H, D)` into
  `pool[physical, :, slot, :]`.
- Attending **gathers** the scattered blocks back into a contiguous `(H, T, D)`
  view *through the block table* and runs the ordinary attention oracle.

The gather is where the indirection lives: logical block `l` is read from
`pool[table[l]]`. If `table[l]` is wrong, the gather silently returns the wrong
KV — the shape and dtype are untouched.

## The invariant

> Paged decode must equal the non-paged reference decode.

Paging is a *storage* optimisation; it must not change the result. This is what
lets the implementation be validated: any divergence is a bug in the paging, not a
property of the model.

## Validation

```bash
vllab paged-demo
pytest tests/test_paged.py -q
```

### Measured (H, T, D) = (3, 17, 16), block_size 4, seed 0

```
paged vs non-paged reference: max_abs = 0.000e+00   (equal within fp64 noise)
block-table fault (logical 0: phys 0 -> 1): max_abs_diff = 4.995e-01,
    shape/dtype unchanged = True
```

Two facts sit side by side:

1. **Correct paging is invisible.** The paged decode matches the contiguous
   reference to `0.0` across block sizes 1, 3, 4, 8, 17 — the block boundaries
   leave no trace in the output.
2. **A one-entry table fault is silent but wrong.** Repointing a single logical
   block at a different physical block changes the output by `~0.5` while the
   tensor's shape and dtype are unchanged. Nothing crashes; no exception; latency
   is identical. Only a differential check against a reference catches it.

This is the paging analogue of the general theme: the dangerous failures do not
announce themselves in the type system or the timing — they show up only when you
compare numbers against something you trust.

## What the tests assert

- `test_block_table_allocation_and_locate` — capacity math and `(physical, slot)`
  resolution are correct;
- `test_block_table_pool_exhaustion_raises` — running out of blocks is a clean
  error, not silent corruption;
- `test_paged_equals_reference_incremental` — paged == reference across many block
  sizes;
- `test_paged_equals_oneshot_causal` — and == one-shot causal attention;
- `test_block_table_fault_is_silent_but_wrong` — the fault keeps the shape/dtype
  but moves the numbers by a wide margin.

## Extensions (not implemented here)

Real PagedAttention pools serve **many sequences** at once and share a physical
block between two sequences' tables to implement **prefix caching**. The single
sequence modelled here is enough to demonstrate the data structure and its
characteristic failure; multi-sequence pooling and prefix sharing are natural next
steps and are where the L3 engine (real vLLM) takes over.
