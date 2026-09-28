# POUNCE Gate-1 reduced flash comparison

`pounce_gate1_smoke_v2.json` is the schema-valid cross-repository artifact for
DiscOpt #1526 and the final Gate-1 item in POUNCE #776. It preserves a real
POUNCE route-matrix result and adds GDP, SOS1, and Scholtes records at 250,
268, 300, 324, and 350 K.

The artifact was executed against:

- POUNCE `7c550f0192fde407e644c4f9f75898c79297f83f` (the companion
  `pounce-flash-results/2` schema commit, PR #972);
- DiscOpt `ac5dcf760b5e65aae58c9b8ac6a5bb66378d97cd` (the implementation
  commit immediately preceding this generated artifact).

All ten GDP/SOS1 cells are globally certified at one node. All five Scholtes
cells are `local_optimal`, with `bound=null`, `gap=null`, and
`gap_certified=false`. Every cell matches the POUNCE regime and algebraic root
branches. Across all 15 cells, the worst errors/residuals are:

| measurement | maximum |
|---|---:|
| beta error versus oracle | `4.054e-7` |
| material balance / normalization | `6.260e-13` |
| isofugacity | `2.712e-14` |
| active-root EOS | `1.565e-13` |
| root selection | `1.256e-12` |
| source complementarity | `6.490e-9` |

The zero objective and zero exact gap are not performance evidence. The
comparison is decided by the physical point, explicit root-count and
nontriviality identities, regime, and source residuals. Lowered-row residuals
are retained separately in every record.

Regenerate the DiscOpt half from the tracked version-2 artifact (or substitute a
fresh POUNCE version-1 artifact) with:

```bash
python discopt_benchmarks/scripts/issue1526_pounce_flash.py \
  discopt_benchmarks/results/issue1526/pounce_gate1_smoke_v2.json \
  discopt_benchmarks/results/issue1526/pounce_gate1_smoke_v2.json
```

Pass `--full` for POUNCE's 34-temperature path. The runner refuses a missing
POUNCE point rather than substituting an unstamped oracle seed, gates every
exact warm start through `accept_local_incumbent`, and validates the output
against POUNCE's packaged version-2 schema before writing it.
