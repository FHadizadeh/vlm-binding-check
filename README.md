# VLM Multi-Feature Binding

Controlled synthetic experiments for studying **object–attribute binding and layer-wise causal computation in Vision–Language Models**.

The repository is designed around two complementary questions:

1. **Behavioral binding demand:** how does performance change when identifying a target requires increasingly stronger conjunction of visual cues?
2. **Mechanistic causal analysis:** where in the VLM is the clean/counterfactual answer difference causally recoverable across visual-token and text-token representations?

The current main model is **Qwen2.5-VL-7B-Instruct**.

Detailed verified results are recorded in [`notes/baseline_results.md`](notes/baseline_results.md).

---

## 1. Task

Each scene contains four objects. Every object has:

- color
- shape
- size

For each sample, one attribute is queried and the other two attributes identify the target.

Examples:

```text
What is the size of the purple triangle?
What is the color of the small triangle?
What is the shape of the large purple object?
```

With `--query_attribute mixed`, queries cycle through `size`, `color`, and `shape`.

| Queried attribute | Identifying attributes |
|---|---|
| size | shape + color |
| color | shape + size |
| shape | color + size |

---

## 2. Binding-demand hierarchy

### `redundant_cues`

Both identifying attributes are individually unique for the target. Either single cue is sufficient.

### `single_cue_ambiguous`

Exactly one identifying attribute is shared with a distractor. One single-cue route remains sufficient.

### `conjunctive_binding`

One distractor matches the first identifying attribute and another distractor matches the second. Neither cue alone identifies the target; their conjunction is required.

This hierarchy is generated as **paired scene families**, so the same `family_id` across conditions preserves the target, question, object positions, queried-attribute values, and base distractor family. Only the identifier values required to create the ambiguity condition are changed.

---

## 3. Dataset controls

### Balanced queried attributes

For `n=108` and `--query_attribute mixed`:

- 36 size questions
- 36 color questions
- 36 shape questions

Balanced answers:

- size: 18 small / 18 large
- shape: 12 square / 12 circle / 12 triangle
- color: 6 examples per color

### Queried-value frequency control

For size questions, each answer cycles through:

- `balanced_2_2`
- `target_majority_3_1`
- `target_minority_1_3`

For color and shape questions, values follow a `2-1-1` profile, balanced between:

- `target_repeated`
- `non_target_repeated`

The queried-value profile is preserved across paired conditions.

### Geometry control

Objects lie on a 3×3 grid with spacing larger than the maximum object size. Clean/counterfactual bounding boxes are validated to remain inside the canvas and not overlap.

---

## 4. Counterfactual construction

Every clean sample has one matched counterfactual.

The counterfactual changes **only the queried attribute of the same target object**.

Preserved:

- question
- target object ID
- target position
- distractors
- identifying attributes

Examples:

```text
size:
small purple triangle -> large purple triangle

color:
small purple triangle -> small red triangle

shape:
small purple triangle -> small purple square
```

A pair is usable for patching when:

- clean prediction is correct,
- counterfactual prediction is correct,
- clean and counterfactual answers differ.

> Current limitation: this intervention isolates the target queried-attribute pathway, but is not yet a pure binding-specific association swap. A future association-swap counterfactual should preserve the feature multiset while changing which attribute belongs to which object.

---

## 5. Generate paired datasets

Use the same `seed`, `n`, and `query_attribute` for all three conditions.

```bash
python src/generate_dataset.py \
  --condition redundant_cues \
  --n 108 \
  --seed 0 \
  --query_attribute mixed \
  --out data/synth_v4/redundant_cues

python src/generate_dataset.py \
  --condition single_cue_ambiguous \
  --n 108 \
  --seed 0 \
  --query_attribute mixed \
  --out data/synth_v4/single_cue_ambiguous

python src/generate_dataset.py \
  --condition conjunctive_binding \
  --n 108 \
  --seed 0 \
  --query_attribute mixed \
  --out data/synth_v4/conjunctive_binding
```

Validate:

```bash
python src/validate_dataset.py --data data/synth_v4/redundant_cues
python src/validate_dataset.py --data data/synth_v4/single_cue_ambiguous
python src/validate_dataset.py --data data/synth_v4/conjunctive_binding
```

Validate pairing:

```bash
python src/validate_paired_conditions.py \
  --redundant data/synth_v4/redundant_cues \
  --single data/synth_v4/single_cue_ambiguous \
  --conjunctive data/synth_v4/conjunctive_binding
```

Inspect scenes:

```bash
python src/inspect_dataset.py \
  --data data/synth_v4/conjunctive_binding \
  --n 12
```

---

## 6. Model setup

The current experiments use **Qwen2.5-VL-7B-Instruct**.

Pass the checkpoint explicitly:

```bash
--model_id /path/to/Qwen2.5-VL-7B-Instruct
```

or set:

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
```
---

## 7. Behavioral baseline

```bash
python src/run_qwen_baseline.py \
  --data data/synth_v4/redundant_cues \
  --out results/redundant.csv

python src/run_qwen_baseline.py \
  --data data/synth_v4/single_cue_ambiguous \
  --out results/single.csv

python src/run_qwen_baseline.py \
  --data data/synth_v4/conjunctive_binding \
  --out results/conjunctive.csv
```

`run_baseline.py` may be kept only as a compatibility wrapper if older commands still depend on it.

---

## 8. Counterfactual endpoint evaluation

Run before activation patching:

```bash
python src/run_qwen_counterfactuals.py \
  --data data/synth_v4/conjunctive_binding \
  --out results/conjunctive_counterfactuals.csv
```

The current conjunctive run produced 108/108 usable pairs.

Inspect pairs:

```bash
python src/inspect_cf_pairs.py \
  --data data/synth_v4/conjunctive_binding \
  --csv results/conjunctive_counterfactuals.csv \
  --n 20
```

---

## 9. Activation patching

`run_activation_patching.py` performs clean/CF interchange interventions.

Supported patch types:

### `target`

Patch the target visual-token region from counterfactual into clean.

For size counterfactuals, the mask uses the union of the clean and CF target boxes so that all changed spatial positions are covered.

### `all_image`

Patch all image tokens. Positive control.

### `last_token`

Patch the final prompt-token residual state from counterfactual into clean.

### `distractor`

Patch the raw bbox of an unchanged non-target object.

### `matched_distractor`

Patch a target-disjoint control region anchored on a distractor.

Priority:

1. same target-mask grid shape near the preferred distractor,
2. same shape near another distractor,
3. same token count with a reshaped rectangle,
4. only as a final fallback, a smaller region.

Per-row metadata records:

- selected control object,
- strategy,
- requested/actual mask shape,
- exact token-count status,
- target overlap,
- anchor overlap.

Target overlap is required to be zero.

### Layer locations

- LM-input patch: after vision encoder/projector embeddings are inserted into the multimodal sequence.
- Decoder patch: output residual stream (`resid_post`) of each LM layer.

Only one counterfactual decoder layer is cached at a time to keep the 7B run memory-safe.

---

## 10. Metrics

For clean answer \(a_c\) and counterfactual answer \(a_{cf}\):

```text
gap = score(a_cf) - score(a_c)
```

Normalized recovery:

```text
recovery =
    (patched_gap - clean_gap)
    / (cf_gap - clean_gap)
```

Interpretation:

```text
recovery ≈ 0  -> clean-like
recovery ≈ 1  -> counterfactual-like
```

The output also records:

- candidate-normalized `P(CF answer)`
- vocabulary probability/log-mass fields
- clean/CF/patch candidate distributions
- CF flip flags
- exact visual grid coordinates
- number of patched tokens
- target/control geometry metadata

Every patching CSV has a companion `<output>.meta.json` file containing run provenance.

---

## 11. Full conjunctive sweep

Verified full command:

```bash
python src/run_activation_patching.py \
  --data data/synth_v4/conjunctive_binding \
  --pairs_csv results/conjunctive_counterfactuals.csv \
  --out results/conjunctive_patching_full.csv \
  --layers 0:28:1 \
  --patch_types target all_image last_token distractor matched_distractor \
  --dilations 0
```

---

## 12. Current verified findings

Detailed tables and caveats: [`notes/baseline_results.md`](notes/baseline_results.md).

On 108 conjunctive pairs:

- early/mid target patching is strongly causal,
- visual-position recoverability drops sharply around L14–L16,
- last-token recoverability rises later and overtakes target recovery at about L20 overall,
- first crossover is approximately:
  - size: L19
  - shape: L21
  - color: L22
- raw and matched distractor controls remain near zero,
- strict same-shape/same-count/zero-overlap matched controls remain near zero,
- for small objects, expanding the target mask from 4 to 16 tokens recovers most of the missing effect.

The correct wording is **a shift in causal recoverability**, not proof that information literally moves between token positions.

---

## 13. Small-target dilation analysis

The full run revealed that small color/shape targets use a 4-token raw bbox. To test whether this bbox is too restrictive, the same 36 small-target pairs are rerun with larger spatial regions.

Prepare the subset from the full results using the reporting script or a short pandas filter, then run:

```bash
python src/run_activation_patching.py \
  --data data/synth_v4/conjunctive_binding \
  --pairs_csv results/conjunctive_small_target_pairs.csv \
  --out results/conjunctive_small_target_dilation12.csv \
  --layers 0:28:1 \
  --patch_types target matched_distractor \
  --dilations 1 2
```

Current result:

```text
dilation 0:  4 target tokens
dilation 1: 16 target-region tokens
dilation 2: 36 target-region tokens
```

Most of the recovery gain occurs from dilation 0 to 1.

Analyze the paired dilation run:

```bash
python src/analyze_dilation.py   --full_csv results/conjunctive_patching_full.csv   --dilation_csv results/conjunctive_small_target_dilation12.csv   --out_dir figures/dilation   --prefix conjunctive_small_target
```

This writes aggregate recovery/flip summaries, paired dilation deltas, and separate overall/color/shape plots.

---

## 14. Plotting and notebook reporting

### Standard plots

```python
%run src/plot_patching_results.py \
    --csv results/conjunctive_patching_full.csv \
    --out_dir figures/conjunctive_full \
    --prefix conjunctive_full \
    --by_attribute \
    --include_lm_input \
    --error ci95 \
    --dashboard \
    --recovery_distribution \
    --distribution_patch_types target all_image last_token distractor matched_distractor \
    --distribution_kind box \
    --distribution_points
```

### Report-style notebook output

```python
%run src/report_patching_notebook.py \
    --csv results/conjunctive_patching_full.csv \
    --out_dir figures/conjunctive_report \
    --prefix conjunctive_full \
    --pairs_csv results/conjunctive_counterfactuals.csv
```

The report script displays styled tables and figures inline in Jupyter and saves CSV/HTML/LaTeX report artifacts.

---

## 15. Recommended repository structure

```text
.
├── README.md
├── requirements.txt
├── configs/
├── notes/
│   └── baseline_results.md
├── src/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── validate_paired_conditions.py
│   ├── inspect_dataset.py
│   ├── inspect_cf_pairs.py
│   ├── prompts.py
│   ├── qwen_utils.py
│   ├── run_qwen_baseline.py
│   ├── run_qwen_counterfactuals.py
│   ├── run_activation_patching.py
│   ├── plot_patching_results.py
│   ├── report_patching_notebook.py
│   └── analyze_dilation.py
└── figures/
    └── conjunctive_full/
```

`run_baseline.py` is optional and should remain only if compatibility with older commands is useful.

Large generated data, checkpoints, and raw result CSVs can remain gitignored. Small publication/report figures and compact summary tables can be committed if desired.

---

## 16. Next experiments

1. Run the same causal pipeline on `redundant_cues` and `single_cue_ambiguous`.
2. Compare causal dynamics across the paired binding-demand hierarchy.
3. Build a binding-specific association-swap counterfactual that preserves the feature multiset.
4. If condition-dependent layer differences emerge, localize them further to attention/MLP/head components.
5. Compare successful and failed examples once task difficulty is high enough to produce meaningful failures.
