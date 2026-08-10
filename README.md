# VLM Multi-Feature Binding

Controlled synthetic experiments for studying how Vision-Language Models identify an object from multiple visual attributes and how that computation changes under localized counterfactual interventions.

Each object has three attributes:

- color
- shape
- size

For every sample, one attribute is queried and the other two attributes identify the target.

Examples:

- `What is the size of the purple triangle?`
- `What is the color of the small triangle?`
- `What is the shape of the large purple object?`

With `--query_attribute mixed`, queries cycle through `size`, `color`, and `shape`.

The identifying attributes are:

| Queried attribute | Identifying attributes |
|---|---|
| size | shape + color |
| color | shape + size |
| shape | color + size |

## Conditions

The three conditions form a controlled hierarchy in the number of single-attribute shortcuts available for locating the target.

### 1. `redundant_cues`

Both identifying attributes are individually unique.

For a size question about a purple triangle:

- `triangle` alone identifies the target.
- `purple` alone identifies the target.
- the conjunction `purple + triangle` also identifies the target.

This is the easiest condition because either single cue is sufficient.

### 2. `single_cue_ambiguous`

Exactly one identifying attribute is made ambiguous by one distractor, while the other identifying attribute remains unique.

For half of the samples within each queried-attribute stream, the primary identifier is ambiguous. For the other half, the secondary identifier is ambiguous.

Therefore the model loses one single-cue shortcut but still has one remaining single-cue route to the target.

### 3. `conjunctive_binding`

One distractor shares the primary identifying attribute and another distractor shares the secondary identifying attribute.

Neither identifying attribute alone is sufficient. The conjunction is required to locate the target.

This is the binding-focused condition.

## Paired scene families

The conditions are generated as paired scene families.

When the same `seed`, `n`, and `query_attribute` are used for all three conditions, records with the same `family_id` share:

- the same target attributes,
- the same queried attribute,
- the same question and answer,
- the same target position,
- the same four object positions,
- the same base distractor family.

The only clean-scene changes across conditions are the identifier values needed to remove single-cue shortcuts.

For a paired family:

- `redundant_cues`: neither relevant distractor shares a target identifier,
- `single_cue_ambiguous`: exactly one relevant distractor is changed to share one target identifier,
- `conjunctive_binding`: both relevant distractors are changed, one for each identifier.

The fourth distractor is identical across all three conditions and shares neither identifying attribute. Queried-attribute frequencies are controlled separately so that answer-frequency shortcuts are not systematic.


## Queried-value frequency control

The queried attribute is frequency-controlled independently from the binding condition.

For size questions, every clean scene contains exactly:

- 2 small objects,
- 2 large objects.

Therefore neither size value is unique or more frequent.

For color and shape questions, every clean scene uses a `2-1-1` frequency profile among the values that appear. To prevent a simple rule such as "the repeated value is the answer" or "the unique value is the answer":

- for half of the occurrences of each answer value, the target answer is the repeated value (`target_repeated`),
- for the other half, a non-target value is repeated and the target answer occurs once (`non_target_repeated`).

This balancing is done separately for each answer value, not only globally.

The queried-attribute multiset is preserved exactly across the three paired conditions because condition construction changes only the identifying attributes.

## Geometry control

The 3x3 grid uses wider spacing than the maximum object side length. Clean and counterfactual bounding boxes are validated so that objects do not overlap and remain inside the canvas. This avoids introducing occlusion as an accidental difficulty difference between conditions.

## Balanced targets and answers

For `n=108` with `--query_attribute mixed`:

- 36 questions ask for size,
- 36 ask for color,
- 36 ask for shape.

Within each queried attribute, the correct answers are balanced:

- size: 18 small / 18 large,
- shape: 12 square / 12 circle / 12 triangle,
- color: 6 examples of each color.

The same balancing is preserved across all three conditions.

## Counterfactual construction

Each clean sample has exactly one matched counterfactual.

Only the queried attribute of the same target object changes. Everything else stays fixed:

- same question,
- same target object ID,
- same target position,
- same distractors,
- same identifying attributes.

Therefore the correct answer changes while the intervention remains localized to one attribute of one target object.

Examples:

### Size

Clean target: `small purple triangle`

Counterfactual target: `large purple triangle`

Question in both: `What is the size of the purple triangle?`

### Color

Clean target: `small purple triangle`

Counterfactual target: `small red triangle`

Question in both: `What is the color of the small triangle?`

### Shape

Clean target: `small purple triangle`

Counterfactual target: `small purple square`

Question in both: `What is the shape of the small purple object?`

## Generate the three paired datasets

Use the same seed for all conditions.

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

## Validate each dataset

```bash
python src/validate_dataset.py --data data/synth_v4/redundant_cues
python src/validate_dataset.py --data data/synth_v4/single_cue_ambiguous
python src/validate_dataset.py --data data/synth_v4/conjunctive_binding
```

## Validate pairing across conditions

```bash
python src/validate_paired_conditions.py \
  --redundant data/synth_v4/redundant_cues \
  --single data/synth_v4/single_cue_ambiguous \
  --conjunctive data/synth_v4/conjunctive_binding
```

## Inspect generated scenes

```bash
python src/inspect_dataset.py \
  --data data/synth_v4/conjunctive_binding \
  --n 12
```

## Qwen baseline evaluation

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

`run_baseline.py` is kept as a compatibility wrapper around `run_qwen_baseline.py`.

## Counterfactual evaluation

```bash
python src/run_qwen_counterfactuals.py \
  --data data/synth_v4/conjunctive_binding \
  --out results/conjunctive_counterfactuals.csv
```

A pair is marked `usable_for_patching` when:

- the clean prediction is correct,
- the counterfactual prediction is correct,
- the correct answer changes.

## Inspect counterfactual pairs

```bash
python src/inspect_cf_pairs.py \
  --data data/synth_v4/conjunctive_binding \
  --csv results/conjunctive_counterfactuals.csv \
  --n 20
```

Optional filters include:

- `--cf_type size_swap`
- `--queried_attribute color`
- `--ambiguous_identifier shape`
- `--only_failures`
