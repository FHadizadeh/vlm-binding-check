# VLM Multi-Feature Binding Check

A minimal starter repo for generating a controlled synthetic dataset for causal tracing of multi-feature binding in VLMs.

Core task examples:

- `What is the size of the blue square?`
- `What is the size of the square?`

The dataset is designed for later causal tracing / activation patching experiments: clean image, counterfactual image, target object, distractors, and metadata are stored explicitly.

## Quick start

```bash
pip install -r requirements.txt

python src/generate_dataset.py \
  --condition compositional_distractor \
  --n 100 \
  --seed 0 \
  --out data/synth_v1

python src/inspect_dataset.py --data data/synth_v1 --n 12
```

This creates:

```text
data/synth_v1/
  images/
  counterfactuals/
  metadata.jsonl
  manifest.json
  samples_grid.png
```

## Conditions

### unique_shape
Only one object has the queried shape, so color is redundant.

Example: `What is the size of the square?`

### multi_same_shape
There are multiple objects with the queried shape, so color is needed to select the correct one.

Example: `What is the size of the blue square?`

### compositional_distractor
There is both a same-color distractor and a same-shape distractor, so the model must use color ∩ shape binding.

Example: target = blue square, distractors = blue circle and red square.

## Counterfactuals

For each sample, the generator tries to create:

- `size_swap`: same target identity but target size is changed.
- `color_swap`: colors are swapped between target and same-shape distractor when available.
- `shape_swap`: shapes are swapped between target and same-color distractor when available.

These are intended for later activation patching.
