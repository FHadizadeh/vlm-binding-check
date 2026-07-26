# Initial Qwen2.5-VL-3B Baseline

Binary size labels: small / large.

| Condition | Accuracy |
|---|---:|
| unique_shape | 95.37% |
| multi_same_shape | 96.30% |
| compositional_distractor | 90.74% |

Observation: The binary version removes the medium-size ambiguity. The model is near-ceiling on unique and multi-same-shape settings, while compositional distractors create a modest accuracy drop.