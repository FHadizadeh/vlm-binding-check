import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_ID = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/mmd/models/Qwen2.5-VL-7B-Instruct",
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_qwen_model_and_processor(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: str = "bfloat16",
    device_map=0,
    allow_download: bool = False,
):
    if dtype not in DTYPES:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Choose from {sorted(DTYPES)}."
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=DTYPES[dtype],
        device_map=device_map,
        local_files_only=not allow_download,
    )
    model.eval()

    # For this project we intentionally keep the model on a single GPU.
    # Activation patching with CPU/disk offload is both much slower and more
    # fragile because hooks would cross device boundaries.
    if device_map != "auto":
        meta_params = [
            name
            for name, parameter in model.named_parameters()
            if parameter.device.type == "meta"
        ]
        if meta_params:
            preview = ", ".join(meta_params[:5])
            raise RuntimeError(
                "Model loading left parameters on the meta device even though "
                "single-device loading was requested. Restart the Jupyter "
                "kernel to release any previously loaded model, then retry. "
                f"Examples: {preview}"
            )

    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=False,
        local_files_only=not allow_download,
    )

    return model, processor


def build_messages(
    image_path: Path,
    question: str,
):
    image_uri = image_path.resolve().as_uri()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_uri,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]


def get_model_input_device(model) -> torch.device:
    """Return a real device for processor outputs; never return the meta device.

    The default loader keeps the whole 7B model on GPU 0. This fallback also
    makes failures clearer if a model was loaded with Accelerate offloading.
    """
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except Exception:
        pass

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def prepare_inputs(
    model,
    processor,
    image_path: Path,
    question: str,
):
    messages = build_messages(
        image_path=image_path,
        question=question,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs.to(get_model_input_device(model))


@torch.inference_mode()
def predict_one(
    model,
    processor,
    image_path: Path,
    question: str,
    max_new_tokens: int = 8,
):
    inputs = prepare_inputs(
        model=model,
        processor=processor,
        image_path=image_path,
        question=question,
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def get_language_model(model):
    candidates = [
        ("model.language_model", lambda: model.model.language_model),
        ("language_model", lambda: model.language_model),
        ("model.model", lambda: model.model.model),
        ("model", lambda: model.model),
    ]

    for name, getter in candidates:
        try:
            module = getter()
        except AttributeError:
            continue

        if hasattr(module, "layers"):
            return module, name

    raise AttributeError(
        "Could not locate the Qwen language model. Expected a module with "
        "a '.layers' attribute under model.model.language_model or a "
        "compatible fallback path."
    )


def get_decoder_layers(model):
    language_model, language_model_path = get_language_model(model)
    return language_model.layers, language_model_path


def last_attended_token_index(inputs) -> int:
    attention_mask = inputs.get("attention_mask")

    if attention_mask is None:
        return int(inputs["input_ids"].shape[1] - 1)

    attended = torch.nonzero(
        attention_mask[0],
        as_tuple=False,
    ).flatten()

    if attended.numel() == 0:
        raise ValueError("attention_mask contains no attended tokens.")

    return int(attended[-1].item())


def image_token_layout(model, inputs):
    if "input_ids" not in inputs:
        raise ValueError("Processor output does not contain input_ids.")

    if "image_grid_thw" not in inputs:
        raise ValueError("Processor output does not contain image_grid_thw.")

    image_token_id = model.config.image_token_id
    image_positions = torch.nonzero(
        inputs["input_ids"][0] == image_token_id,
        as_tuple=False,
    ).flatten()

    if inputs["image_grid_thw"].shape[0] != 1:
        raise ValueError(
            "This patching implementation expects exactly one image per sample."
        )

    grid_t, grid_h, grid_w = [
        int(x)
        for x in inputs["image_grid_thw"][0].tolist()
    ]

    merge = int(model.config.vision_config.spatial_merge_size)

    if grid_h % merge != 0 or grid_w % merge != 0:
        raise ValueError(
            f"image_grid_thw={(grid_t, grid_h, grid_w)} is not divisible "
            f"by spatial_merge_size={merge}."
        )

    llm_t = grid_t
    llm_h = grid_h // merge
    llm_w = grid_w // merge
    expected_tokens = llm_t * llm_h * llm_w

    if image_positions.numel() != expected_tokens:
        raise ValueError(
            "Image-token count does not match image_grid_thw after spatial "
            f"merge: found {image_positions.numel()} image tokens but "
            f"expected {expected_tokens} from grid "
            f"{(grid_t, grid_h, grid_w)} with merge={merge}."
        )

    return {
        "image_positions": image_positions,
        "grid_t": grid_t,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "spatial_merge_size": merge,
        "llm_t": llm_t,
        "llm_h": llm_h,
        "llm_w": llm_w,
    }


def union_bbox(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
) -> Tuple[float, float, float, float]:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    return (
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2),
    )


def bbox_to_image_token_region(
    layout: Dict[str, object],
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    dilation: int = 0,
) -> Dict[str, int]:
    """Map a pixel-space bbox to an inclusive region on the post-merge LLM image grid.

    The returned coordinates are useful both for selecting sequence positions and
    for logging exactly which visual grid cells were intervened on.
    """
    if dilation < 0:
        raise ValueError("dilation must be non-negative.")

    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image_size={image_size}.")

    if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
        raise ValueError(
            f"bbox={bbox} lies outside image_size={image_size}."
        )

    llm_t = int(layout["llm_t"])
    llm_h = int(layout["llm_h"])
    llm_w = int(layout["llm_w"])

    x_start = math.floor(x1 / width * llm_w)
    x_stop = math.ceil(x2 / width * llm_w) - 1
    y_start = math.floor(y1 / height * llm_h)
    y_stop = math.ceil(y2 / height * llm_h) - 1

    x_start = max(0, min(llm_w - 1, x_start - dilation))
    x_stop = max(0, min(llm_w - 1, x_stop + dilation))
    y_start = max(0, min(llm_h - 1, y_start - dilation))
    y_stop = max(0, min(llm_h - 1, y_stop + dilation))

    if x_stop < x_start or y_stop < y_start:
        raise ValueError(
            f"bbox={bbox} maps to an empty visual-token region."
        )

    return {
        "t_start": 0,
        "t_stop": llm_t - 1,
        "y_start": int(y_start),
        "y_stop": int(y_stop),
        "x_start": int(x_start),
        "x_stop": int(x_stop),
    }



def image_token_positions_from_region(
    layout: Dict[str, object],
    region: Dict[str, int],
) -> torch.Tensor:
    """Return sequence positions for an inclusive region on the post-merge image grid."""
    llm_t = int(layout["llm_t"])
    llm_h = int(layout["llm_h"])
    llm_w = int(layout["llm_w"])
    image_positions = layout["image_positions"]

    bounds = {
        "t_start": 0,
        "t_stop": llm_t - 1,
        "y_start": 0,
        "y_stop": llm_h - 1,
        "x_start": 0,
        "x_stop": llm_w - 1,
    }
    for key, limit in bounds.items():
        value = int(region[key])
        if key.endswith("_start"):
            if value < 0:
                raise ValueError(f"Region coordinate {key}={value} is negative.")
        else:
            if value > limit:
                raise ValueError(
                    f"Region coordinate {key}={value} exceeds grid limit {limit}."
                )

    if (
        int(region["t_stop"]) < int(region["t_start"])
        or int(region["y_stop"]) < int(region["y_start"])
        or int(region["x_stop"]) < int(region["x_start"])
    ):
        raise ValueError(f"Invalid empty region: {region}")

    grid = image_positions.reshape(llm_t, llm_h, llm_w)
    selected = grid[
        int(region["t_start"]):int(region["t_stop"]) + 1,
        int(region["y_start"]):int(region["y_stop"]) + 1,
        int(region["x_start"]):int(region["x_stop"]) + 1,
    ]
    return selected.reshape(-1)


def grid_region_overlap_size(
    region_a: Dict[str, int],
    region_b: Dict[str, int],
) -> int:
    """Return the number of post-merge grid cells shared by two regions."""
    overlap = 1
    for axis in ("t", "y", "x"):
        start = max(
            int(region_a[f"{axis}_start"]),
            int(region_b[f"{axis}_start"]),
        )
        stop = min(
            int(region_a[f"{axis}_stop"]),
            int(region_b[f"{axis}_stop"]),
        )
        if stop < start:
            return 0
        overlap *= stop - start + 1
    return int(overlap)


def _grid_region_shape(region: Dict[str, int]) -> Tuple[int, int, int]:
    return tuple(
        int(region[f"{axis}_stop"]) - int(region[f"{axis}_start"]) + 1
        for axis in ("t", "y", "x")
    )


def _region_center(region: Dict[str, int], axis: str) -> float:
    return (
        int(region[f"{axis}_start"])
        + int(region[f"{axis}_stop"])
    ) / 2.0


def _best_region_for_shape_near_anchor(
    layout: Dict[str, object],
    shape: Tuple[int, int, int],
    anchor_region: Dict[str, int],
    forbidden_region: Optional[Dict[str, int]],
    require_anchor_overlap: bool = True,
) -> Optional[Dict[str, int]]:
    """Find the best placement of ``shape`` near ``anchor_region``.

    The candidate is always kept inside the post-merge image-token grid.  If a
    forbidden region is supplied, candidates overlapping it are rejected.  The
    ranking first maximizes overlap with the anchor and then minimizes center
    distance to the anchor.  This is intentionally exhaustive because the Qwen
    image-token grid in this project is small (typically 16x16).
    """
    dims = (
        int(layout["llm_t"]),
        int(layout["llm_h"]),
        int(layout["llm_w"]),
    )
    if any(size <= 0 or size > dim for size, dim in zip(shape, dims)):
        return None

    anchor_centers = {
        axis: _region_center(anchor_region, axis)
        for axis in ("t", "y", "x")
    }

    best_region = None
    best_rank = None
    t_size, y_size, x_size = shape

    for t_start in range(dims[0] - t_size + 1):
        for y_start in range(dims[1] - y_size + 1):
            for x_start in range(dims[2] - x_size + 1):
                candidate = {
                    "t_start": int(t_start),
                    "t_stop": int(t_start + t_size - 1),
                    "y_start": int(y_start),
                    "y_stop": int(y_start + y_size - 1),
                    "x_start": int(x_start),
                    "x_stop": int(x_start + x_size - 1),
                }

                if (
                    forbidden_region is not None
                    and grid_region_overlap_size(candidate, forbidden_region) != 0
                ):
                    continue

                anchor_overlap = grid_region_overlap_size(candidate, anchor_region)
                if require_anchor_overlap and anchor_overlap == 0:
                    continue

                center_distance_sq = 0.0
                for axis in ("t", "y", "x"):
                    center_distance_sq += (
                        _region_center(candidate, axis) - anchor_centers[axis]
                    ) ** 2

                rank = (
                    -int(anchor_overlap),
                    float(center_distance_sq),
                    int(t_start),
                    int(y_start),
                    int(x_start),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_region = candidate

    return best_region


def matched_region_around_anchor(
    layout: Dict[str, object],
    reference_region: Dict[str, int],
    anchor_region: Dict[str, int],
    forbidden_region: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Return an exact-shape control region near an anchor.

    With ``forbidden_region`` set, the window is shifted as far as necessary to
    eliminate target overlap while still touching the distractor anchor.  This
    is the first-choice matched control used by the runner.
    """
    shape = _grid_region_shape(reference_region)
    region = _best_region_for_shape_near_anchor(
        layout,
        shape,
        anchor_region,
        forbidden_region,
        require_anchor_overlap=True,
    )
    if region is None:
        raise ValueError(
            "No exact-shape matched region can stay target-disjoint while "
            "still overlapping this distractor anchor."
        )
    return region


def robust_matched_region_around_anchors(
    layout: Dict[str, object],
    reference_region: Dict[str, int],
    anchors: Sequence[Tuple[int, Dict[str, int]]],
    forbidden_region: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Choose a robust target-disjoint matched-distractor control.

    Fallback order is deliberately conservative:

    1. Same grid shape/token count, shifting around the preferred distractor.
    2. Same grid shape/token count around another distractor object.
    3. Same token count with a different rectangular spatial shape.
    4. Fewer tokens (largest feasible region first) around a distractor.
    5. As a final non-crashing fallback, a target-disjoint region nearest the
       preferred distractor, even if it no longer intersects the anchor.

    ``anchors`` must be ordered with the preferred control object first.  The
    returned metadata records which fallback was used so downstream analyses
    can stratify or exclude non-exact controls if desired.
    """
    if not anchors:
        raise ValueError("At least one distractor anchor is required.")

    ref_shape = _grid_region_shape(reference_region)
    ref_t, ref_y, ref_x = ref_shape
    requested_n = int(ref_t * ref_y * ref_x)
    llm_h = int(layout["llm_h"])
    llm_w = int(layout["llm_w"])

    def build_info(
        region: Dict[str, int],
        anchor_index: int,
        strategy: str,
        exact_shape: bool,
    ) -> Dict[str, object]:
        actual_shape = _grid_region_shape(region)
        actual_n = int(actual_shape[0] * actual_shape[1] * actual_shape[2])
        anchor_id, anchor_region = anchors[anchor_index]
        return {
            "object_id": int(anchor_id),
            "strategy": strategy,
            "requested_shape": [int(v) for v in ref_shape],
            "actual_shape": [int(v) for v in actual_shape],
            "requested_n_tokens": requested_n,
            "actual_n_tokens": actual_n,
            "exact_shape": bool(exact_shape),
            "exact_token_count": bool(actual_n == requested_n),
            "target_overlap_tokens": int(
                grid_region_overlap_size(region, forbidden_region)
            ),
            "anchor_overlap_tokens": int(
                grid_region_overlap_size(region, anchor_region)
            ),
        }

    # 1-2) Exact shape.  The helper searches *all* placements, so an overlap at
    # the centered location simply causes the window to move farther away.
    for anchor_index, (_, anchor_region) in enumerate(anchors):
        region = _best_region_for_shape_near_anchor(
            layout,
            ref_shape,
            anchor_region,
            forbidden_region,
            require_anchor_overlap=True,
        )
        if region is not None:
            strategy = (
                "exact_shape_preferred_object"
                if anchor_index == 0
                else "exact_shape_alternate_object"
            )
            return region, build_info(
                region, anchor_index, strategy, exact_shape=True
            )

    # Enumerate alternative y/x rectangle shapes with the same number of
    # spatial cells.  Keep the temporal extent unchanged.
    spatial_area = int(ref_y * ref_x)
    exact_count_shapes = []
    for y_size in range(1, llm_h + 1):
        if spatial_area % y_size != 0:
            continue
        x_size = spatial_area // y_size
        if x_size < 1 or x_size > llm_w:
            continue
        shape = (ref_t, int(y_size), int(x_size))
        if shape == ref_shape:
            continue
        shape_distance = abs(y_size - ref_y) + abs(x_size - ref_x)
        exact_count_shapes.append((shape_distance, shape))
    exact_count_shapes.sort(key=lambda item: (item[0], item[1]))

    # 3) Same token count but reshape the window.
    for anchor_index, (_, anchor_region) in enumerate(anchors):
        for _, shape in exact_count_shapes:
            region = _best_region_for_shape_near_anchor(
                layout,
                shape,
                anchor_region,
                forbidden_region,
                require_anchor_overlap=True,
            )
            if region is not None:
                strategy = (
                    "exact_count_reshaped_preferred_object"
                    if anchor_index == 0
                    else "exact_count_reshaped_alternate_object"
                )
                return region, build_info(
                    region, anchor_index, strategy, exact_shape=False
                )

    # 4) If exact matching is geometrically impossible, use the largest
    # target-disjoint region with fewer tokens.  Prefer shapes close to the
    # target shape at each token count.
    for spatial_count in range(spatial_area - 1, 0, -1):
        smaller_shapes = []
        for y_size in range(1, llm_h + 1):
            if spatial_count % y_size != 0:
                continue
            x_size = spatial_count // y_size
            if x_size < 1 or x_size > llm_w:
                continue
            shape = (ref_t, int(y_size), int(x_size))
            shape_distance = abs(y_size - ref_y) + abs(x_size - ref_x)
            smaller_shapes.append((shape_distance, shape))
        smaller_shapes.sort(key=lambda item: (item[0], item[1]))

        for anchor_index, (_, anchor_region) in enumerate(anchors):
            for _, shape in smaller_shapes:
                region = _best_region_for_shape_near_anchor(
                    layout,
                    shape,
                    anchor_region,
                    forbidden_region,
                    require_anchor_overlap=True,
                )
                if region is not None:
                    strategy = (
                        "reduced_count_preferred_object"
                        if anchor_index == 0
                        else "reduced_count_alternate_object"
                    )
                    return region, build_info(
                        region, anchor_index, strategy, exact_shape=False
                    )

    # 5) Extremely defensive fallback.  In the current synthetic dataset this
    # should never be needed because the target occupies only a small fraction
    # of the 16x16 image-token grid.  We still avoid aborting a long run.
    preferred_anchor_id, preferred_anchor = anchors[0]
    fallback_shape = (1, 1, 1)
    region = _best_region_for_shape_near_anchor(
        layout,
        fallback_shape,
        preferred_anchor,
        forbidden_region,
        require_anchor_overlap=False,
    )
    if region is None:
        raise RuntimeError(
            "The target region covers the entire image-token grid; no "
            "target-disjoint negative control exists."
        )

    info = {
        "object_id": int(preferred_anchor_id),
        "strategy": "nearest_disjoint_single_token_background_fallback",
        "requested_shape": [int(v) for v in ref_shape],
        "actual_shape": [1, 1, 1],
        "requested_n_tokens": requested_n,
        "actual_n_tokens": 1,
        "exact_shape": False,
        "exact_token_count": bool(requested_n == 1),
        "target_overlap_tokens": 0,
        "anchor_overlap_tokens": int(
            grid_region_overlap_size(region, preferred_anchor)
        ),
    }
    return region, info

def bbox_to_image_token_positions(
    layout: Dict[str, object],
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    dilation: int = 0,
) -> torch.Tensor:
    region = bbox_to_image_token_region(
        layout=layout,
        bbox=bbox,
        image_size=image_size,
        dilation=dilation,
    )

    return image_token_positions_from_region(layout, region)


def answer_token_variants(tokenizer, answer: str) -> List[int]:
    variants = [
        answer,
        answer.capitalize(),
        f" {answer}",
        f" {answer.capitalize()}",
    ]

    token_ids = []

    for variant in variants:
        ids = tokenizer.encode(
            variant,
            add_special_tokens=False,
        )

        if len(ids) == 1:
            token_ids.append(int(ids[0]))

    token_ids = list(dict.fromkeys(token_ids))

    if not token_ids:
        raise ValueError(
            f"No single-token spelling variant found for answer '{answer}'. "
            "The current scorer assumes at least one of answer/Answer/' answer'/"
            "' Answer' is a single token."
        )

    return token_ids


def build_answer_token_map(
    tokenizer,
    answer_options: Iterable[str],
) -> Dict[str, List[int]]:
    return {
        answer: answer_token_variants(tokenizer, answer)
        for answer in answer_options
    }


def score_answer_options(
    next_token_logits: torch.Tensor,
    answer_token_map: Dict[str, List[int]],
):
    if next_token_logits.ndim != 1:
        raise ValueError(
            "next_token_logits must be a 1D vocabulary-sized tensor."
        )

    log_masses = {}

    for answer, ids in answer_token_map.items():
        answer_logits = next_token_logits[
            torch.tensor(ids, device=next_token_logits.device)
        ]
        log_masses[answer] = torch.logsumexp(
            answer_logits.float(),
            dim=0,
        )

    answers = list(answer_token_map)
    stacked = torch.stack([log_masses[a] for a in answers])
    candidate_probs = torch.softmax(stacked, dim=0)

    full_probs = torch.softmax(
        next_token_logits.float(),
        dim=-1,
    )

    vocab_probs = {}
    for answer, ids in answer_token_map.items():
        vocab_probs[answer] = float(
            full_probs[
                torch.tensor(ids, device=full_probs.device)
            ].sum().item()
        )

    candidate_prob_dict = {
        answer: float(candidate_probs[i].item())
        for i, answer in enumerate(answers)
    }

    prediction = max(
        answers,
        key=lambda answer: candidate_prob_dict[answer],
    )

    return {
        "log_mass": {
            answer: float(log_masses[answer].item())
            for answer in answers
        },
        "candidate_prob": candidate_prob_dict,
        "vocab_prob": vocab_probs,
        "prediction": prediction,
    }
