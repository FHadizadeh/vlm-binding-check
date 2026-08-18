import argparse
import gc
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import transformers
from PIL import Image
from tqdm import tqdm

from qwen_utils import (
    DEFAULT_MODEL_ID,
    bbox_to_image_token_positions,
    bbox_to_image_token_region,
    build_answer_token_map,
    get_decoder_layers,
    get_language_model,
    image_token_layout,
    image_token_positions_from_region,
    last_attended_token_index,
    robust_matched_region_around_anchors,
    load_qwen_model_and_processor,
    prepare_inputs,
    score_answer_options,
    union_bbox,
)


PATCH_TYPES = [
    "target",
    "all_image",
    "distractor",
    "matched_distractor",
    "last_token",
]

RESULT_SCHEMA_VERSION = 5
RESUME_REQUIRED_COLUMNS = {
    "result_schema_version",
    "sample_id",
    "cf_type",
    "scope",
    "layer",
    "patch_type",
    "dilation",
    "clean_candidate_probs",
    "cf_candidate_probs",
    "patched_candidate_probs",
    "target_clean_bbox",
    "target_cf_bbox",
    "control_object_id",
    "patch_grid_x_start",
    "patch_grid_x_stop",
    "target_reference_n_patch_tokens",
    "patch_token_count_difference_from_target",
    "matched_control_strategy",
    "matched_control_object_id",
    "matched_control_target_overlap_tokens",
}


def stable_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def none_if_nan(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def object_attributes(obj: Optional[dict]):
    if obj is None:
        return None
    return {
        key: obj.get(key)
        for key in ("shape", "color", "size")
        if key in obj
    }


def score_distribution_fields(prefix: str, scores: dict) -> dict:
    return {
        f"{prefix}_candidate_probs": stable_json(scores["candidate_prob"]),
        f"{prefix}_vocab_probs": stable_json(scores["vocab_prob"]),
        f"{prefix}_log_masses": stable_json(scores["log_mass"]),
    }


def patch_grid_fields(region: Optional[dict]) -> dict:
    if region is None:
        return {
            "patch_grid_t_start": None,
            "patch_grid_t_stop": None,
            "patch_grid_y_start": None,
            "patch_grid_y_stop": None,
            "patch_grid_x_start": None,
            "patch_grid_x_stop": None,
        }

    return {
        "patch_grid_t_start": int(region["t_start"]),
        "patch_grid_t_stop": int(region["t_stop"]),
        "patch_grid_y_start": int(region["y_start"]),
        "patch_grid_y_stop": int(region["y_stop"]),
        "patch_grid_x_start": int(region["x_start"]),
        "patch_grid_x_stop": int(region["x_stop"]),
    }


def full_image_region(layout: dict) -> dict:
    return {
        "t_start": 0,
        "t_stop": int(layout["llm_t"]) - 1,
        "y_start": 0,
        "y_stop": int(layout["llm_h"]) - 1,
        "x_start": 0,
        "x_stop": int(layout["llm_w"]) - 1,
    }


def validate_resume_output_schema(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return

    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    missing = RESUME_REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            "Cannot --resume this output because it was created with an older "
            "result schema. Missing columns: " + ", ".join(sorted(missing)) +
            ". Use a new --out file (recommended) or rerun from scratch."
        )

    first_row = pd.read_csv(path, nrows=1, usecols=["result_schema_version"])
    if first_row.empty:
        return
    existing_version = int(first_row.iloc[0]["result_schema_version"])
    if existing_version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "Cannot --resume an output created with result schema "
            f"v{existing_version}; this runner writes schema v{RESULT_SCHEMA_VERSION}. "
            "Use a new --out file (recommended) or rerun from scratch."
        )


def package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def write_run_metadata(
    args,
    model,
    processor,
    language_model_path: str,
    n_layers: int,
    selected_layers: Sequence[int],
):
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")

    existing = None
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = None

    created_at = (
        existing.get("created_at_utc")
        if isinstance(existing, dict)
        else None
    ) or datetime.now(timezone.utc).isoformat()

    manifest = None
    manifest_path = args.data / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        hf_device_map = {key: str(value) for key, value in hf_device_map.items()}

    try:
        model_parameter_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        model_parameter_dtype = None

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "last_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "data_dir": str(args.data.resolve()),
        "pairs_csv": str(args.pairs_csv.resolve()) if args.pairs_csv else None,
        "output_csv": str(args.out.resolve()),
        "dataset_manifest": manifest,
        "model_id": args.model_id,
        "model_config_name_or_path": getattr(model.config, "_name_or_path", None),
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "processor_use_fast": False,
        "requested_dtype": args.dtype,
        "model_parameter_dtype": model_parameter_dtype,
        "allow_download": bool(args.allow_download),
        "hf_device_map": hf_device_map,
        "language_model_path": language_model_path,
        "n_decoder_layers": int(n_layers),
        "selected_layers": [int(x) for x in selected_layers],
        "patch_types": list(args.patch_types),
        "dilations": [int(x) for x in args.dilations],
        "control_object_id": int(args.control_object_id),
        "matched_distractor_strategy": (
            "robust target-disjoint control. Priority: exact target grid shape "
            "near the preferred distractor; exact shape near another distractor; "
            "same token count with a reshaped rectangle; then the largest feasible "
            "smaller region. Every selected region is target-disjoint. Per-row "
            "strategy/object/token-count metadata records any fallback."
        ),
        "skip_input_patches": bool(args.skip_input_patches),
        "include_unusable": bool(args.include_unusable),
        "limit": args.limit,
        "max_pairs": args.max_pairs,
        "image_token_id": int(model.config.image_token_id),
        "spatial_merge_size": int(model.config.vision_config.spatial_merge_size),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "qwen_vl_utils_version": package_version("qwen-vl-utils"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_reported_by_torch": torch.version.cuda,
        "gpu_names": gpu_names,
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta_path


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open(
        "r",
        encoding="utf-8",
    ) as f:
        return [json.loads(line) for line in f]


def iter_counterfactuals(record):
    cfs = record.get("counterfactuals", {})

    if not isinstance(cfs, dict):
        raise ValueError(
            "run_activation_patching.py expects counterfactuals to be a dictionary."
        )

    for cf_type, cf in cfs.items():
        if cf is None:
            continue
        yield cf_type, cf


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def load_pair_table(path: Optional[Path]):
    if path is None:
        return None

    df = pd.read_csv(path)

    required = {
        "sample_id",
        "cf_type",
        "usable_for_patching",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Pairs CSV is missing required columns: {sorted(missing)}"
        )

    return {
        (str(row.sample_id), str(row.cf_type)): row._asdict()
        for row in df.itertuples(index=False)
    }


def find_object(objects: Sequence[dict], object_id: int) -> dict:
    matches = [
        obj
        for obj in objects
        if int(obj["object_id"]) == int(object_id)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one object with object_id={object_id}, "
            f"found {len(matches)}."
        )

    return matches[0]


def parse_layers(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip().lower()

    if spec == "all":
        return list(range(n_layers))

    layers = []

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            pieces = [int(x) for x in part.split(":") if x != ""]
            if len(pieces) == 2:
                start, stop = pieces
                step = 1
            elif len(pieces) == 3:
                start, stop, step = pieces
            else:
                raise ValueError(
                    f"Invalid layer range '{part}'. Use start:stop[:step]."
                )
            layers.extend(range(start, stop, step))
        else:
            layers.append(int(part))

    layers = list(dict.fromkeys(layers))

    invalid = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(
            f"Layer indices out of range for {n_layers} layers: {invalid}"
        )

    if not layers:
        raise ValueError("No layers selected.")

    return layers


def validate_pair_alignment(clean_inputs, cf_inputs):
    clean_ids = clean_inputs["input_ids"]
    cf_ids = cf_inputs["input_ids"]

    if clean_ids.shape != cf_ids.shape or not torch.equal(clean_ids, cf_ids):
        raise ValueError(
            "Clean and counterfactual input_ids are not identical. The current "
            "patching design requires the same question, same image dimensions, "
            "and exact sequence alignment between clean and counterfactual runs."
        )

    clean_grid = clean_inputs.get("image_grid_thw")
    cf_grid = cf_inputs.get("image_grid_thw")

    if clean_grid is None or cf_grid is None:
        raise ValueError("Missing image_grid_thw in processor output.")

    if clean_grid.shape != cf_grid.shape or not torch.equal(clean_grid, cf_grid):
        raise ValueError(
            "Clean and counterfactual image_grid_thw differ; visual-token "
            "positions cannot be patched one-to-one."
        )




def model_forward(model, inputs):
    kwargs = dict(inputs)
    kwargs["use_cache"] = False

    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1

    return model(**kwargs)


def next_token_logits(outputs, prompt_index: int):
    if outputs.logits.shape[1] == 1:
        return outputs.logits[0, -1].detach()

    return outputs.logits[0, prompt_index].detach()


def output_hidden_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]

    raise TypeError(
        f"Unsupported decoder-layer output type: {type(output).__name__}"
    )


def replace_output_hidden(output, patched_hidden):
    if torch.is_tensor(output):
        return patched_hidden

    if isinstance(output, tuple):
        return (patched_hidden,) + output[1:]

    if isinstance(output, list):
        return [patched_hidden] + list(output[1:])

    raise TypeError(
        f"Unsupported decoder-layer output type: {type(output).__name__}"
    )


@torch.inference_mode()
def forward_logits_and_lm_inputs(
    model,
    inputs,
    prompt_index: int,
):
    language_model, _ = get_language_model(model)
    captured = {}

    def capture_pre_hook(module, args, kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")

        if inputs_embeds is None:
            raise RuntimeError(
                "Language-model pre-hook did not receive inputs_embeds."
            )

        captured["inputs_embeds"] = inputs_embeds.detach().clone()
        return args, kwargs

    handle = language_model.register_forward_pre_hook(
        capture_pre_hook,
        with_kwargs=True,
    )

    try:
        outputs = model_forward(model, inputs)
    finally:
        handle.remove()

    if "inputs_embeds" not in captured:
        raise RuntimeError("Failed to capture language-model input embeddings.")

    return (
        next_token_logits(outputs, prompt_index),
        captured["inputs_embeds"],
    )


@torch.inference_mode()
def forward_with_lm_input_patch(
    model,
    clean_inputs,
    cf_lm_inputs: torch.Tensor,
    positions: torch.Tensor,
    prompt_index: int,
):
    language_model, _ = get_language_model(model)
    positions = positions.to(cf_lm_inputs.device)

    def patch_pre_hook(module, args, kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")

        if inputs_embeds is None:
            raise RuntimeError(
                "Language-model pre-hook did not receive inputs_embeds."
            )

        patched = inputs_embeds.clone()
        src = cf_lm_inputs.to(
            device=patched.device,
            dtype=patched.dtype,
        )
        patch_positions = positions.to(patched.device)
        patched[:, patch_positions, :] = src[:, patch_positions, :]

        new_kwargs = dict(kwargs)
        new_kwargs["inputs_embeds"] = patched
        return args, new_kwargs

    handle = language_model.register_forward_pre_hook(
        patch_pre_hook,
        with_kwargs=True,
    )

    try:
        outputs = model_forward(model, clean_inputs)
    finally:
        handle.remove()

    return next_token_logits(outputs, prompt_index)


@torch.inference_mode()
def capture_layer_output(
    model,
    inputs,
    layer_module,
):
    captured = {}

    def capture_hook(module, args, output):
        captured["hidden"] = output_hidden_tensor(output).detach().clone()

    handle = layer_module.register_forward_hook(capture_hook)

    try:
        model_forward(model, inputs)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture decoder-layer output.")

    return captured["hidden"]


@torch.inference_mode()
def forward_with_layer_patch(
    model,
    clean_inputs,
    layer_module,
    cf_hidden: torch.Tensor,
    positions: torch.Tensor,
    prompt_index: int,
):
    positions = positions.to(cf_hidden.device)

    def patch_hook(module, args, output):
        hidden = output_hidden_tensor(output)
        patched = hidden.clone()
        src = cf_hidden.to(
            device=patched.device,
            dtype=patched.dtype,
        )
        patch_positions = positions.to(patched.device)
        patched[:, patch_positions, :] = src[:, patch_positions, :]
        return replace_output_hidden(output, patched)

    handle = layer_module.register_forward_hook(patch_hook)

    try:
        outputs = model_forward(model, clean_inputs)
    finally:
        handle.remove()

    return next_token_logits(outputs, prompt_index)


def gap_from_scores(scores, clean_answer: str, cf_answer: str) -> float:
    return (
        scores["log_mass"][cf_answer]
        - scores["log_mass"][clean_answer]
    )


def recovery_value(
    patched_gap: float,
    clean_gap: float,
    cf_gap: float,
) -> float:
    denominator = cf_gap - clean_gap

    if abs(denominator) < 1e-8:
        return float("nan")

    return (patched_gap - clean_gap) / denominator


def base_result_fields(
    record,
    cf_type: str,
    cf_record: dict,
    clean_scores,
    cf_scores,
    clean_gap: float,
    cf_gap: float,
    pair_csv_row: Optional[dict],
    layout: dict,
    clean_target: dict,
    cf_target: dict,
    clean_control: Optional[dict],
    cf_control: Optional[dict],
    image_size: Tuple[int, int],
    answer_token_map: Dict[str, List[int]],
    control_object_id: int,
):
    clean_answer = record["answer"]
    cf_answer = cf_record["answer"]
    clean_counts = record.get("queried_value_counts", {})
    cf_counts = cf_record.get("queried_value_counts", {})

    fields = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "sample_id": record["sample_id"],
        "family_id": record.get("family_id"),
        "condition": record["condition"],
        "difficulty_level": record.get("difficulty_level"),
        "queried_attribute": record["queried_attribute"],
        "identifier_attributes": stable_json(record.get("identifier_attributes", [])),
        "ambiguous_identifier": record.get("ambiguous_identifier"),
        "ambiguous_identifiers": stable_json(record.get("ambiguous_identifiers", [])),
        "identifier_match_counts": stable_json(record.get("identifier_match_counts", {})),
        "queried_value_frequency_role": record.get("queried_value_frequency_role"),
        "clean_queried_value_counts": stable_json(clean_counts),
        "cf_queried_value_counts": stable_json(cf_counts),
        "clean_answer_frequency": clean_counts.get(clean_answer),
        "cf_answer_frequency": cf_counts.get(cf_answer),
        "cf_type": cf_type,
        "cf_changed_attribute": cf_record.get("changed_attribute"),
        "cf_changed_object_id": cf_record.get("changed_object_id"),
        "clean_image": record["image"],
        "cf_image": cf_record["image"],
        "clean_question": record["question"],
        "cf_question": cf_record.get("question", record["question"]),
        "clean_answer": clean_answer,
        "cf_answer": cf_answer,
        "answer_options": stable_json(record.get("answer_options", [])),
        "answer_token_ids": stable_json(answer_token_map),
        "target_object_id": int(record["target_object_id"]),
        "target_clean_attributes": stable_json(object_attributes(clean_target)),
        "target_cf_attributes": stable_json(object_attributes(cf_target)),
        "target_clean_bbox": stable_json(clean_target.get("bbox")),
        "target_cf_bbox": stable_json(cf_target.get("bbox")),
        "target_clean_center": stable_json(clean_target.get("center")),
        "target_cf_center": stable_json(cf_target.get("center")),
        "control_object_id": int(control_object_id),
        "control_clean_attributes": (
            stable_json(object_attributes(clean_control))
            if clean_control is not None
            else None
        ),
        "control_cf_attributes": (
            stable_json(object_attributes(cf_control))
            if cf_control is not None
            else None
        ),
        "control_clean_bbox": (
            stable_json(clean_control.get("bbox"))
            if clean_control is not None
            else None
        ),
        "control_cf_bbox": (
            stable_json(cf_control.get("bbox"))
            if cf_control is not None
            else None
        ),
        "control_clean_center": (
            stable_json(clean_control.get("center"))
            if clean_control is not None
            else None
        ),
        "control_cf_center": (
            stable_json(cf_control.get("center"))
            if cf_control is not None
            else None
        ),
        "clean_candidate_prediction": clean_scores["prediction"],
        "cf_candidate_prediction": cf_scores["prediction"],
        "clean_gap": clean_gap,
        "cf_gap": cf_gap,
        "clean_vocab_p_clean_answer": clean_scores["vocab_prob"][clean_answer],
        "clean_vocab_p_cf_answer": clean_scores["vocab_prob"][cf_answer],
        "cf_vocab_p_clean_answer": cf_scores["vocab_prob"][clean_answer],
        "cf_vocab_p_cf_answer": cf_scores["vocab_prob"][cf_answer],
        "clean_candidate_p_clean_answer": clean_scores["candidate_prob"][clean_answer],
        "clean_candidate_p_cf_answer": clean_scores["candidate_prob"][cf_answer],
        "cf_candidate_p_clean_answer": cf_scores["candidate_prob"][clean_answer],
        "cf_candidate_p_cf_answer": cf_scores["candidate_prob"][cf_answer],
        "image_width_px": int(image_size[0]),
        "image_height_px": int(image_size[1]),
        "image_grid_t": layout["grid_t"],
        "image_grid_h": layout["grid_h"],
        "image_grid_w": layout["grid_w"],
        "spatial_merge_size": layout["spatial_merge_size"],
        "llm_image_grid_t": layout["llm_t"],
        "llm_image_grid_h": layout["llm_h"],
        "llm_image_grid_w": layout["llm_w"],
        "n_image_tokens": int(layout["image_positions"].numel()),
    }

    fields.update(score_distribution_fields("clean", clean_scores))
    fields.update(score_distribution_fields("cf", cf_scores))

    if pair_csv_row is not None:
        fields.update(
            {
                "clean_generated_prediction_text": none_if_nan(
                    pair_csv_row.get("clean_prediction_text")
                ),
                "cf_generated_prediction_text": none_if_nan(
                    pair_csv_row.get("cf_prediction_text")
                ),
                "clean_generated_prediction": none_if_nan(
                    pair_csv_row.get("clean_prediction")
                ),
                "cf_generated_prediction": none_if_nan(
                    pair_csv_row.get("cf_prediction")
                ),
                "clean_generated_correct": bool_value(
                    pair_csv_row.get("clean_correct")
                ),
                "cf_generated_correct": bool_value(
                    pair_csv_row.get("cf_correct")
                ),
                "answer_changed": bool_value(pair_csv_row.get("answer_changed")),
                "prediction_changed": bool_value(
                    pair_csv_row.get("prediction_changed")
                ),
                "usable_for_patching": bool_value(
                    pair_csv_row.get("usable_for_patching")
                ),
            }
        )
    else:
        fields.update(
            {
                "clean_generated_prediction_text": None,
                "cf_generated_prediction_text": None,
                "clean_generated_prediction": None,
                "cf_generated_prediction": None,
                "clean_generated_correct": None,
                "cf_generated_correct": None,
                "answer_changed": clean_answer != cf_answer,
                "prediction_changed": (
                    clean_scores["prediction"] != cf_scores["prediction"]
                ),
                "usable_for_patching": (
                    clean_scores["prediction"] == clean_answer
                    and cf_scores["prediction"] == cf_answer
                    and clean_answer != cf_answer
                ),
            }
        )

    return fields


def make_patch_row(
    base_fields: dict,
    scope: str,
    layer: int,
    patch_type: str,
    dilation: int,
    positions: torch.Tensor,
    patch_bbox: Optional[Sequence[float]],
    patch_grid_region: Optional[dict],
    patched_scores,
    target_reference_n_patch_tokens: Optional[int] = None,
    matched_control_info: Optional[dict] = None,
):
    clean_answer = base_fields["clean_answer"]
    cf_answer = base_fields["cf_answer"]
    patched_gap = gap_from_scores(
        patched_scores,
        clean_answer,
        cf_answer,
    )

    patched_candidate_p_clean = patched_scores["candidate_prob"][clean_answer]
    patched_candidate_p_cf = patched_scores["candidate_prob"][cf_answer]
    patched_vocab_p_clean = patched_scores["vocab_prob"][clean_answer]
    patched_vocab_p_cf = patched_scores["vocab_prob"][cf_answer]

    row = dict(base_fields)
    row.update(
        {
            "scope": scope,
            "layer": layer,
            "patch_type": patch_type,
            "dilation": dilation,
            "n_patch_tokens": int(positions.numel()),
            "target_reference_n_patch_tokens": (
                int(target_reference_n_patch_tokens)
                if target_reference_n_patch_tokens is not None
                else None
            ),
            "patch_token_count_difference_from_target": (
                int(positions.numel()) - int(target_reference_n_patch_tokens)
                if target_reference_n_patch_tokens is not None
                else None
            ),
            "patch_token_count_ratio_to_target": (
                float(positions.numel()) / float(target_reference_n_patch_tokens)
                if target_reference_n_patch_tokens not in (None, 0)
                else None
            ),
            "patch_token_count_exact_match": (
                bool(int(positions.numel()) == int(target_reference_n_patch_tokens))
                if target_reference_n_patch_tokens is not None
                else None
            ),
            "matched_control_object_id": (
                int(matched_control_info["object_id"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_strategy": (
                str(matched_control_info["strategy"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_requested_shape": (
                stable_json(matched_control_info["requested_shape"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_actual_shape": (
                stable_json(matched_control_info["actual_shape"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_exact_shape": (
                bool(matched_control_info["exact_shape"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_exact_token_count": (
                bool(matched_control_info["exact_token_count"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_target_overlap_tokens": (
                int(matched_control_info["target_overlap_tokens"])
                if matched_control_info is not None
                else None
            ),
            "matched_control_anchor_overlap_tokens": (
                int(matched_control_info["anchor_overlap_tokens"])
                if matched_control_info is not None
                else None
            ),
            "patch_bbox": (
                stable_json([float(v) for v in patch_bbox])
                if patch_bbox is not None
                else None
            ),
            "patched_candidate_prediction": patched_scores["prediction"],
            "patched_gap": patched_gap,
            "delta_gap_from_clean": patched_gap - base_fields["clean_gap"],
            "recovery": recovery_value(
                patched_gap,
                base_fields["clean_gap"],
                base_fields["cf_gap"],
            ),
            "patched_vocab_p_clean_answer": patched_vocab_p_clean,
            "patched_vocab_p_cf_answer": patched_vocab_p_cf,
            "patched_candidate_p_clean_answer": patched_candidate_p_clean,
            "patched_candidate_p_cf_answer": patched_candidate_p_cf,
            "delta_vocab_p_cf_answer_from_clean": (
                patched_vocab_p_cf - base_fields["clean_vocab_p_cf_answer"]
            ),
            "delta_candidate_p_cf_answer_from_clean": (
                patched_candidate_p_cf
                - base_fields["clean_candidate_p_cf_answer"]
            ),
            "patched_matches_clean_answer": (
                patched_scores["prediction"] == clean_answer
            ),
            "patched_matches_cf_answer": (
                patched_scores["prediction"] == cf_answer
            ),
            "patched_prediction_changed_from_clean": (
                patched_scores["prediction"]
                != base_fields["clean_candidate_prediction"]
            ),
        }
    )
    row.update(patch_grid_fields(patch_grid_region))
    row.update(score_distribution_fields("patched", patched_scores))

    return row


def append_rows(path: Path, rows: List[dict]):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists() and path.stat().st_size > 0
    pd.DataFrame(rows).to_csv(
        path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
    )


def requested_keys_for_pair(
    sample_id: str,
    cf_type: str,
    selected_layers: Sequence[int],
    patch_types: Sequence[str],
    dilations: Sequence[int],
    skip_input_patches: bool,
):
    keys = set()

    if not skip_input_patches:
        if "target" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "lm_input", -1, "target", dilation))

        if "all_image" in patch_types:
            keys.add((sample_id, cf_type, "lm_input", -1, "all_image", -1))

        if "distractor" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "lm_input", -1, "distractor", dilation))

        if "matched_distractor" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "lm_input", -1, "matched_distractor", dilation))

    for layer_idx in selected_layers:
        if "target" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "resid_post", layer_idx, "target", dilation))

        if "all_image" in patch_types:
            keys.add((sample_id, cf_type, "resid_post", layer_idx, "all_image", -1))

        if "distractor" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "resid_post", layer_idx, "distractor", dilation))

        if "matched_distractor" in patch_types:
            for dilation in dilations:
                keys.add((sample_id, cf_type, "resid_post", layer_idx, "matched_distractor", dilation))

        if "last_token" in patch_types:
            keys.add((sample_id, cf_type, "resid_post", layer_idx, "last_token", -1))

    return keys


def completed_keys(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return set()

    df = pd.read_csv(
        path,
        usecols=[
            "sample_id",
            "cf_type",
            "scope",
            "layer",
            "patch_type",
            "dilation",
        ],
    )

    return {
        (
            str(row.sample_id),
            str(row.cf_type),
            str(row.scope),
            int(row.layer),
            str(row.patch_type),
            int(row.dilation),
        )
        for row in df.itertuples(index=False)
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Causal activation patching for paired clean/counterfactual "
            "Qwen2.5-VL binding examples."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=None,
        help=(
            "CSV produced by run_qwen_counterfactuals.py. When supplied, only "
            "usable_for_patching pairs are used unless --include_unusable is set."
        ),
    )
    parser.add_argument(
        "--include_unusable",
        action="store_true",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Local checkpoint path or Hugging Face model ID. Defaults to "
            "$QWEN_MODEL_PATH when set, otherwise "
            "/home/mmd/models/Qwen2.5-VL-7B-Instruct."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help=(
            "Decoder layers to patch: 'all', comma-separated indices such as "
            "'0,4,8,12', or ranges such as '0:28:4'."
        ),
    )
    parser.add_argument(
        "--patch_types",
        nargs="+",
        choices=PATCH_TYPES,
        default=["target", "all_image", "last_token"],
        help=(
            "Patch controls to run. 'distractor' is the raw unchanged-distractor control; 'matched_distractor' first tries a target-disjoint region with the same shape/token count near that distractor, then uses recorded geometric fallbacks if needed. "
            "Both controls can be added explicitly."
        ),
    )
    parser.add_argument(
        "--dilations",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "Visual-grid dilation values for target/distractor masks. Use "
            "'--dilations 0 1 2' for the distributed-region sweep."
        ),
    )
    parser.add_argument(
        "--control_object_id",
        type=int,
        default=3,
        help="Distractor object used for the optional negative-control patch.",
    )
    parser.add_argument(
        "--skip_input_patches",
        action="store_true",
        help="Skip patches at the LM input (post-vision-projector image embeddings).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of clean records considered before pair filtering.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Stop after this many usable clean/counterfactual pairs have been processed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run using rows already present in --out.",
    )

    args = parser.parse_args()

    if any(dilation < 0 for dilation in args.dilations):
        raise ValueError("All dilation values must be non-negative.")

    if args.out.exists() and not args.resume:
        args.out.unlink()

    if args.resume:
        validate_resume_output_schema(args.out)

    done = completed_keys(args.out) if args.resume else set()
    pair_table = load_pair_table(args.pairs_csv)

    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]

    model = None
    processor = None

    try:
        print(f"Loading model: {args.model_id}")
        model, processor = load_qwen_model_and_processor(
            model_id=args.model_id,
            dtype=args.dtype,
            allow_download=args.allow_download,
        )

        layers, language_model_path = get_decoder_layers(model)
        selected_layers = parse_layers(args.layers, len(layers))

        print(f"Language model path: {language_model_path}.layers")
        print(f"Number of decoder layers: {len(layers)}")
        print(f"Selected layers: {selected_layers}")
        print(f"Patch types: {args.patch_types}")
        print(f"Dilations: {args.dilations}")

        meta_path = write_run_metadata(
            args=args,
            model=model,
            processor=processor,
            language_model_path=language_model_path,
            n_layers=len(layers),
            selected_layers=selected_layers,
        )
        print(f"Run metadata: {meta_path}")

        if pair_table is None:
            print(
                "WARNING: --pairs_csv was not supplied. Usability will be determined "
                "from candidate next-token scores rather than generated clean/CF "
                "answers. For final experiments, use run_qwen_counterfactuals.py and "
                "pass its CSV here."
            )

        processed_pairs = 0
        skipped_unusable = 0

        stop_requested = False

        for record in tqdm(records, desc="Activation patching"):
            if stop_requested:
                break

            for cf_type, cf_record in iter_counterfactuals(record):
                if args.max_pairs is not None and processed_pairs >= args.max_pairs:
                    stop_requested = True
                    break
                pair_csv_row = None

                if pair_table is not None:
                    pair_csv_row = pair_table.get(
                        (record["sample_id"], cf_type)
                    )

                    if pair_csv_row is None:
                        continue

                    if (
                        not args.include_unusable
                        and not bool_value(pair_csv_row["usable_for_patching"])
                    ):
                        skipped_unusable += 1
                        continue

                pair_requested_keys = requested_keys_for_pair(
                    record["sample_id"],
                    cf_type,
                    selected_layers,
                    args.patch_types,
                    args.dilations,
                    args.skip_input_patches,
                )

                if args.resume and pair_requested_keys and pair_requested_keys.issubset(done):
                    continue

                clean_image_path = args.data / record["image"]
                cf_image_path = args.data / cf_record["image"]

                clean_inputs = prepare_inputs(
                    model,
                    processor,
                    clean_image_path,
                    record["question"],
                )
                cf_inputs = prepare_inputs(
                    model,
                    processor,
                    cf_image_path,
                    cf_record.get("question", record["question"]),
                )

                validate_pair_alignment(clean_inputs, cf_inputs)

                prompt_index = last_attended_token_index(clean_inputs)
                layout = image_token_layout(model, clean_inputs)

                with Image.open(clean_image_path) as clean_img:
                    clean_image_size = clean_img.size
                with Image.open(cf_image_path) as cf_img:
                    cf_image_size = cf_img.size

                if clean_image_size != cf_image_size:
                    raise ValueError(
                        "Clean and counterfactual image sizes differ."
                    )

                target_id = int(record["target_object_id"])
                clean_target = find_object(record["objects"], target_id)
                cf_target = find_object(cf_record["objects"], target_id)
                target_bbox = union_bbox(
                    clean_target["bbox"],
                    cf_target["bbox"],
                )

                clean_control = None
                cf_control = None
                control_bbox = None

                if args.control_object_id == target_id:
                    if any(x in args.patch_types for x in ("distractor", "matched_distractor")):
                        raise ValueError(
                            "control_object_id must refer to a distractor, not the target."
                        )
                else:
                    try:
                        clean_control = find_object(
                            record["objects"],
                            args.control_object_id,
                        )
                        cf_control = find_object(
                            cf_record["objects"],
                            args.control_object_id,
                        )
                        control_bbox = union_bbox(
                            clean_control["bbox"],
                            cf_control["bbox"],
                        )
                    except ValueError:
                        if any(x in args.patch_types for x in ("distractor", "matched_distractor")):
                            raise
                        clean_control = None
                        cf_control = None
                        control_bbox = None

                target_regions_by_dilation = {
                    dilation: bbox_to_image_token_region(
                        layout,
                        target_bbox,
                        clean_image_size,
                        dilation=dilation,
                    )
                    for dilation in args.dilations
                }
                target_positions_by_dilation = {
                    dilation: bbox_to_image_token_positions(
                        layout,
                        target_bbox,
                        clean_image_size,
                        dilation=dilation,
                    )
                    for dilation in args.dilations
                }

                control_regions_by_dilation = {}
                control_positions_by_dilation = {}
                matched_control_regions_by_dilation = {}
                matched_control_positions_by_dilation = {}
                if control_bbox is not None:
                    control_regions_by_dilation = {
                        dilation: bbox_to_image_token_region(
                            layout,
                            control_bbox,
                            clean_image_size,
                            dilation=dilation,
                        )
                        for dilation in args.dilations
                    }
                    control_positions_by_dilation = {
                        dilation: bbox_to_image_token_positions(
                            layout,
                            control_bbox,
                            clean_image_size,
                            dilation=dilation,
                        )
                        for dilation in args.dilations
                    }

                    # Robust matched-distractor control.  Start with the
                    # configured control object, but keep all other distractors
                    # available as fallbacks.  The selector first shifts an
                    # exact-shape window away from the target; only if that is
                    # impossible does it try another distractor, reshape while
                    # preserving token count, or finally reduce token count.
                    distractor_anchor_candidates = []
                    object_ids = [
                        int(obj["object_id"])
                        for obj in record["objects"]
                        if int(obj["object_id"]) != target_id
                    ]
                    ordered_object_ids = []
                    if args.control_object_id in object_ids:
                        ordered_object_ids.append(int(args.control_object_id))
                    ordered_object_ids.extend(
                        object_id
                        for object_id in object_ids
                        if object_id != int(args.control_object_id)
                    )

                    for object_id in ordered_object_ids:
                        clean_anchor_obj = find_object(record["objects"], object_id)
                        cf_anchor_obj = find_object(cf_record["objects"], object_id)
                        anchor_bbox = union_bbox(
                            clean_anchor_obj["bbox"],
                            cf_anchor_obj["bbox"],
                        )
                        anchor_region = bbox_to_image_token_region(
                            layout,
                            anchor_bbox,
                            clean_image_size,
                            dilation=0,
                        )
                        distractor_anchor_candidates.append(
                            (int(object_id), anchor_region)
                        )

                    matched_control_info_by_dilation = {}
                    matched_control_regions_by_dilation = {}
                    matched_control_positions_by_dilation = {}

                    if "matched_distractor" in args.patch_types:
                        for dilation in args.dilations:
                            matched_region, matched_info = (
                                robust_matched_region_around_anchors(
                                    layout=layout,
                                    reference_region=target_regions_by_dilation[dilation],
                                    anchors=distractor_anchor_candidates,
                                    forbidden_region=target_regions_by_dilation[dilation],
                                )
                            )
                            matched_positions = image_token_positions_from_region(
                                layout,
                                matched_region,
                            )

                            target_positions = target_positions_by_dilation[dilation]
                            target_position_set = set(
                                target_positions.detach().cpu().tolist()
                            )
                            matched_position_set = set(
                                matched_positions.detach().cpu().tolist()
                            )
                            overlap_n = len(
                                target_position_set & matched_position_set
                            )
                            if overlap_n != 0:
                                raise RuntimeError(
                                    "Internal matched-control bug: selected region "
                                    "still overlaps the target token region: "
                                    f"overlap={overlap_n}, dilation={dilation}."
                                )

                            # Keep metadata derived from the actual tensor mask as
                            # the final source of truth.
                            matched_info = dict(matched_info)
                            matched_info["actual_n_tokens"] = int(
                                matched_positions.numel()
                            )
                            matched_info["requested_n_tokens"] = int(
                                target_positions.numel()
                            )
                            matched_info["exact_token_count"] = bool(
                                matched_positions.numel() == target_positions.numel()
                            )
                            matched_info["target_overlap_tokens"] = int(overlap_n)

                            if matched_info["strategy"] != "exact_shape_preferred_object":
                                print(
                                    "Matched-control fallback: "
                                    f"sample={record['sample_id']}, "
                                    f"dilation={dilation}, "
                                    f"strategy={matched_info['strategy']}, "
                                    f"object_id={matched_info['object_id']}, "
                                    f"tokens={matched_info['actual_n_tokens']}/"
                                    f"{matched_info['requested_n_tokens']}"
                                )

                            matched_control_regions_by_dilation[dilation] = matched_region
                            matched_control_positions_by_dilation[dilation] = matched_positions
                            matched_control_info_by_dilation[dilation] = matched_info

                all_image_positions = layout["image_positions"]
                all_image_grid_region = full_image_region(layout)
                last_token_positions = torch.tensor(
                    [prompt_index],
                    device=all_image_positions.device,
                    dtype=torch.long,
                )

                answer_options = record["answer_options"]
                clean_answer = record["answer"]
                cf_answer = cf_record["answer"]

                if clean_answer == cf_answer:
                    raise ValueError(
                        f"Counterfactual answer did not change for {record['sample_id']}."
                    )

                answer_token_map = build_answer_token_map(
                    processor.tokenizer,
                    answer_options,
                )

                clean_logits, clean_lm_inputs = forward_logits_and_lm_inputs(
                    model,
                    clean_inputs,
                    prompt_index,
                )
                cf_logits, cf_lm_inputs = forward_logits_and_lm_inputs(
                    model,
                    cf_inputs,
                    prompt_index,
                )

                clean_scores = score_answer_options(
                    clean_logits,
                    answer_token_map,
                )
                cf_scores = score_answer_options(
                    cf_logits,
                    answer_token_map,
                )

                clean_gap = gap_from_scores(
                    clean_scores,
                    clean_answer,
                    cf_answer,
                )
                cf_gap = gap_from_scores(
                    cf_scores,
                    clean_answer,
                    cf_answer,
                )

                if pair_table is None and not args.include_unusable:
                    candidate_usable = (
                        clean_scores["prediction"] == clean_answer
                        and cf_scores["prediction"] == cf_answer
                        and clean_answer != cf_answer
                    )

                    if not candidate_usable:
                        skipped_unusable += 1
                        del clean_inputs, cf_inputs, clean_lm_inputs, cf_lm_inputs
                        del clean_logits, cf_logits
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                base_fields = base_result_fields(
                    record=record,
                    cf_type=cf_type,
                    cf_record=cf_record,
                    clean_scores=clean_scores,
                    cf_scores=cf_scores,
                    clean_gap=clean_gap,
                    cf_gap=cf_gap,
                    pair_csv_row=pair_csv_row,
                    layout=layout,
                    clean_target=clean_target,
                    cf_target=cf_target,
                    clean_control=clean_control,
                    cf_control=cf_control,
                    image_size=clean_image_size,
                    answer_token_map=answer_token_map,
                    control_object_id=args.control_object_id,
                )

                sample_rows = []

                if not args.skip_input_patches:
                    if "target" in args.patch_types:
                        for dilation, positions in target_positions_by_dilation.items():
                            key = (
                                record["sample_id"],
                                cf_type,
                                "lm_input",
                                -1,
                                "target",
                                dilation,
                            )
                            if key in done:
                                continue

                            patched_logits = forward_with_lm_input_patch(
                                model,
                                clean_inputs,
                                cf_lm_inputs,
                                positions,
                                prompt_index,
                            )
                            patched_scores = score_answer_options(
                                patched_logits,
                                answer_token_map,
                            )
                            sample_rows.append(
                                make_patch_row(
                                    base_fields,
                                    scope="lm_input",
                                    layer=-1,
                                    patch_type="target",
                                    dilation=dilation,
                                    positions=positions,
                                    patch_bbox=target_bbox,
                                    patch_grid_region=target_regions_by_dilation[dilation],
                                    patched_scores=patched_scores,
                                    target_reference_n_patch_tokens=int(positions.numel()),
                                )
                            )

                    if "all_image" in args.patch_types:
                        key = (
                            record["sample_id"],
                            cf_type,
                            "lm_input",
                            -1,
                            "all_image",
                            -1,
                        )
                        if key not in done:
                            patched_logits = forward_with_lm_input_patch(
                                model,
                                clean_inputs,
                                cf_lm_inputs,
                                all_image_positions,
                                prompt_index,
                            )
                            patched_scores = score_answer_options(
                                patched_logits,
                                answer_token_map,
                            )
                            sample_rows.append(
                                make_patch_row(
                                    base_fields,
                                    scope="lm_input",
                                    layer=-1,
                                    patch_type="all_image",
                                    dilation=-1,
                                    positions=all_image_positions,
                                    patch_bbox=None,
                                    patch_grid_region=all_image_grid_region,
                                    patched_scores=patched_scores,
                                )
                            )

                    if "distractor" in args.patch_types:
                        for dilation, positions in control_positions_by_dilation.items():
                            key = (
                                record["sample_id"],
                                cf_type,
                                "lm_input",
                                -1,
                                "distractor",
                                dilation,
                            )
                            if key in done:
                                continue

                            patched_logits = forward_with_lm_input_patch(
                                model,
                                clean_inputs,
                                cf_lm_inputs,
                                positions,
                                prompt_index,
                            )
                            patched_scores = score_answer_options(
                                patched_logits,
                                answer_token_map,
                            )
                            sample_rows.append(
                                make_patch_row(
                                    base_fields,
                                    scope="lm_input",
                                    layer=-1,
                                    patch_type="distractor",
                                    dilation=dilation,
                                    positions=positions,
                                    patch_bbox=control_bbox,
                                    patch_grid_region=control_regions_by_dilation[dilation],
                                    patched_scores=patched_scores,
                                    target_reference_n_patch_tokens=int(
                                        target_positions_by_dilation[dilation].numel()
                                    ),
                                )
                            )

                    if "matched_distractor" in args.patch_types:
                        for dilation, positions in matched_control_positions_by_dilation.items():
                            key = (
                                record["sample_id"],
                                cf_type,
                                "lm_input",
                                -1,
                                "matched_distractor",
                                dilation,
                            )
                            if key in done:
                                continue

                            patched_logits = forward_with_lm_input_patch(
                                model,
                                clean_inputs,
                                cf_lm_inputs,
                                positions,
                                prompt_index,
                            )
                            patched_scores = score_answer_options(
                                patched_logits,
                                answer_token_map,
                            )
                            sample_rows.append(
                                make_patch_row(
                                    base_fields,
                                    scope="lm_input",
                                    layer=-1,
                                    patch_type="matched_distractor",
                                    dilation=dilation,
                                    positions=positions,
                                    patch_bbox=None,
                                    patch_grid_region=matched_control_regions_by_dilation[dilation],
                                    patched_scores=patched_scores,
                                    target_reference_n_patch_tokens=int(
                                        target_positions_by_dilation[dilation].numel()
                                    ),
                                    matched_control_info=matched_control_info_by_dilation[dilation],
                                )
                            )

                    append_rows(args.out, sample_rows)
                    for row in sample_rows:
                        done.add(
                            (
                                str(row["sample_id"]),
                                str(row["cf_type"]),
                                str(row["scope"]),
                                int(row["layer"]),
                                str(row["patch_type"]),
                                int(row["dilation"]),
                            )
                        )
                    sample_rows = []

                del clean_lm_inputs

                for layer_idx in selected_layers:
                    layer_module = layers[layer_idx]

                    pending = []

                    if "target" in args.patch_types:
                        for dilation in args.dilations:
                            key = (
                                record["sample_id"],
                                cf_type,
                                "resid_post",
                                layer_idx,
                                "target",
                                dilation,
                            )
                            if key not in done:
                                pending.append(("target", dilation))

                    if "all_image" in args.patch_types:
                        key = (
                            record["sample_id"],
                            cf_type,
                            "resid_post",
                            layer_idx,
                            "all_image",
                            -1,
                        )
                        if key not in done:
                            pending.append(("all_image", -1))

                    if "distractor" in args.patch_types:
                        for dilation in args.dilations:
                            key = (
                                record["sample_id"],
                                cf_type,
                                "resid_post",
                                layer_idx,
                                "distractor",
                                dilation,
                            )
                            if key not in done:
                                pending.append(("distractor", dilation))

                    if "matched_distractor" in args.patch_types:
                        for dilation in args.dilations:
                            key = (
                                record["sample_id"],
                                cf_type,
                                "resid_post",
                                layer_idx,
                                "matched_distractor",
                                dilation,
                            )
                            if key not in done:
                                pending.append(("matched_distractor", dilation))

                    if "last_token" in args.patch_types:
                        key = (
                            record["sample_id"],
                            cf_type,
                            "resid_post",
                            layer_idx,
                            "last_token",
                            -1,
                        )
                        if key not in done:
                            pending.append(("last_token", -1))

                    if not pending:
                        continue

                    cf_hidden = capture_layer_output(
                        model,
                        cf_inputs,
                        layer_module,
                    )

                    for patch_type, dilation in pending:
                        target_reference_n_patch_tokens = None
                        matched_control_info = None
                        if patch_type == "target":
                            positions = target_positions_by_dilation[dilation]
                            patch_bbox = target_bbox
                            patch_grid_region = target_regions_by_dilation[dilation]
                            target_reference_n_patch_tokens = int(positions.numel())
                        elif patch_type == "all_image":
                            positions = all_image_positions
                            patch_bbox = None
                            patch_grid_region = all_image_grid_region
                        elif patch_type == "distractor":
                            positions = control_positions_by_dilation[dilation]
                            patch_bbox = control_bbox
                            patch_grid_region = control_regions_by_dilation[dilation]
                            target_reference_n_patch_tokens = int(
                                target_positions_by_dilation[dilation].numel()
                            )
                        elif patch_type == "matched_distractor":
                            positions = matched_control_positions_by_dilation[dilation]
                            patch_bbox = None
                            patch_grid_region = matched_control_regions_by_dilation[dilation]
                            target_reference_n_patch_tokens = int(
                                target_positions_by_dilation[dilation].numel()
                            )
                            matched_control_info = matched_control_info_by_dilation[dilation]
                        elif patch_type == "last_token":
                            positions = last_token_positions
                            patch_bbox = None
                            patch_grid_region = None
                        else:
                            raise RuntimeError(f"Unexpected patch type: {patch_type}")

                        patched_logits = forward_with_layer_patch(
                            model,
                            clean_inputs,
                            layer_module,
                            cf_hidden,
                            positions,
                            prompt_index,
                        )
                        patched_scores = score_answer_options(
                            patched_logits,
                            answer_token_map,
                        )

                        sample_rows.append(
                            make_patch_row(
                                base_fields,
                                scope="resid_post",
                                layer=layer_idx,
                                patch_type=patch_type,
                                dilation=dilation,
                                positions=positions,
                                patch_bbox=patch_bbox,
                                patch_grid_region=patch_grid_region,
                                patched_scores=patched_scores,
                                target_reference_n_patch_tokens=target_reference_n_patch_tokens,
                                matched_control_info=matched_control_info,
                            )
                        )

                    append_rows(args.out, sample_rows)
                    for row in sample_rows:
                        done.add(
                            (
                                str(row["sample_id"]),
                                str(row["cf_type"]),
                                str(row["scope"]),
                                int(row["layer"]),
                                str(row["patch_type"]),
                                int(row["dilation"]),
                            )
                        )
                    sample_rows = []

                    del cf_hidden
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                processed_pairs += 1

                del clean_inputs, cf_inputs, cf_lm_inputs
                del clean_logits, cf_logits
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Saved patching results to: {args.out}")
        print(f"Processed usable pairs: {processed_pairs}")
        print(f"Skipped unusable pairs: {skipped_unusable}")
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass



if __name__ == "__main__":
    main()
