import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from prompts import normalize_answer
from qwen_utils import (
    DEFAULT_MODEL_ID,
    load_qwen_model_and_processor,
    predict_one,
)


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open(
        "r",
        encoding="utf-8",
    ) as f:
        return [json.loads(line) for line in f]


def iter_counterfactuals(record):
    cfs = record.get("counterfactuals", {})

    if isinstance(cfs, dict):
        items = cfs.items()
    elif isinstance(cfs, list):
        items = [
            (
                cf.get("type", f"cf_{i}"),
                cf,
            )
            for i, cf in enumerate(cfs)
        ]
    else:
        return

    for cf_type, cf in items:
        if cf is None:
            continue

        image = (
            cf.get("image")
            or cf.get("counterfactual_image")
            or cf.get("image_path")
        )

        answer = (
            cf.get("answer")
            or cf.get("counterfactual_answer")
            or cf.get("new_answer")
        )

        question = cf.get("question") or record["question"]

        if image is None or answer is None:
            continue

        yield (
            cf_type,
            image,
            question,
            answer,
            cf,
        )


def main():
    parser = argparse.ArgumentParser()

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
        "--model_id",
        type=str,
        default="/home/mmd/models/Qwen2.5-VL-7B-Instruct",
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
        help="Allow Transformers to access the network instead of local-files-only loading.",
    )

    parser.add_argument(
        "--cf_types",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

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

        rows = []

        for record in tqdm(
            records,
            desc="Running counterfactual evaluation",
        ):
            clean_image_path = args.data / record["image"]
            clean_question = record["question"]
            clean_answer = record["answer"]
            answer_options = record.get("answer_options")

            clean_text = predict_one(
                model,
                processor,
                clean_image_path,
                clean_question,
            )

            clean_pred = normalize_answer(
                clean_text,
                answer_options,
            )

            clean_correct = clean_pred == clean_answer

            for (
                cf_type,
                cf_image,
                cf_question,
                cf_answer,
                cf_record,
            ) in iter_counterfactuals(record):
                if args.cf_types is not None and cf_type not in args.cf_types:
                    continue

                cf_image_path = args.data / cf_image

                cf_text = predict_one(
                    model,
                    processor,
                    cf_image_path,
                    cf_question,
                )

                cf_pred = normalize_answer(
                    cf_text,
                    answer_options,
                )

                cf_correct = cf_pred == cf_answer

                clean_counts = record.get(
                    "queried_value_counts",
                    {},
                )

                cf_counts = cf_record.get(
                    "queried_value_counts",
                    {},
                )

                clean_answer_frequency = clean_counts.get(clean_answer)
                cf_answer_frequency = cf_counts.get(cf_answer)

                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "family_id": record.get("family_id"),
                        "condition": record["condition"],
                        "difficulty_level": record.get("difficulty_level"),
                        "queried_attribute": record.get("queried_attribute"),
                        "ambiguous_identifier": record.get("ambiguous_identifier"),
                        "ambiguous_identifiers": json.dumps(
                            record.get("ambiguous_identifiers", [])
                        ),
                        "queried_value_frequency_role": record.get(
                            "queried_value_frequency_role"
                        ),
                        "clean_queried_value_counts": json.dumps(
                            clean_counts,
                            sort_keys=True,
                        ),
                        "cf_queried_value_counts": json.dumps(
                            cf_counts,
                            sort_keys=True,
                        ),
                        "clean_answer_frequency": clean_answer_frequency,
                        "cf_answer_frequency": cf_answer_frequency,
                        "cf_type": cf_type,
                        "clean_image": record["image"],
                        "clean_question": clean_question,
                        "clean_answer": clean_answer,
                        "clean_prediction_text": clean_text,
                        "clean_prediction": clean_pred,
                        "clean_correct": clean_correct,
                        "cf_image": cf_image,
                        "cf_question": cf_question,
                        "cf_answer": cf_answer,
                        "cf_prediction_text": cf_text,
                        "cf_prediction": cf_pred,
                        "cf_correct": cf_correct,
                        "answer_changed": clean_answer != cf_answer,
                        "prediction_changed": clean_pred != cf_pred,
                        "usable_for_patching": (
                            clean_correct
                            and cf_correct
                            and clean_answer != cf_answer
                        ),
                    }
                )

        df = pd.DataFrame(rows)

        args.out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            args.out,
            index=False,
        )

        print(f"Saved: {args.out}")
        print(f"Number of pairs: {len(df)}")

        if len(df) == 0:
            print(
                "No counterfactual pairs found. Check the counterfactuals field "
                "in metadata.jsonl."
            )
            return

        print("\nSummary by counterfactual type:")

        summary = df.groupby("cf_type").agg(
            n=("sample_id", "count"),
            clean_acc=("clean_correct", "mean"),
            cf_acc=("cf_correct", "mean"),
            answer_changed_rate=("answer_changed", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
            usable_rate=("usable_for_patching", "mean"),
        )

        print(summary)

        print("\nSummary by queried attribute:")

        attr_summary = df.groupby("queried_attribute").agg(
            n=("sample_id", "count"),
            clean_acc=("clean_correct", "mean"),
            cf_acc=("cf_correct", "mean"),
            usable_rate=("usable_for_patching", "mean"),
        )

        print(attr_summary)

        print("\nSummary by clean answer frequency:")

        clean_frequency_summary = (
            df.groupby(
                ["queried_attribute", "clean_answer_frequency"]
            )
            .agg(
                n=("sample_id", "count"),
                clean_acc=("clean_correct", "mean"),
                cf_acc=("cf_correct", "mean"),
                usable_rate=("usable_for_patching", "mean"),
            )
        )

        print(clean_frequency_summary)

        print("\nSummary by counterfactual answer frequency:")

        cf_frequency_summary = (
            df.groupby(
                ["queried_attribute", "cf_answer_frequency"]
            )
            .agg(
                n=("sample_id", "count"),
                cf_acc=("cf_correct", "mean"),
                usable_rate=("usable_for_patching", "mean"),
            )
        )

        print(cf_frequency_summary)

        print("\nSummary by queried-value frequency role:")

        role_summary = df.groupby(
            ["queried_attribute", "queried_value_frequency_role"]
        ).agg(
            n=("sample_id", "count"),
            clean_acc=("clean_correct", "mean"),
            cf_acc=("cf_correct", "mean"),
            usable_rate=("usable_for_patching", "mean"),
        )

        print(role_summary)
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
