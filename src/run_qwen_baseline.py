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
            desc="Running baseline evaluation",
        ):
            image_path = args.data / record["image"]

            prediction_text = predict_one(
                model,
                processor,
                image_path,
                record["question"],
            )

            prediction = normalize_answer(
                prediction_text,
                record.get("answer_options"),
            )

            counts = record.get(
                "queried_value_counts",
                {},
            )

            answer_frequency = counts.get(record["answer"])

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
                    "queried_value_counts": json.dumps(
                        counts,
                        sort_keys=True,
                    ),
                    "answer_frequency": answer_frequency,
                    "image": record["image"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "prediction_text": prediction_text,
                    "prediction": prediction,
                    "correct": prediction == record["answer"],
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
        print(f"Number of samples: {len(df)}")

        if len(df) == 0:
            return

        print("\nOverall accuracy:")
        print(df["correct"].mean())

        print("\nAccuracy by queried attribute:")
        print(
            df.groupby("queried_attribute")["correct"].agg(
                ["count", "mean"]
            )
        )

        print("\nAccuracy by queried attribute and answer frequency:")
        print(
            df.groupby(
                ["queried_attribute", "answer_frequency"]
            )["correct"].agg(["count", "mean"])
        )

        print("\nAccuracy by queried-value frequency role:")
        print(
            df.groupby("queried_value_frequency_role")["correct"].agg(
                ["count", "mean"]
            )
        )

        print("\nAccuracy by queried attribute and frequency role:")
        print(
            df.groupby(
                ["queried_attribute", "queried_value_frequency_role"]
            )["correct"].agg(["count", "mean"])
        )

        if df["ambiguous_identifier"].isin(["shape", "color", "size"]).any():
            print("\nAccuracy by ambiguous identifier (single-cue condition):")

            single_rows = df[
                df["ambiguous_identifier"].isin(["shape", "color", "size"])
            ]

            print(
                single_rows.groupby("ambiguous_identifier")["correct"].agg(
                    ["count", "mean"]
                )
            )
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
