"""
Skeleton for later VLM baseline inference.

Replace dummy_predict with Qwen2.5-VL or LLaVA inference.
For the first implementation, keep this file simple: one image + one question -> one prediction.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from prompts import normalize_answer


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dummy_predict(question: str, image_path: Path) -> str:
    """Replace this with VLM inference."""
    raise NotImplementedError("Plug in Qwen2.5-VL or LLaVA inference here.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.data)
    rows = []

    for r in tqdm(records):
        image_path = args.data / r["image"]
        try:
            pred_text = dummy_predict(r["question"], image_path)
        except NotImplementedError:
            pred_text = ""

        pred = normalize_answer(pred_text)
        rows.append({
            "sample_id": r["sample_id"],
            "condition": r["condition"],
            "question": r["question"],
            "answer": r["answer"],
            "prediction_text": pred_text,
            "prediction": pred,
            "correct": pred == r["answer"],
            "image": r["image"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
