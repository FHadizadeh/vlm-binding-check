import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from prompts import normalize_answer


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(image_path: Path, question: str):
    image_uri = image_path.resolve().as_uri()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": question},
            ],
        }
    ]


@torch.inference_mode()
def predict_one(model, processor, image_path: Path, question: str, max_new_tokens: int = 8):
    messages = build_messages(image_path, question)

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
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def iter_counterfactuals(record):
    """
    Supports both:
      counterfactuals = {"size_swap": {...}, "color_swap": {...}}
    and:
      counterfactuals = [{"type": "size_swap", ...}, ...]
    """
    cfs = record.get("counterfactuals", {})

    if isinstance(cfs, dict):
        items = cfs.items()
    elif isinstance(cfs, list):
        items = [(cf.get("type", f"cf_{i}"), cf) for i, cf in enumerate(cfs)]
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

        yield cf_type, image, question, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cf_types", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Loading model: {args.model_id}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    rows = []

    for r in tqdm(records, desc="Running counterfactual evaluation"):
        clean_image_path = args.data / r["image"]
        clean_question = r["question"]
        clean_answer = r["answer"]

        clean_text = predict_one(model, processor, clean_image_path, clean_question)
        clean_pred = normalize_answer(clean_text)
        clean_correct = clean_pred == clean_answer

        for cf_type, cf_image, cf_question, cf_answer in iter_counterfactuals(r):
            if args.cf_types is not None and cf_type not in args.cf_types:
                continue

            cf_image_path = args.data / cf_image

            cf_text = predict_one(model, processor, cf_image_path, cf_question)
            cf_pred = normalize_answer(cf_text)
            cf_correct = cf_pred == cf_answer

            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "condition": r["condition"],
                    "cf_type": cf_type,

                    "clean_image": r["image"],
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
                    "usable_for_patching": clean_correct and cf_correct and (clean_answer != cf_answer),
                }
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved: {args.out}")
    print(f"Number of pairs: {len(df)}")

    if len(df) == 0:
        print("No counterfactual pairs found. Check the counterfactuals field in metadata.jsonl.")
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

    print("\nUsable pairs:")
    print(df["usable_for_patching"].value_counts())


if __name__ == "__main__":
    main()