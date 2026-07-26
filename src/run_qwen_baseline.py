import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
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
def predict_one(model, processor, image_path: Path, question: str, max_new_tokens: int = 8) -> str:
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
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Remove input prompt tokens from generated sequence.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
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

    for r in tqdm(records, desc="Running baseline"):
        image_path = args.data / r["image"]
        pred_text = predict_one(model, processor, image_path, r["question"])
        pred = normalize_answer(pred_text)

        rows.append(
            {
                "sample_id": r["sample_id"],
                "condition": r["condition"],
                "question": r["question"],
                "answer": r["answer"],
                "prediction_text": pred_text,
                "prediction": pred,
                "correct": pred == r["answer"],
                "image": r["image"],
            }
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    acc = df["correct"].mean()
    print(f"Saved: {args.out}")
    print(f"Accuracy: {acc:.4f}")
    print(df["prediction"].value_counts())


if __name__ == "__main__":
    main()