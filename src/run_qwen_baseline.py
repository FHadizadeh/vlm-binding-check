import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from prompts import normalize_answer


def load_records(data_dir: Path):
    with (
        data_dir / "metadata.jsonl"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
        ]


def build_messages(
    image_path: Path,
    question: str,
):
    image_uri = (
        image_path.resolve().as_uri()
    )

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


@torch.inference_mode()
def predict_one(
    model,
    processor,
    image_path: Path,
    question: str,
    max_new_tokens: int = 8,
):
    messages = build_messages(
        image_path,
        question,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

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
        for in_ids, out_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = (
        processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    return output_text.strip()


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
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    records = load_records(
        args.data
    )

    if args.limit is not None:
        records = records[
            : args.limit
        ]

    print(
        f"Loading model: {args.model_id}"
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    )

    processor = (
        AutoProcessor.from_pretrained(
            args.model_id
        )
    )

    rows = []

    for record in tqdm(
        records,
        desc="Running baseline evaluation",
    ):
        image_path = (
            args.data
            / record["image"]
        )

        prediction_text = predict_one(
            model,
            processor,
            image_path,
            record["question"],
        )

        prediction = normalize_answer(
            prediction_text,
            record.get(
                "answer_options"
            ),
        )

        counts = record.get(
            "queried_value_counts",
            {},
        )

        answer_frequency = counts.get(
            record["answer"]
        )

        rows.append(
            {
                "sample_id": record[
                    "sample_id"
                ],
                "family_id": record.get(
                    "family_id"
                ),
                "condition": record[
                    "condition"
                ],
                "difficulty_level": record.get(
                    "difficulty_level"
                ),
                "queried_attribute": record.get(
                    "queried_attribute"
                ),
                "ambiguous_identifier": record.get(
                    "ambiguous_identifier"
                ),
                "ambiguous_identifiers": json.dumps(
                    record.get(
                        "ambiguous_identifiers",
                        [],
                    )
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
                "question": record[
                    "question"
                ],
                "answer": record[
                    "answer"
                ],
                "prediction_text": prediction_text,
                "prediction": prediction,
                "correct": (
                    prediction
                    == record["answer"]
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.out,
        index=False,
    )

    print(
        f"Saved: {args.out}"
    )

    print(
        f"Number of samples: {len(df)}"
    )

    if len(df) == 0:
        return

    print(
        "\nOverall accuracy:"
    )
    print(
        df["correct"].mean()
    )

    print(
        "\nAccuracy by queried attribute:"
    )
    print(
        df.groupby(
            "queried_attribute"
        )["correct"].agg(
            [
                "count",
                "mean",
            ]
        )
    )

    print(
        "\nAccuracy by queried attribute "
        "and answer frequency:"
    )
    print(
        df.groupby(
            [
                "queried_attribute",
                "answer_frequency",
            ]
        )["correct"].agg(
            [
                "count",
                "mean",
            ]
        )
    )

    print(
        "\nAccuracy by queried-value frequency role:"
    )
    print(
        df.groupby(
            "queried_value_frequency_role"
        )["correct"].agg(
            [
                "count",
                "mean",
            ]
        )
    )

    print(
        "\nAccuracy by queried attribute and frequency role:"
    )
    print(
        df.groupby(
            [
                "queried_attribute",
                "queried_value_frequency_role",
            ]
        )["correct"].agg(
            [
                "count",
                "mean",
            ]
        )
    )

    if (
        df["ambiguous_identifier"]
        .isin(
            [
                "shape",
                "color",
                "size",
            ]
        )
        .any()
    ):
        print(
            "\nAccuracy by ambiguous identifier "
            "(single-cue condition):"
        )

        single_rows = df[
            df[
                "ambiguous_identifier"
            ].isin(
                [
                    "shape",
                    "color",
                    "size",
                ]
            )
        ]

        print(
            single_rows.groupby(
                "ambiguous_identifier"
            )["correct"].agg(
                [
                    "count",
                    "mean",
                ]
            )
        )


if __name__ == "__main__":
    main()
