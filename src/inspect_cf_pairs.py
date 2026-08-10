import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def wrap(text, width=70):
    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
        )
    )


def draw_multiline(
    draw,
    xy,
    text,
    fill=(0, 0, 0),
    line_spacing=4,
):
    x, y = xy

    for line in str(text).split("\n"):
        draw.text(
            (x, y),
            line,
            fill=fill,
        )

        y += (
            14
            + line_spacing
        )

    return y


def make_pair_image(
    row,
    data_dir: Path,
    out_path: Path,
):
    clean_path = (
        data_dir
        / row["clean_image"]
    )

    cf_path = (
        data_dir
        / row["cf_image"]
    )

    clean_img = Image.open(
        clean_path
    ).convert("RGB")

    cf_img = Image.open(
        cf_path
    ).convert("RGB")

    w, h = clean_img.size
    top_h = 270
    gap = 20

    canvas = Image.new(
        "RGB",
        (
            2 * w + gap,
            h + top_h,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    queried_attribute = row.get(
        "queried_attribute",
        "",
    )

    ambiguous_identifier = row.get(
        "ambiguous_identifier",
        "",
    )

    family_id = row.get(
        "family_id",
        "",
    )

    title = (
        f"sample_id={row['sample_id']} | "
        f"family_id={family_id} | "
        f"query={queried_attribute} | "
        f"ambiguous={ambiguous_identifier} | "
        f"freq_role={row.get('queried_value_frequency_role', '')} | "
        f"cf_type={row['cf_type']} | "
        f"usable={row['usable_for_patching']}"
    )

    draw.text(
        (
            10,
            10,
        ),
        title,
        fill=(0, 0, 0),
    )

    left_text = (
        "CLEAN\n"
        f"Q: {row['clean_question']}\n"
        f"gold: {row['clean_answer']} | "
        f"pred: {row['clean_prediction']} "
        f"({row['clean_prediction_text']})\n"
        f"answer frequency: "
        f"{row.get('clean_answer_frequency', '')}"
    )

    right_text = (
        "COUNTERFACTUAL\n"
        f"Q: {row['cf_question']}\n"
        f"gold: {row['cf_answer']} | "
        f"pred: {row['cf_prediction']} "
        f"({row['cf_prediction_text']})\n"
        f"answer frequency: "
        f"{row.get('cf_answer_frequency', '')}"
    )

    draw_multiline(
        draw,
        (
            10,
            45,
        ),
        wrap(
            left_text,
            60,
        ),
    )

    draw_multiline(
        draw,
        (
            w + gap + 10,
            45,
        ),
        wrap(
            right_text,
            60,
        ),
    )

    canvas.paste(
        clean_img,
        (
            0,
            top_h,
        ),
    )

    canvas.paste(
        cf_img,
        (
            w + gap,
            top_h,
        ),
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        out_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "inspection/cf_pairs"
        ),
    )

    parser.add_argument(
        "--cf_type",
        type=str,
        choices=[
            "size_swap",
            "color_swap",
            "shape_swap",
        ],
        default=None,
    )

    parser.add_argument(
        "--queried_attribute",
        type=str,
        choices=[
            "size",
            "color",
            "shape",
        ],
        default=None,
    )

    parser.add_argument(
        "--ambiguous_identifier",
        type=str,
        choices=[
            "none",
            "shape",
            "color",
            "size",
            "both",
        ],
        default=None,
    )

    parser.add_argument(
        "--only_failures",
        action="store_true",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.csv
    )

    if args.cf_type is not None:
        df = df[
            df["cf_type"]
            == args.cf_type
        ].copy()

    if (
        args.queried_attribute
        is not None
    ):
        if (
            "queried_attribute"
            not in df.columns
        ):
            raise ValueError(
                "CSV has no queried_attribute column."
            )

        df = df[
            df["queried_attribute"]
            == args.queried_attribute
        ].copy()

    if (
        args.ambiguous_identifier
        is not None
    ):
        if (
            "ambiguous_identifier"
            not in df.columns
        ):
            raise ValueError(
                "CSV has no ambiguous_identifier column."
            )

        df = df[
            df["ambiguous_identifier"]
            == args.ambiguous_identifier
        ].copy()

    if args.only_failures:
        df = df[
            df[
                "usable_for_patching"
            ]
            == False
        ].copy()

    print(
        f"Found {len(df)} rows"
    )

    if len(df) == 0:
        return

    for _, row in df.head(
        args.n
    ).iterrows():
        out_path = (
            args.out_dir
            / (
                f"sample_{row['sample_id']}_"
                f"{row['cf_type']}_"
                f"usable_"
                f"{row['usable_for_patching']}.png"
            )
        )

        make_pair_image(
            row,
            args.data,
            out_path,
        )

        print(
            out_path
        )


if __name__ == "__main__":
    main()
