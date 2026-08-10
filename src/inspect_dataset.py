import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw


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


def wrap(text, width=48):
    lines = []

    for part in str(text).split("\n"):
        wrapped = textwrap.wrap(
            part,
            width=width,
        )

        lines.extend(
            wrapped
            if wrapped
            else [""]
        )

    return "\n".join(lines)


def make_grid(
    data_dir: Path,
    records,
    n: int,
    out_name: str = "samples_grid.png",
):
    records = records[:n]
    thumbs = []

    cell_w = 380
    image_max = 280
    text_y = 300
    cell_h = 550

    for record in records:
        img = Image.open(
            data_dir / record["image"]
        ).convert("RGB")

        img.thumbnail(
            (
                image_max,
                image_max,
            )
        )

        canvas = Image.new(
            "RGB",
            (
                cell_w,
                cell_h,
            ),
            "white",
        )

        x = (
            cell_w - img.width
        ) // 2

        canvas.paste(
            img,
            (
                x,
                10,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        text = (
            f'{record["sample_id"]}\n'
            f'family: {record.get("family_id", "?")}\n'
            f'condition: {record.get("condition", "?")}\n'
            f'query: {record.get("queried_attribute", "?")}\n'
            f'ambiguous cue: {record.get("ambiguous_identifier", "?")}\n'
            f'ambiguous cues: {record.get("ambiguous_identifiers", [])}\n'
            f'frequency role: {record.get("queried_value_frequency_role", "?")}\n'
            f'queried-value counts: {record.get("queried_value_counts", {})}\n'
            f'Q: {record["question"]}\n'
            f'A: {record["answer"]}'
        )

        draw.multiline_text(
            (
                10,
                text_y,
            ),
            wrap(text),
            fill=(0, 0, 0),
            spacing=4,
        )

        thumbs.append(
            canvas
        )

    cols = min(
        3,
        max(
            1,
            len(thumbs),
        ),
    )

    rows = (
        len(thumbs)
        + cols
        - 1
    ) // cols

    grid = Image.new(
        "RGB",
        (
            cols * cell_w,
            rows * cell_h,
        ),
        (
            240,
            240,
            240,
        ),
    )

    for idx, thumb in enumerate(
        thumbs
    ):
        row, col = divmod(
            idx,
            cols,
        )

        grid.paste(
            thumb,
            (
                col * cell_w,
                row * cell_h,
            ),
        )

    out_path = (
        data_dir
        / out_name
    )

    grid.save(
        out_path
    )

    print(
        f"Saved {out_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    records = load_records(
        args.data
    )

    print(
        f"Loaded {len(records)} records"
    )

    for record in records[
        : min(
            5,
            len(records),
        )
    ]:
        print(
            record["sample_id"],
            "| family:",
            record.get("family_id"),
            "| condition:",
            record.get("condition"),
            "| query:",
            record.get(
                "queried_attribute"
            ),
            "| ambiguous:",
            record.get(
                "ambiguous_identifier"
            ),
            "| ambiguous cues:",
            record.get(
                "ambiguous_identifiers"
            ),
            "| frequency role:",
            record.get(
                "queried_value_frequency_role"
            ),
            "| counts:",
            record.get(
                "queried_value_counts"
            ),
            "|",
            record["question"],
            "|",
            record["answer"],
            "| cfs:",
            list(
                record[
                    "counterfactuals"
                ].keys()
            ),
        )

    make_grid(
        args.data,
        records,
        args.n,
    )


if __name__ == "__main__":
    main()
