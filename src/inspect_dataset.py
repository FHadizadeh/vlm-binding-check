import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_grid(data_dir: Path, records, n: int, out_name: str = "samples_grid.png"):
    records = records[:n]
    thumbs = []
    cell_w, cell_h = 320, 390

    for r in records:
        img = Image.open(data_dir / r["image"]).convert("RGB")
        img.thumbnail((280, 280))
        canvas = Image.new("RGB", (cell_w, cell_h), "white")
        x = (cell_w - img.width) // 2
        canvas.paste(img, (x, 10))

        draw = ImageDraw.Draw(canvas)
        text = f'{r["sample_id"]}\nQ: {r["question"]}\nA: {r["answer"]}'
        draw.multiline_text((10, 300), text, fill=(0, 0, 0), spacing=4)
        thumbs.append(canvas)

    cols = min(3, max(1, len(thumbs)))
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (240, 240, 240))

    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        grid.paste(thumb, (col * cell_w, row * cell_h))

    out_path = data_dir / out_name
    grid.save(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    records = load_records(args.data)
    print(f"Loaded {len(records)} records")
    for r in records[: min(5, len(records))]:
        print(r["sample_id"], "|", r["question"], "|", r["answer"], "| cfs:", list(r["counterfactuals"].keys()))

    make_grid(args.data, records, args.n)


if __name__ == "__main__":
    main()
