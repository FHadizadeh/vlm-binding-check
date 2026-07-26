import argparse
import json
from pathlib import Path
from collections import Counter


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_record(r):
    objects = r["objects"]
    target = r["target_object"]
    condition = r["condition"]

    target_shape = target["shape"]
    target_color = target["color"]

    same_shape = [o for o in objects if o["shape"] == target_shape]
    same_color = [o for o in objects if o["color"] == target_color]
    matching_query = [
        o for o in objects
        if o["shape"] == target_shape and o["color"] == target_color
    ]

    errors = []

    if len(objects) != 4:
        errors.append(f"expected 4 objects, got {len(objects)}")

    if len(matching_query) != 1:
        errors.append(f"query is ambiguous/missing: {len(matching_query)} matches")

    if condition == "unique_shape":
        if len(same_shape) != 1:
            errors.append(f"unique_shape violated: {len(same_shape)} objects with target shape")

    elif condition == "multi_same_shape":
        if len(same_shape) < 2:
            errors.append(f"multi_same_shape violated: only {len(same_shape)} objects with target shape")
        if len(same_color) < 1:
            errors.append("target color missing somehow")

    elif condition == "compositional_distractor":
        if len(same_shape) < 2:
            errors.append("missing same-shape distractor")
        if len(same_color) < 2:
            errors.append("missing same-color distractor")

    else:
        errors.append(f"unknown condition: {condition}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.data)
    all_errors = []

    answer_counter = Counter()
    target_shape_counter = Counter()
    target_color_counter = Counter()

    for r in records:
        errors = validate_record(r)
        if errors:
            all_errors.append((r["sample_id"], errors))

        answer_counter[r["answer"]] += 1
        target_shape_counter[r["target_object"]["shape"]] += 1
        target_color_counter[r["target_object"]["color"]] += 1

    print(f"Loaded {len(records)} records")
    print(f"Number of invalid records: {len(all_errors)}")

    if all_errors:
        for sample_id, errors in all_errors[:20]:
            print(sample_id, errors)

    print("\nAnswer distribution:")
    print(answer_counter)

    print("\nTarget shape distribution:")
    print(target_shape_counter)

    print("\nTarget color distribution:")
    print(target_color_counter)


if __name__ == "__main__":
    main()