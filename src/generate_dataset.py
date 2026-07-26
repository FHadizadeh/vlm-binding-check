import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm


CANVAS_SIZE = 448
GRID_POSITIONS = [
    (112, 112), (224, 112), (336, 112),
    (112, 224), (224, 224), (336, 224),
    (112, 336), (224, 336), (336, 336),
]

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 50, 47),
    "blue": (38, 139, 210),
    "green": (133, 153, 0),
    "yellow": (181, 137, 0),
    "purple": (108, 113, 196),
    "orange": (203, 75, 22),
}

SHAPES = ["square", "circle", "triangle"]
SIZES = {"small": 42, "medium": 62, "large": 82}


@dataclass
class Obj:
    object_id: int
    shape: str
    color: str
    size: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


def bbox_from_center(center: Tuple[int, int], side: int) -> Tuple[int, int, int, int]:
    x, y = center
    half = side // 2
    return (x - half, y - half, x + half, y + half)


def make_obj(object_id: int, shape: str, color: str, size: str, center: Tuple[int, int]) -> Obj:
    return Obj(
        object_id=object_id,
        shape=shape,
        color=color,
        size=size,
        center=center,
        bbox=bbox_from_center(center, SIZES[size]),
    )


def draw_object(draw: ImageDraw.ImageDraw, obj: Obj) -> None:
    fill = COLORS[obj.color]
    x1, y1, x2, y2 = obj.bbox

    if obj.shape == "square":
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=(20, 20, 20), width=3)
    elif obj.shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=fill, outline=(20, 20, 20), width=3)
    elif obj.shape == "triangle":
        cx, cy = obj.center
        side = x2 - x1
        h = int(side * math.sqrt(3) / 2)
        points = [(cx, cy - h // 2), (cx - side // 2, cy + h // 2), (cx + side // 2, cy + h // 2)]
        draw.polygon(points, fill=fill)
        draw.line([points[0], points[1], points[2], points[0]], fill=(20, 20, 20), width=3)
    else:
        raise ValueError(f"Unknown shape: {obj.shape}")


def render_scene(objects: List[Obj], out_path: Path) -> None:
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Faint grid markers make positions stable without being visually dominant.
    for x, y in GRID_POSITIONS:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(230, 230, 230))

    for obj in objects:
        draw_object(draw, obj)

    img.save(out_path)


def sample_positions(rng: random.Random, k: int) -> List[Tuple[int, int]]:
    return rng.sample(GRID_POSITIONS, k)


def different_size(rng: random.Random, old_size: str) -> str:
    return rng.choice([s for s in SIZES if s != old_size])


def different_color(rng: random.Random, old_color: str) -> str:
    return rng.choice([c for c in COLORS if c != old_color])


def different_shape(rng: random.Random, old_shape: str) -> str:
    return rng.choice([s for s in SHAPES if s != old_shape])


def choose_target_attrs(rng: random.Random, sample_id: Optional[int] = None) -> Tuple[str, str, str]:
    """
    Choose target attributes.

    If sample_id is provided, we cycle through all shape-color-size combinations
    in shuffled blocks. This makes target shape/color/size approximately balanced.
    """
    combos = [
        (shape, color, size)
        for shape in SHAPES
        for color in COLORS.keys()
        for size in SIZES.keys()
    ]

    if sample_id is None:
        return rng.choice(combos)

    block = sample_id // len(combos)
    idx = sample_id % len(combos)

    # Shuffle each full block deterministically so the order is not always structured.
    local_rng = random.Random(10_000 + block)
    local_combos = combos[:]
    local_rng.shuffle(local_combos)

    return local_combos[idx]

def build_unique_shape(rng: random.Random, sample_id: int) -> Tuple[List[Obj], int, str]:
    target_shape, target_color, target_size = choose_target_attrs(rng, sample_id)
    positions = sample_positions(rng, 4)

    objects = [
        make_obj(0, target_shape, target_color, target_size, positions[0])
    ]

    # Fillers should NOT have the target shape.
    # This keeps the queried shape unique.
    other_shapes = [s for s in SHAPES if s != target_shape]

    for oid in range(1, 4):
        shape = rng.choice(other_shapes)
        color = rng.choice(list(COLORS.keys()))
        size = rng.choice(list(SIZES.keys()))
        objects.append(make_obj(oid, shape, color, size, positions[oid]))

    # Color is mentioned but logically redundant, because the target shape is unique.
    question = f"What is the size of the {target_color} {target_shape}? Answer with one word: small, medium, or large."
    return objects, 0, question


def build_multi_same_shape(rng: random.Random, sample_id: int) -> Tuple[List[Obj], int, str]:
    target_shape, target_color, target_size = choose_target_attrs(rng, sample_id)
    positions = sample_positions(rng, 4)

    distractor_color = different_color(rng, target_color)
    distractor_size = different_size(rng, target_size)

    objects = [
        # target
        make_obj(0, target_shape, target_color, target_size, positions[0]),

        # same-shape distractor: color is needed to choose the target
        make_obj(1, target_shape, distractor_color, distractor_size, positions[1]),
    ]

    # Two fillers with shapes different from the target shape.
    other_shapes = [s for s in SHAPES if s != target_shape]

    for oid in range(2, 4):
        shape = rng.choice(other_shapes)
        color = rng.choice(list(COLORS.keys()))
        size = rng.choice(list(SIZES.keys()))
        objects.append(make_obj(oid, shape, color, size, positions[oid]))

    question = f"What is the size of the {target_color} {target_shape}? Answer with one word: small, medium, or large."
    return objects, 0, question


def build_compositional_distractor(rng: random.Random, sample_id: int) -> Tuple[List[Obj], int, str]:
    target_shape, target_color, target_size = choose_target_attrs(rng, sample_id)
    positions = sample_positions(rng, 4)

    same_shape_color = different_color(rng, target_color)
    same_color_shape = different_shape(rng, target_shape)

    # Choose an unrelated distractor that does not duplicate the target color-shape pair,
    # otherwise the query like "blue square" would become ambiguous.
    while True:
        shape4 = rng.choice(SHAPES)
        color4 = rng.choice(list(COLORS.keys()))
        if not (shape4 == target_shape and color4 == target_color):
            break

    objects = [
        # target
        make_obj(0, target_shape, target_color, target_size, positions[0]),
        # same-shape distractor: color distinguishes it from target
        make_obj(1, target_shape, same_shape_color, different_size(rng, target_size), positions[1]),
        # same-color distractor: shape distinguishes it from target
        make_obj(2, same_color_shape, target_color, different_size(rng, target_size), positions[2]),
        # unrelated distractor
        make_obj(3, shape4, color4, rng.choice(list(SIZES.keys())), positions[3]),
    ]

    # Both color and shape are necessary: target = color ∩ shape.
    question = f"What is the size of the {target_color} {target_shape}? Answer with one word: small, medium, or large."
    return objects, 0, question


BUILDERS = {
    "unique_shape": build_unique_shape,
    "multi_same_shape": build_multi_same_shape,
    "compositional_distractor": build_compositional_distractor,
}


def objects_to_dicts(objects: List[Obj]) -> List[dict]:
    return [asdict(o) for o in objects]


def clone_objects(objects: List[Obj]) -> List[Obj]:
    return [Obj(**asdict(o)) for o in objects]


def recompute_bbox(obj: Obj) -> None:
    obj.bbox = bbox_from_center(obj.center, SIZES[obj.size])


def make_counterfactuals(objects: List[Obj], target_id: int, rng: random.Random) -> Dict[str, List[Obj]]:
    target = objects[target_id]
    cfs: Dict[str, List[Obj]] = {}

    # size change for the target: same selected object, different queried answer
    size_objs = clone_objects(objects)
    size_objs[target_id].size = different_size(rng, target.size)
    recompute_bbox(size_objs[target_id])
    cfs["size_swap"] = size_objs

    # color swap with same-shape distractor: tests color-based target selection
    same_shape = [o for o in objects if o.object_id != target_id and o.shape == target.shape]
    if same_shape:
        d = same_shape[0]
        color_objs = clone_objects(objects)
        color_objs[target_id].color, color_objs[d.object_id].color = color_objs[d.object_id].color, color_objs[target_id].color
        cfs["color_swap"] = color_objs

    # shape swap with same-color distractor: tests shape-based target selection
    same_color = [o for o in objects if o.object_id != target_id and o.color == target.color]
    if same_color:
        d = same_color[0]
        shape_objs = clone_objects(objects)
        shape_objs[target_id].shape, shape_objs[d.object_id].shape = shape_objs[d.object_id].shape, shape_objs[target_id].shape
        cfs["shape_swap"] = shape_objs

    return cfs


def find_answer_for_question(objects: List[Obj], query_color: Optional[str], query_shape: str) -> Optional[str]:
    matches = []
    for obj in objects:
        if obj.shape != query_shape:
            continue
        if query_color is not None and obj.color != query_color:
            continue
        matches.append(obj)
    if len(matches) != 1:
        return None
    return matches[0].size


def parse_query_from_target(objects: List[Obj], target_id: int, condition: str) -> Tuple[Optional[str], str]:
    target = objects[target_id]
    if condition == "unique_shape":
        return None, target.shape
    return target.color, target.shape


def generate_dataset(condition: str, n: int, seed: int, out: Path) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    cf_dir = out / "counterfactuals"
    images_dir.mkdir(exist_ok=True)
    cf_dir.mkdir(exist_ok=True)

    builder = BUILDERS[condition]
    metadata_path = out / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=f"Generating {condition}"):
            objects, target_id, question = builder(rng, i)
            target = objects[target_id]
            query_color, query_shape = parse_query_from_target(objects, target_id, condition)
            answer = find_answer_for_question(objects, query_color, query_shape)
            if answer is None:
                raise RuntimeError("Invalid/ambiguous generated query.")

            image_name = f"{condition}_{i:05d}.png"
            render_scene(objects, images_dir / image_name)

            cf_records = {}
            for cf_name, cf_objects in make_counterfactuals(objects, target_id, rng).items():
                cf_answer = find_answer_for_question(cf_objects, query_color, query_shape)
                if cf_answer is None:
                    # Skip counterfactuals that break query uniqueness.
                    continue
                cf_image_name = f"{condition}_{i:05d}_{cf_name}.png"
                render_scene(cf_objects, cf_dir / cf_image_name)
                cf_records[cf_name] = {
                    "image": str(Path("counterfactuals") / cf_image_name),
                    "answer": cf_answer,
                    "objects": objects_to_dicts(cf_objects),
                }

            record = {
                "sample_id": f"{condition}_{i:05d}",
                "condition": condition,
                "image": str(Path("images") / image_name),
                "question": question,
                "answer": answer,
                "answer_options": list(SIZES.keys()),
                "target_object_id": target_id,
                "query_color": query_color,
                "query_shape": query_shape,
                "target_object": asdict(target),
                "objects": objects_to_dicts(objects),
                "counterfactuals": cf_records,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "condition": condition,
        "n": n,
        "seed": seed,
        "canvas_size": CANVAS_SIZE,
        "colors": COLORS,
        "shapes": SHAPES,
        "sizes": SIZES,
        "metadata": "metadata.jsonl",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(BUILDERS.keys()), required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate_dataset(args.condition, args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
