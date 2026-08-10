import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm


CANVAS_SIZE = 448
GRID_POSITIONS = [
    (84, 84), (224, 84), (364, 84),
    (84, 224), (224, 224), (364, 224),
    (84, 364), (224, 364), (364, 364),
]

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 45, 45),
    "blue": (45, 105, 220),
    "green": (45, 170, 80),
    "yellow": (245, 200, 35),
    "purple": (150, 70, 190),
    "orange": (240, 130, 30),
}

SHAPES = ["square", "circle", "triangle"]
SIZES = {
    "small": 32,
    "large": 125,
}

QUERY_ATTRIBUTES = ["size", "color", "shape"]

ATTRIBUTE_VALUES = {
    "size": list(SIZES.keys()),
    "color": list(COLORS.keys()),
    "shape": SHAPES,
}

IDENTIFIER_ATTRIBUTES = {
    "size": ("shape", "color"),
    "color": ("shape", "size"),
    "shape": ("color", "size"),
}

CONDITIONS = [
    "redundant_cues",
    "single_cue_ambiguous",
    "conjunctive_binding",
]

DIFFICULTY_LEVEL = {
    "redundant_cues": 1,
    "single_cue_ambiguous": 2,
    "conjunctive_binding": 3,
}


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


def make_obj(
    object_id: int,
    shape: str,
    color: str,
    size: str,
    center: Tuple[int, int],
) -> Obj:
    return Obj(
        object_id=object_id,
        shape=shape,
        color=color,
        size=size,
        center=center,
        bbox=bbox_from_center(center, SIZES[size]),
    )


def make_obj_from_attrs(
    object_id: int,
    attrs: Dict[str, str],
    center: Tuple[int, int],
) -> Obj:
    return make_obj(
        object_id=object_id,
        shape=attrs["shape"],
        color=attrs["color"],
        size=attrs["size"],
        center=center,
    )


def draw_object(draw: ImageDraw.ImageDraw, obj: Obj) -> None:
    fill = COLORS[obj.color]
    x1, y1, x2, y2 = obj.bbox

    if obj.shape == "square":
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=fill,
            outline=(20, 20, 20),
            width=3,
        )
    elif obj.shape == "circle":
        draw.ellipse(
            [x1, y1, x2, y2],
            fill=fill,
            outline=(20, 20, 20),
            width=3,
        )
    elif obj.shape == "triangle":
        cx, cy = obj.center
        side = x2 - x1
        h = int(side * math.sqrt(3) / 2)
        points = [
            (cx, cy - h // 2),
            (cx - side // 2, cy + h // 2),
            (cx + side // 2, cy + h // 2),
        ]
        draw.polygon(points, fill=fill)
        draw.line(
            [points[0], points[1], points[2], points[0]],
            fill=(20, 20, 20),
            width=3,
        )
    else:
        raise ValueError(f"Unknown shape: {obj.shape}")


def render_scene(objects: List[Obj], out_path: Path) -> None:
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    for x, y in GRID_POSITIONS:
        draw.ellipse(
            [x - 2, y - 2, x + 2, y + 2],
            fill=(230, 230, 230),
        )

    for obj in objects:
        draw_object(draw, obj)

    img.save(out_path)


def all_attribute_combinations() -> List[Tuple[str, str, str]]:
    return [
        (shape, color, size)
        for shape in SHAPES
        for color in COLORS
        for size in SIZES
    ]


def attrs_dict(shape: str, color: str, size: str) -> Dict[str, str]:
    return {
        "shape": shape,
        "color": color,
        "size": size,
    }


def target_attrs_for_query(
    seed: int,
    queried_attribute: str,
    query_index: int,
) -> Dict[str, str]:
    combos = all_attribute_combinations()
    block_size = len(combos)
    block = query_index // block_size
    index_in_block = query_index % block_size

    attribute_offset = {
        "size": 11_000,
        "color": 21_000,
        "shape": 31_000,
    }[queried_attribute]

    local_rng = random.Random(
        seed * 1_000_003 + attribute_offset + block * 97_409
    )
    shuffled = combos[:]
    local_rng.shuffle(shuffled)

    shape, color, size = shuffled[index_in_block]
    return attrs_dict(shape, color, size)


def family_rng(seed: int, sample_index: int) -> random.Random:
    return random.Random(
        seed * 10_000_019 + sample_index * 1_000_003 + 73_919
    )


def counterfactual_rng(seed: int, sample_index: int) -> random.Random:
    return random.Random(
        seed * 15_485_863 + sample_index * 3_245_287 + 91_337
    )


def base_distractor_candidates(
    target_attrs: Dict[str, str],
    queried_attribute: str,
) -> List[Dict[str, str]]:
    primary, secondary = IDENTIFIER_ATTRIBUTES[queried_attribute]
    candidates = []

    for shape, color, size in all_attribute_combinations():
        attrs = attrs_dict(shape, color, size)

        if attrs[primary] == target_attrs[primary]:
            continue
        if attrs[secondary] == target_attrs[secondary]:
            continue

        candidates.append(attrs)

    return candidates


def frequency_role_for_occurrence(
    queried_attribute: str,
    answer_occurrence_index: int,
) -> str:
    if queried_attribute == "size":
        roles = [
            "balanced_2_2",
            "target_majority_3_1",
            "target_minority_1_3",
        ]
    else:
        roles = [
            "target_repeated",
            "non_target_repeated",
        ]

    return roles[answer_occurrence_index % len(roles)]


def planned_distractor_query_values(
    target_value: str,
    queried_attribute: str,
    frequency_role: str,
    rng: random.Random,
) -> List[str]:
    alternatives = [
        value
        for value in ATTRIBUTE_VALUES[queried_attribute]
        if value != target_value
    ]

    if queried_attribute == "size":
        other = alternatives[0]

        if frequency_role == "balanced_2_2":
            values = [target_value, other, other]
        elif frequency_role == "target_majority_3_1":
            values = [target_value, target_value, other]
        elif frequency_role == "target_minority_1_3":
            values = [other, other, other]
        else:
            raise ValueError(f"Unknown size frequency role: {frequency_role}")

    else:
        if frequency_role == "target_repeated":
            first, second = rng.sample(alternatives, 2)
            values = [target_value, first, second]
        elif frequency_role == "non_target_repeated":
            repeated, singleton = rng.sample(alternatives, 2)
            values = [repeated, repeated, singleton]
        else:
            raise ValueError(
                f"Unknown {queried_attribute} frequency role: {frequency_role}"
            )

    rng.shuffle(values)
    return values


def sample_base_distractors(
    target_attrs: Dict[str, str],
    queried_attribute: str,
    desired_query_values: List[str],
    rng: random.Random,
) -> List[Dict[str, str]]:
    candidates = base_distractor_candidates(
        target_attrs,
        queried_attribute,
    )

    available = [dict(attrs) for attrs in candidates]
    distractors = []

    for desired_value in desired_query_values:
        matching = [
            attrs
            for attrs in available
            if attrs[queried_attribute] == desired_value
        ]

        if not matching:
            raise RuntimeError(
                f"No valid distractor available for "
                f"{queried_attribute}={desired_value}"
            )

        chosen = rng.choice(matching)
        distractors.append(dict(chosen))
        available.remove(chosen)

    return distractors


def build_base_family(
    seed: int,
    sample_index: int,
    target_attrs: Dict[str, str],
    queried_attribute: str,
    frequency_role: str,
) -> Tuple[
    List[Dict[str, str]],
    List[Tuple[int, int]],
]:
    rng = family_rng(seed, sample_index)

    desired_query_values = planned_distractor_query_values(
        target_value=target_attrs[queried_attribute],
        queried_attribute=queried_attribute,
        frequency_role=frequency_role,
        rng=rng,
    )

    base_distractors = sample_base_distractors(
        target_attrs=target_attrs,
        queried_attribute=queried_attribute,
        desired_query_values=desired_query_values,
        rng=rng,
    )

    positions = rng.sample(GRID_POSITIONS, 4)

    return base_distractors, positions


def ambiguous_identifier_for_sample(
    queried_attribute: str,
    query_index: int,
) -> str:
    primary, secondary = IDENTIFIER_ATTRIBUTES[queried_attribute]
    return primary if query_index % 2 == 0 else secondary


def ambiguity_metadata(
    condition: str,
    queried_attribute: str,
    query_index: int,
) -> Tuple[str, List[str]]:
    primary, secondary = IDENTIFIER_ATTRIBUTES[queried_attribute]

    if condition == "redundant_cues":
        return "none", []

    if condition == "single_cue_ambiguous":
        ambiguous = ambiguous_identifier_for_sample(
            queried_attribute,
            query_index,
        )
        return ambiguous, [ambiguous]

    if condition == "conjunctive_binding":
        return "both", [primary, secondary]

    raise ValueError(f"Unknown condition: {condition}")


def construct_condition_objects(
    condition: str,
    target_attrs: Dict[str, str],
    base_distractors: List[Dict[str, str]],
    positions: List[Tuple[int, int]],
    queried_attribute: str,
    query_index: int,
) -> Tuple[List[Obj], str, List[str]]:
    primary, secondary = IDENTIFIER_ATTRIBUTES[queried_attribute]

    distractors = [dict(attrs) for attrs in base_distractors]
    ambiguous_identifier, ambiguous_identifiers = ambiguity_metadata(
        condition,
        queried_attribute,
        query_index,
    )

    if condition == "redundant_cues":
        pass

    elif condition == "single_cue_ambiguous":
        if ambiguous_identifier == primary:
            distractors[0][primary] = target_attrs[primary]
        else:
            distractors[1][secondary] = target_attrs[secondary]

    elif condition == "conjunctive_binding":
        distractors[0][primary] = target_attrs[primary]
        distractors[1][secondary] = target_attrs[secondary]

    else:
        raise ValueError(f"Unknown condition: {condition}")

    objects = [
        make_obj_from_attrs(0, target_attrs, positions[0]),
        make_obj_from_attrs(1, distractors[0], positions[1]),
        make_obj_from_attrs(2, distractors[1], positions[2]),
        make_obj_from_attrs(3, distractors[2], positions[3]),
    ]

    validate_condition(
        objects=objects,
        target_id=0,
        queried_attribute=queried_attribute,
        condition=condition,
        ambiguous_identifier=ambiguous_identifier,
        ambiguous_identifiers=ambiguous_identifiers,
    )

    return objects, ambiguous_identifier, ambiguous_identifiers


def build_question(target: Obj, queried_attribute: str) -> str:
    prefix = (
        "Each object has a color, a shape, and one of two visual sizes: "
        "small or large. "
    )

    if queried_attribute == "size":
        return (
            prefix
            + f"What is the size of the {target.color} {target.shape}? "
            + "Answer with exactly one word: small or large."
        )

    if queried_attribute == "color":
        return (
            prefix
            + f"What is the color of the {target.size} {target.shape}? "
            + "Answer with exactly one word: red, blue, green, yellow, purple, or orange."
        )

    if queried_attribute == "shape":
        return (
            prefix
            + f"What is the shape of the {target.size} {target.color} object? "
            + "Answer with exactly one word: square, circle, or triangle."
        )

    raise ValueError(f"Unknown queried attribute: {queried_attribute}")


def query_constraints(target: Obj, queried_attribute: str) -> Dict[str, str]:
    return {
        attribute: getattr(target, attribute)
        for attribute in IDENTIFIER_ATTRIBUTES[queried_attribute]
    }


def matching_objects(
    objects: List[Obj],
    constraints: Dict[str, str],
) -> List[Obj]:
    return [
        obj
        for obj in objects
        if all(
            getattr(obj, attribute) == value
            for attribute, value in constraints.items()
        )
    ]


def find_answer_for_question(
    objects: List[Obj],
    queried_attribute: str,
    constraints: Dict[str, str],
) -> Optional[str]:
    matches = matching_objects(objects, constraints)
    if len(matches) != 1:
        return None
    return getattr(matches[0], queried_attribute)


def queried_value_counts(
    objects: List[Obj],
    queried_attribute: str,
) -> Dict[str, int]:
    return {
        value: sum(
            getattr(obj, queried_attribute) == value
            for obj in objects
        )
        for value in ATTRIBUTE_VALUES[queried_attribute]
    }


def boxes_overlap(a: Obj, b: Obj) -> bool:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    return (
        max(ax1, bx1) < min(ax2, bx2)
        and max(ay1, by1) < min(ay2, by2)
    )


def validate_geometry(objects: List[Obj]) -> None:
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        if x1 < 0 or y1 < 0 or x2 >= CANVAS_SIZE or y2 >= CANVAS_SIZE:
            raise RuntimeError(f"Object {obj.object_id} extends outside the canvas.")

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if boxes_overlap(objects[i], objects[j]):
                raise RuntimeError(
                    f"Objects {objects[i].object_id} and {objects[j].object_id} overlap."
                )


def validate_condition(
    objects: List[Obj],
    target_id: int,
    queried_attribute: str,
    condition: str,
    ambiguous_identifier: str,
    ambiguous_identifiers: List[str],
) -> None:
    target = next(obj for obj in objects if obj.object_id == target_id)
    primary, secondary = IDENTIFIER_ATTRIBUTES[queried_attribute]

    constraints = query_constraints(target, queried_attribute)
    full_matches = matching_objects(objects, constraints)

    if len(full_matches) != 1 or full_matches[0].object_id != target_id:
        raise RuntimeError(
            "The full question does not uniquely identify the target."
        )

    primary_matches = [
        obj
        for obj in objects
        if getattr(obj, primary) == getattr(target, primary)
    ]

    secondary_matches = [
        obj
        for obj in objects
        if getattr(obj, secondary) == getattr(target, secondary)
    ]

    validate_geometry(objects)

    if condition == "redundant_cues":
        if len(primary_matches) != 1 or len(secondary_matches) != 1:
            raise RuntimeError(
                "redundant_cues requires both identifiers to be individually unique."
            )
        if ambiguous_identifier != "none" or ambiguous_identifiers != []:
            raise RuntimeError(
                "redundant_cues must store no ambiguous identifiers."
            )

    elif condition == "single_cue_ambiguous":
        if ambiguous_identifier not in (primary, secondary):
            raise RuntimeError(
                "single_cue_ambiguous requires exactly one ambiguous identifier."
            )

        if ambiguous_identifiers != [ambiguous_identifier]:
            raise RuntimeError(
                "single_cue_ambiguous has inconsistent ambiguity metadata."
            )

        expected_primary = 2 if ambiguous_identifier == primary else 1
        expected_secondary = 2 if ambiguous_identifier == secondary else 1

        if len(primary_matches) != expected_primary:
            raise RuntimeError(
                "single_cue_ambiguous has the wrong primary-identifier multiplicity."
            )

        if len(secondary_matches) != expected_secondary:
            raise RuntimeError(
                "single_cue_ambiguous has the wrong secondary-identifier multiplicity."
            )

    elif condition == "conjunctive_binding":
        if len(primary_matches) != 2 or len(secondary_matches) != 2:
            raise RuntimeError(
                "conjunctive_binding requires one primary-match distractor "
                "and one secondary-match distractor."
            )

        if ambiguous_identifier != "both":
            raise RuntimeError(
                "conjunctive_binding must store ambiguous_identifier='both'."
            )

        if ambiguous_identifiers != [primary, secondary]:
            raise RuntimeError(
                "conjunctive_binding must store both identifier attributes "
                "in ambiguous_identifiers."
            )

    else:
        raise ValueError(f"Unknown condition: {condition}")


def objects_to_dicts(objects: List[Obj]) -> List[dict]:
    return [asdict(obj) for obj in objects]


def clone_objects(objects: List[Obj]) -> List[Obj]:
    return [Obj(**asdict(obj)) for obj in objects]


def recompute_bbox(obj: Obj) -> None:
    obj.bbox = bbox_from_center(obj.center, SIZES[obj.size])


def different_attribute_value(
    rng: random.Random,
    attribute: str,
    old_value: str,
) -> str:
    choices = [
        value
        for value in ATTRIBUTE_VALUES[attribute]
        if value != old_value
    ]
    return rng.choice(choices)


def make_counterfactual(
    clean_objects: List[Obj],
    target_id: int,
    queried_attribute: str,
    rng: random.Random,
) -> Tuple[str, List[Obj]]:
    cf_objects = clone_objects(clean_objects)
    target = next(
        obj
        for obj in cf_objects
        if obj.object_id == target_id
    )

    old_value = getattr(target, queried_attribute)
    new_value = different_attribute_value(
        rng,
        queried_attribute,
        old_value,
    )

    setattr(target, queried_attribute, new_value)

    if queried_attribute == "size":
        recompute_bbox(target)

    return f"{queried_attribute}_swap", cf_objects


def validate_counterfactual(
    clean_objects: List[Obj],
    cf_objects: List[Obj],
    target_id: int,
    queried_attribute: str,
    constraints: Dict[str, str],
) -> Tuple[str, str]:
    clean_answer = find_answer_for_question(
        clean_objects,
        queried_attribute,
        constraints,
    )

    cf_answer = find_answer_for_question(
        cf_objects,
        queried_attribute,
        constraints,
    )

    if clean_answer is None:
        raise RuntimeError("Clean question is ambiguous.")

    if cf_answer is None:
        raise RuntimeError(
            "Counterfactual made the question ambiguous."
        )

    if clean_answer == cf_answer:
        raise RuntimeError(
            "Counterfactual did not change the correct answer."
        )

    clean_by_id = {
        obj.object_id: obj
        for obj in clean_objects
    }

    cf_by_id = {
        obj.object_id: obj
        for obj in cf_objects
    }

    if set(clean_by_id) != set(cf_by_id):
        raise RuntimeError(
            "Object IDs changed between clean and counterfactual."
        )

    for object_id in sorted(clean_by_id):
        clean_obj = clean_by_id[object_id]
        cf_obj = cf_by_id[object_id]

        if clean_obj.center != cf_obj.center:
            raise RuntimeError(
                "An object position changed in the counterfactual."
            )

        changed_semantic_attributes = [
            attribute
            for attribute in ("shape", "color", "size")
            if getattr(clean_obj, attribute)
            != getattr(cf_obj, attribute)
        ]

        if object_id == target_id:
            if changed_semantic_attributes != [queried_attribute]:
                raise RuntimeError(
                    "The target must change in exactly the queried attribute."
                )
        elif changed_semantic_attributes:
            raise RuntimeError(
                "A distractor changed in the counterfactual."
            )

    return clean_answer, cf_answer


def generate_dataset(
    condition: str,
    n: int,
    seed: int,
    out: Path,
    query_attribute: str = "mixed",
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    images_dir = out / "images"
    cf_dir = out / "counterfactuals"

    images_dir.mkdir(exist_ok=True)
    cf_dir.mkdir(exist_ok=True)

    metadata_path = out / "metadata.jsonl"
    query_counts = {
        attribute: 0
        for attribute in QUERY_ATTRIBUTES
    }
    answer_occurrence_counts = {
        attribute: {
            value: 0
            for value in ATTRIBUTE_VALUES[attribute]
        }
        for attribute in QUERY_ATTRIBUTES
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        for i in tqdm(
            range(n),
            desc=f"Generating {condition}",
        ):
            queried_attribute = (
                QUERY_ATTRIBUTES[i % len(QUERY_ATTRIBUTES)]
                if query_attribute == "mixed"
                else query_attribute
            )

            query_index = query_counts[queried_attribute]
            query_counts[queried_attribute] += 1

            target_attrs = target_attrs_for_query(
                seed,
                queried_attribute,
                query_index,
            )

            target_answer = target_attrs[queried_attribute]
            answer_occurrence_index = answer_occurrence_counts[
                queried_attribute
            ][target_answer]
            answer_occurrence_counts[queried_attribute][target_answer] += 1

            frequency_role = frequency_role_for_occurrence(
                queried_attribute,
                answer_occurrence_index,
            )

            (
                base_distractors,
                positions,
            ) = build_base_family(
                seed=seed,
                sample_index=i,
                target_attrs=target_attrs,
                queried_attribute=queried_attribute,
                frequency_role=frequency_role,
            )

            (
                objects,
                ambiguous_identifier,
                ambiguous_identifiers,
            ) = construct_condition_objects(
                condition=condition,
                target_attrs=target_attrs,
                base_distractors=base_distractors,
                positions=positions,
                queried_attribute=queried_attribute,
                query_index=query_index,
            )

            target_id = 0
            target = objects[target_id]

            question = build_question(
                target,
                queried_attribute,
            )

            constraints = query_constraints(
                target,
                queried_attribute,
            )

            answer = find_answer_for_question(
                objects,
                queried_attribute,
                constraints,
            )

            if answer is None:
                raise RuntimeError(
                    "Invalid or ambiguous clean query."
                )

            family_id = f"family_{i:05d}"
            sample_id = f"{condition}_{i:05d}"

            image_name = f"{sample_id}.png"
            render_scene(
                objects,
                images_dir / image_name,
            )

            cf_name, cf_objects = make_counterfactual(
                objects,
                target_id,
                queried_attribute,
                counterfactual_rng(seed, i),
            )

            validate_geometry(cf_objects)

            clean_answer, cf_answer = validate_counterfactual(
                objects,
                cf_objects,
                target_id,
                queried_attribute,
                constraints,
            )

            if clean_answer != answer:
                raise RuntimeError(
                    "Clean-answer validation failed."
                )

            cf_image_name = (
                f"{sample_id}_{cf_name}.png"
            )

            render_scene(
                cf_objects,
                cf_dir / cf_image_name,
            )

            primary, secondary = (
                IDENTIFIER_ATTRIBUTES[queried_attribute]
            )

            identifier_match_counts = {
                primary: sum(
                    getattr(obj, primary)
                    == getattr(target, primary)
                    for obj in objects
                ),
                secondary: sum(
                    getattr(obj, secondary)
                    == getattr(target, secondary)
                    for obj in objects
                ),
            }

            frequency_counts = queried_value_counts(
                objects,
                queried_attribute,
            )

            cf_frequency_counts = queried_value_counts(
                cf_objects,
                queried_attribute,
            )

            cf_records = {
                cf_name: {
                    "image": str(
                        Path("counterfactuals")
                        / cf_image_name
                    ),
                    "question": question,
                    "answer": cf_answer,
                    "queried_attribute": queried_attribute,
                    "changed_attribute": queried_attribute,
                    "changed_object_id": target_id,
                    "queried_value_counts": cf_frequency_counts,
                    "target_object": asdict(
                        cf_objects[target_id]
                    ),
                    "objects": objects_to_dicts(
                        cf_objects
                    ),
                }
            }

            record = {
                "sample_id": sample_id,
                "family_id": family_id,
                "condition": condition,
                "difficulty_level": DIFFICULTY_LEVEL[condition],
                "queried_attribute": queried_attribute,
                "identifier_attributes": list(
                    IDENTIFIER_ATTRIBUTES[queried_attribute]
                ),
                "ambiguous_identifier": ambiguous_identifier,
                "ambiguous_identifiers": ambiguous_identifiers,
                "queried_value_frequency_role": frequency_role,
                "queried_value_counts": frequency_counts,
                "identifier_match_counts": identifier_match_counts,
                "image": str(
                    Path("images") / image_name
                ),
                "question": question,
                "answer": answer,
                "answer_options": ATTRIBUTE_VALUES[
                    queried_attribute
                ],
                "target_object_id": target_id,
                "query_color": constraints.get("color"),
                "query_shape": constraints.get("shape"),
                "query_size": constraints.get("size"),
                "query_attributes": constraints,
                "target_object": asdict(target),
                "objects": objects_to_dicts(objects),
                "counterfactuals": cf_records,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "condition": condition,
        "difficulty_level": DIFFICULTY_LEVEL[condition],
        "n": n,
        "seed": seed,
        "query_attribute": query_attribute,
        "query_attributes": QUERY_ATTRIBUTES,
        "paired_generation": (
            "Use the same seed, n, and query_attribute for all three conditions. "
            "Records with the same family_id share the target, question, object "
            "positions, queried-attribute values, and base distractor family. "
            "Only identifier attributes needed to create the condition are changed."
        ),
        "queried_value_sampling": (
            "Queried-attribute frequency is controlled but answer-independent. "
            "For color and shape, the profile is always 2-1-1, with the target "
            "value repeated in half of the occurrences of each answer and a "
            "non-target value repeated in the other half. For size, each answer "
            "cycles through 2-2, target-majority 3-1, and target-minority 1-3. "
            "The same profile and queried-attribute values are preserved across "
            "paired conditions for a family."
        ),
        "ambiguity_metadata": (
            "ambiguous_identifier is 'none' for redundant_cues, the single "
            "ambiguous attribute for single_cue_ambiguous, and 'both' for "
            "conjunctive_binding. ambiguous_identifiers stores the corresponding list."
        ),
        "geometry_control": (
            "Grid spacing is larger than the maximum object side length, and "
            "clean/counterfactual bounding boxes are validated to be non-overlapping."
        ),
        "canvas_size": CANVAS_SIZE,
        "colors": COLORS,
        "shapes": SHAPES,
        "sizes": SIZES,
        "metadata": "metadata.jsonl",
    }

    (out / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        required=True,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=108,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query_attribute",
        choices=["mixed"] + QUERY_ATTRIBUTES,
        default="mixed",
    )

    args = parser.parse_args()

    generate_dataset(
        condition=args.condition,
        n=args.n,
        seed=args.seed,
        out=args.out,
        query_attribute=args.query_attribute,
    )


if __name__ == "__main__":
    main()
