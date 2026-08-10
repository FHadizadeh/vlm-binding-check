import argparse
import json
from collections import Counter
from pathlib import Path


IDENTIFIER_ATTRIBUTES = {
    "size": ("shape", "color"),
    "color": ("shape", "size"),
    "shape": ("color", "size"),
}

ATTRIBUTE_VALUES = {
    "size": {"small", "large"},
    "color": {"red", "blue", "green", "yellow", "purple", "orange"},
    "shape": {"square", "circle", "triangle"},
}

DIFFICULTY_LEVEL = {
    "redundant_cues": 1,
    "single_cue_ambiguous": 2,
    "conjunctive_binding": 3,
}

CANVAS_SIZE = 448

SIZES = {
    "small": 32,
    "large": 125,
}


FREQUENCY_ROLES = {
    "size": [
        "balanced_2_2",
        "target_majority_3_1",
        "target_minority_1_3",
    ],
    "color": [
        "target_repeated",
        "non_target_repeated",
    ],
    "shape": [
        "target_repeated",
        "non_target_repeated",
    ],
}


def load_records(data_dir: Path):
    with (data_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def matching_objects(objects, constraints):
    return [
        obj
        for obj in objects
        if all(
            obj.get(attribute) == value
            for attribute, value in constraints.items()
        )
    ]


def semantic_changes(clean_obj, cf_obj):
    return [
        attribute
        for attribute in ("shape", "color", "size")
        if clean_obj.get(attribute) != cf_obj.get(attribute)
    ]


def bbox_from_center(center, side):
    x, y = center
    half = side // 2
    return [
        x - half,
        y - half,
        x + half,
        y + half,
    ]


def bboxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]

    return (
        max(ax1, bx1) < min(ax2, bx2)
        and max(ay1, by1) < min(ay2, by2)
    )


def validate_geometry(objects, context, errors):
    for obj in objects:
        expected_bbox = bbox_from_center(
            obj["center"],
            SIZES[obj["size"]],
        )

        if obj.get("bbox") != expected_bbox:
            errors.append(
                f"{context}: object {obj['object_id']} has bbox={obj.get('bbox')}, "
                f"expected {expected_bbox}"
            )

        x1, y1, x2, y2 = obj["bbox"]

        if (
            x1 < 0
            or y1 < 0
            or x2 >= CANVAS_SIZE
            or y2 >= CANVAS_SIZE
        ):
            errors.append(
                f"{context}: object {obj['object_id']} extends outside the canvas"
            )

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if bboxes_overlap(
                objects[i],
                objects[j],
            ):
                errors.append(
                    f"{context}: objects {objects[i]['object_id']} and "
                    f"{objects[j]['object_id']} overlap"
                )


def queried_value_counts(
    objects,
    queried_attribute,
):
    return {
        value: sum(
            obj[queried_attribute] == value
            for obj in objects
        )
        for value in ATTRIBUTE_VALUES[queried_attribute]
    }




def validate_frequency_profile(record, counts, errors):
    queried_attribute = record["queried_attribute"]
    answer = record["answer"]
    role = record.get("queried_value_frequency_role")

    if role not in FREQUENCY_ROLES[queried_attribute]:
        errors.append(
            f"invalid queried_value_frequency_role={role} "
            f"for queried_attribute={queried_attribute}"
        )
        return

    if queried_attribute in {"color", "shape"}:
        positive_counts = sorted(
            count
            for count in counts.values()
            if count > 0
        )

        if positive_counts != [1, 1, 2]:
            errors.append(
                f"{queried_attribute} frequency profile must be 2-1-1, "
                f"got {counts}"
            )
            return

        if role == "target_repeated":
            if counts[answer] != 2:
                errors.append(
                    "target_repeated requires the target answer to occur twice"
                )
        elif role == "non_target_repeated":
            if counts[answer] != 1:
                errors.append(
                    "non_target_repeated requires the target answer to occur once"
                )

            repeated_non_target = [
                value
                for value, count in counts.items()
                if value != answer and count == 2
            ]

            if len(repeated_non_target) != 1:
                errors.append(
                    "non_target_repeated requires exactly one non-target value "
                    "to occur twice"
                )

    elif queried_attribute == "size":
        other = next(
            value
            for value in ATTRIBUTE_VALUES["size"]
            if value != answer
        )

        expected = {
            "balanced_2_2": (2, 2),
            "target_majority_3_1": (3, 1),
            "target_minority_1_3": (1, 3),
        }[role]

        actual = (
            counts[answer],
            counts[other],
        )

        if actual != expected:
            errors.append(
                f"size frequency role {role} expects target/other counts "
                f"{expected}, got {actual}"
            )


def validate_ambiguity_metadata(
    record,
    primary,
    secondary,
    primary_count,
    secondary_count,
    errors,
):
    condition = record["condition"]
    ambiguous_identifier = record.get(
        "ambiguous_identifier"
    )
    ambiguous_identifiers = record.get(
        "ambiguous_identifiers"
    )

    if condition == "redundant_cues":
        if primary_count != 1 or secondary_count != 1:
            errors.append(
                "redundant_cues requires both identifiers "
                "to be individually unique"
            )

        if ambiguous_identifier != "none":
            errors.append(
                "redundant_cues must store "
                "ambiguous_identifier='none'"
            )

        if ambiguous_identifiers != []:
            errors.append(
                "redundant_cues must store "
                "ambiguous_identifiers=[]"
            )

    elif condition == "single_cue_ambiguous":
        if ambiguous_identifier not in {
            primary,
            secondary,
        }:
            errors.append(
                "single_cue_ambiguous must store exactly "
                "one identifier attribute"
            )
            return

        if ambiguous_identifiers != [
            ambiguous_identifier
        ]:
            errors.append(
                "single_cue_ambiguous has inconsistent "
                "ambiguous_identifiers metadata"
            )

        expected_primary = (
            2
            if ambiguous_identifier == primary
            else 1
        )

        expected_secondary = (
            2
            if ambiguous_identifier == secondary
            else 1
        )

        if primary_count != expected_primary:
            errors.append(
                f"single_cue_ambiguous expected "
                f"{expected_primary} matches for '{primary}', "
                f"got {primary_count}"
            )

        if secondary_count != expected_secondary:
            errors.append(
                f"single_cue_ambiguous expected "
                f"{expected_secondary} matches for '{secondary}', "
                f"got {secondary_count}"
            )

    elif condition == "conjunctive_binding":
        if primary_count != 2 or secondary_count != 2:
            errors.append(
                "conjunctive_binding requires both individual "
                "identifiers to be ambiguous"
            )

        if ambiguous_identifier != "both":
            errors.append(
                "conjunctive_binding must store "
                "ambiguous_identifier='both'"
            )

        if ambiguous_identifiers != [
            primary,
            secondary,
        ]:
            errors.append(
                "conjunctive_binding must store both "
                "identifier attributes in ambiguous_identifiers"
            )

    else:
        errors.append(
            f"unknown condition: {condition}"
        )


def validate_counterfactual(
    record,
    cf_type,
    cf,
    target_id,
    queried_attribute,
    constraints,
    clean_answer,
    errors,
):
    expected_cf_type = (
        f"{queried_attribute}_swap"
    )

    if cf_type != expected_cf_type:
        errors.append(
            f"unexpected cf_type '{cf_type}', "
            f"expected '{expected_cf_type}'"
        )

    if cf.get("question") != record["question"]:
        errors.append(
            "counterfactual question differs "
            "from clean question"
        )

    if cf.get("changed_object_id") != target_id:
        errors.append(
            f"counterfactual changed_object_id="
            f"{cf.get('changed_object_id')} but "
            f"target_object_id={target_id}"
        )

    if cf.get(
        "changed_attribute"
    ) != queried_attribute:
        errors.append(
            f"counterfactual changed_attribute="
            f"{cf.get('changed_attribute')} but "
            f"queried_attribute={queried_attribute}"
        )

    cf_objects = cf.get("objects")

    if not isinstance(
        cf_objects,
        list,
    ):
        errors.append(
            "counterfactual objects missing or invalid"
        )
        return

    clean_objects = record["objects"]

    if len(cf_objects) != len(clean_objects):
        errors.append(
            "counterfactual object count changed: "
            f"{len(clean_objects)} -> {len(cf_objects)}"
        )
        return

    validate_geometry(
        cf_objects,
        "counterfactual",
        errors,
    )

    derived_cf_counts = queried_value_counts(
        cf_objects,
        queried_attribute,
    )

    if cf.get(
        "queried_value_counts"
    ) != derived_cf_counts:
        errors.append(
            "counterfactual queried_value_counts="
            f"{cf.get('queried_value_counts')}, "
            f"expected {derived_cf_counts}"
        )

    clean_by_id = {
        obj["object_id"]: obj
        for obj in clean_objects
    }

    cf_by_id = {
        obj["object_id"]: obj
        for obj in cf_objects
    }

    if set(clean_by_id) != set(cf_by_id):
        errors.append(
            "object IDs differ between clean "
            "and counterfactual"
        )
        return

    for object_id in sorted(clean_by_id):
        clean_obj = clean_by_id[object_id]
        cf_obj = cf_by_id[object_id]

        if clean_obj.get(
            "center"
        ) != cf_obj.get("center"):
            errors.append(
                f"object {object_id} changed position "
                "in counterfactual"
            )

        changed = semantic_changes(
            clean_obj,
            cf_obj,
        )

        if object_id == target_id:
            if changed != [
                queried_attribute
            ]:
                errors.append(
                    f"target object changed attributes "
                    f"{changed}; expected only "
                    f"['{queried_attribute}']"
                )
        elif changed:
            errors.append(
                f"distractor object {object_id} "
                f"changed attributes {changed}"
            )

    cf_matches = matching_objects(
        cf_objects,
        constraints,
    )

    if len(cf_matches) != 1:
        errors.append(
            "counterfactual query is ambiguous/missing: "
            f"{len(cf_matches)} matches"
        )
        return

    if cf_matches[0][
        "object_id"
    ] != target_id:
        errors.append(
            "counterfactual query no longer refers "
            "to the original target object"
        )

    derived_cf_answer = cf_matches[0][
        queried_attribute
    ]

    stored_cf_answer = cf.get("answer")

    if stored_cf_answer != derived_cf_answer:
        errors.append(
            "counterfactual answer mismatch: "
            f"stored={stored_cf_answer}, "
            f"derived={derived_cf_answer}"
        )

    if stored_cf_answer == clean_answer:
        errors.append(
            "counterfactual answer did not change"
        )


def validate_record(record):
    errors = []

    required_fields = [
        "sample_id",
        "family_id",
        "condition",
        "difficulty_level",
        "objects",
        "target_object",
        "target_object_id",
        "queried_attribute",
        "identifier_attributes",
        "ambiguous_identifier",
        "ambiguous_identifiers",
        "query_attributes",
        "question",
        "answer",
        "answer_options",
        "queried_value_frequency_role",
        "queried_value_counts",
        "identifier_match_counts",
        "counterfactuals",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(
                f"missing field: {field}"
            )

    if errors:
        return errors

    condition = record["condition"]

    if record[
        "difficulty_level"
    ] != DIFFICULTY_LEVEL.get(condition):
        errors.append(
            "wrong difficulty_level for condition"
        )

    objects = record["objects"]
    target = record["target_object"]
    target_id = record["target_object_id"]
    queried_attribute = record[
        "queried_attribute"
    ]
    constraints = record[
        "query_attributes"
    ]

    if len(objects) != 4:
        errors.append(
            f"expected 4 objects, got {len(objects)}"
        )

    if queried_attribute not in (
        IDENTIFIER_ATTRIBUTES
    ):
        errors.append(
            f"unknown queried_attribute: {queried_attribute}"
        )
        return errors

    primary, secondary = (
        IDENTIFIER_ATTRIBUTES[
            queried_attribute
        ]
    )

    expected_identifiers = [
        primary,
        secondary,
    ]

    if record[
        "identifier_attributes"
    ] != expected_identifiers:
        errors.append(
            "identifier_attributes="
            f"{record['identifier_attributes']} "
            f"but expected {expected_identifiers}"
        )

    expected_constraints = {
        attribute: target[attribute]
        for attribute in (
            primary,
            secondary,
        )
    }

    if constraints != expected_constraints:
        errors.append(
            f"query_attributes={constraints} "
            f"but expected {expected_constraints}"
        )

    target_matches_by_id = [
        obj
        for obj in objects
        if obj["object_id"] == target_id
    ]

    if len(target_matches_by_id) != 1:
        errors.append(
            f"target_object_id={target_id} appears "
            f"{len(target_matches_by_id)} times"
        )
        return errors

    target_from_objects = (
        target_matches_by_id[0]
    )

    for attribute in (
        "shape",
        "color",
        "size",
        "center",
        "bbox",
    ):
        if target.get(
            attribute
        ) != target_from_objects.get(attribute):
            errors.append(
                "target_object disagrees with "
                f"objects list on '{attribute}'"
            )

    clean_matches = matching_objects(
        objects,
        constraints,
    )

    if len(clean_matches) != 1:
        errors.append(
            "clean query is ambiguous/missing: "
            f"{len(clean_matches)} matches"
        )
    elif clean_matches[0][
        "object_id"
    ] != target_id:
        errors.append(
            "clean query does not refer "
            "to target_object_id"
        )

    derived_answer = target[
        queried_attribute
    ]

    if record[
        "answer"
    ] != derived_answer:
        errors.append(
            "clean answer mismatch: "
            f"stored={record['answer']}, "
            f"derived={derived_answer}"
        )

    expected_options = (
        ATTRIBUTE_VALUES[
            queried_attribute
        ]
    )

    if set(
        record["answer_options"]
    ) != expected_options:
        errors.append(
            "answer_options="
            f"{record['answer_options']} "
            f"but expected {sorted(expected_options)}"
        )

    validate_geometry(
        objects,
        "clean",
        errors,
    )

    derived_counts = queried_value_counts(
        objects,
        queried_attribute,
    )

    if record[
        "queried_value_counts"
    ] != derived_counts:
        errors.append(
            "queried_value_counts="
            f"{record['queried_value_counts']}, "
            f"expected {derived_counts}"
        )

    validate_frequency_profile(
        record,
        derived_counts,
        errors,
    )

    primary_count = sum(
        obj[primary] == target[primary]
        for obj in objects
    )

    secondary_count = sum(
        obj[secondary] == target[secondary]
        for obj in objects
    )

    expected_match_counts = {
        primary: primary_count,
        secondary: secondary_count,
    }

    if record[
        "identifier_match_counts"
    ] != expected_match_counts:
        errors.append(
            "identifier_match_counts="
            f"{record['identifier_match_counts']}, "
            f"expected {expected_match_counts}"
        )

    validate_ambiguity_metadata(
        record,
        primary,
        secondary,
        primary_count,
        secondary_count,
        errors,
    )

    cfs = record["counterfactuals"]

    if not isinstance(
        cfs,
        dict,
    ):
        errors.append(
            "counterfactuals must be a dictionary"
        )
        return errors

    if len(cfs) != 1:
        errors.append(
            "expected exactly 1 counterfactual, "
            f"got {len(cfs)}"
        )

    for cf_type, cf in cfs.items():
        validate_counterfactual(
            record,
            cf_type,
            cf,
            target_id,
            queried_attribute,
            constraints,
            record["answer"],
            errors,
        )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    records = load_records(args.data)
    all_errors = []

    answer_counter = Counter()
    queried_attribute_counter = Counter()
    ambiguity_counter = Counter()
    target_shape_counter = Counter()
    target_color_counter = Counter()
    target_size_counter = Counter()
    cf_type_counter = Counter()
    clean_answer_frequency_counter = Counter()
    frequency_role_counter = Counter()
    frequency_role_by_answer_counter = Counter()

    for record in records:
        errors = validate_record(record)

        if errors:
            all_errors.append(
                (
                    record.get(
                        "sample_id",
                        "<missing-id>",
                    ),
                    errors,
                )
            )

        answer_counter[
            record.get("answer")
        ] += 1

        queried_attribute_counter[
            record.get("queried_attribute")
        ] += 1

        ambiguity_counter[
            record.get(
                "ambiguous_identifier"
            )
        ] += 1

        frequency_role = record.get(
            "queried_value_frequency_role"
        )

        frequency_role_counter[
            frequency_role
        ] += 1

        frequency_role_by_answer_counter[
            (
                record.get("queried_attribute"),
                record.get("answer"),
                frequency_role,
            )
        ] += 1

        target = record.get(
            "target_object",
            {},
        )

        target_shape_counter[
            target.get("shape")
        ] += 1

        target_color_counter[
            target.get("color")
        ] += 1

        target_size_counter[
            target.get("size")
        ] += 1

        for cf_type in record.get(
            "counterfactuals",
            {},
        ):
            cf_type_counter[cf_type] += 1

        queried_attribute = record.get(
            "queried_attribute"
        )

        answer = record.get("answer")

        counts = record.get(
            "queried_value_counts",
            {},
        )

        if (
            queried_attribute is not None
            and answer is not None
        ):
            clean_answer_frequency_counter[
                (
                    queried_attribute,
                    counts.get(answer),
                )
            ] += 1

    print(
        f"Loaded {len(records)} records"
    )

    print(
        "Number of invalid records: "
        f"{len(all_errors)}"
    )

    if all_errors:
        print(
            "\nFirst validation errors:"
        )

        for sample_id, errors in all_errors[:20]:
            print(f"\n{sample_id}")

            for error in errors:
                print(f"  - {error}")

    print(
        "\nQueried attribute distribution:"
    )
    print(queried_attribute_counter)

    print(
        "\nAnswer distribution:"
    )
    print(answer_counter)

    print(
        "\nCounterfactual type distribution:"
    )
    print(cf_type_counter)

    print(
        "\nAmbiguous identifier distribution:"
    )
    print(ambiguity_counter)

    print(
        "\nQueried-value frequency role distribution:"
    )
    print(frequency_role_counter)

    print(
        "\nFrequency role distribution by queried attribute and answer:"
    )
    print(frequency_role_by_answer_counter)

    print(
        "\nClean answer-frequency distribution "
        "(observational only):"
    )
    print(clean_answer_frequency_counter)

    print(
        "\nTarget shape distribution:"
    )
    print(target_shape_counter)

    print(
        "\nTarget color distribution:"
    )
    print(target_color_counter)

    print(
        "\nTarget size distribution:"
    )
    print(target_size_counter)

    balance_errors = []
    grouped_role_counts = {}

    for (queried_attribute, answer, role), count in (
        frequency_role_by_answer_counter.items()
    ):
        grouped_role_counts.setdefault(
            (queried_attribute, answer),
            {}
        )[role] = count

    for (queried_attribute, answer), role_counts in grouped_role_counts.items():
        expected_roles = FREQUENCY_ROLES[queried_attribute]
        counts = [role_counts.get(role, 0) for role in expected_roles]

        if max(counts) - min(counts) > 1:
            balance_errors.append(
                f"frequency roles are not balanced for "
                f"{queried_attribute} answer={answer}: {role_counts}"
            )

    if balance_errors:
        print("\nFrequency-role balance errors:")
        for error in balance_errors:
            print(f"- {error}")

    if all_errors or balance_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
