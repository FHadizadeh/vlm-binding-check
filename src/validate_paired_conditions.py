import argparse
import json
from pathlib import Path


EXPECTED_CONDITIONS = [
    "redundant_cues",
    "single_cue_ambiguous",
    "conjunctive_binding",
]

IDENTIFIER_ATTRIBUTES = {
    "size": ("shape", "color"),
    "color": ("shape", "size"),
    "shape": ("color", "size"),
}


def load_records(data_dir: Path):
    with (
        data_dir / "metadata.jsonl"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        records = [
            json.loads(line)
            for line in f
        ]

    return {
        record["family_id"]: record
        for record in records
    }


def semantic_attrs(obj):
    return {
        "shape": obj["shape"],
        "color": obj["color"],
        "size": obj["size"],
    }


def changed_semantic_attrs(a, b):
    return [
        attr
        for attr in (
            "shape",
            "color",
            "size",
        )
        if a[attr] != b[attr]
    ]


def queried_value_multiset(record):
    queried_attribute = record[
        "queried_attribute"
    ]

    return sorted(
        obj[queried_attribute]
        for obj in record["objects"]
    )


def validate_family(
    family_id,
    redundant,
    single,
    conjunctive,
):
    errors = []

    records = [
        redundant,
        single,
        conjunctive,
    ]

    for (
        record,
        expected_condition,
    ) in zip(
        records,
        EXPECTED_CONDITIONS,
    ):
        if (
            record["condition"]
            != expected_condition
        ):
            errors.append(
                "expected condition "
                f"{expected_condition}, got "
                f"{record['condition']}"
            )

    shared_fields = [
        "family_id",
        "queried_attribute",
        "question",
        "answer",
        "answer_options",
        "target_object_id",
        "query_attributes",
        "identifier_attributes",
        "queried_value_frequency_role",
        "queried_value_counts",
    ]

    for field in shared_fields:
        values = [
            record[field]
            for record in records
        ]

        if not all(
            value == values[0]
            for value in values[1:]
        ):
            errors.append(
                f"field '{field}' differs "
                "across conditions"
            )

    queried_attribute = redundant[
        "queried_attribute"
    ]

    primary, secondary = (
        IDENTIFIER_ATTRIBUTES[
            queried_attribute
        ]
    )

    if (
        redundant.get(
            "ambiguous_identifier"
        )
        != "none"
        or redundant.get(
            "ambiguous_identifiers"
        )
        != []
    ):
        errors.append(
            "redundant_cues ambiguity metadata "
            "is incorrect"
        )

    single_ambiguous = single.get(
        "ambiguous_identifier"
    )

    if single_ambiguous not in {
        primary,
        secondary,
    }:
        errors.append(
            "single_cue_ambiguous has invalid "
            "ambiguous_identifier"
        )

    if single.get(
        "ambiguous_identifiers"
    ) != [single_ambiguous]:
        errors.append(
            "single_cue_ambiguous has inconsistent "
            "ambiguous_identifiers"
        )

    if (
        conjunctive.get(
            "ambiguous_identifier"
        )
        != "both"
    ):
        errors.append(
            "conjunctive_binding must store "
            "ambiguous_identifier='both'"
        )

    if conjunctive.get(
        "ambiguous_identifiers"
    ) != [
        primary,
        secondary,
    ]:
        errors.append(
            "conjunctive_binding must store "
            "both identifier attributes in "
            "ambiguous_identifiers"
        )

    target_semantics = [
        semantic_attrs(
            record["target_object"]
        )
        for record in records
    ]

    target_centers = [
        record["target_object"]["center"]
        for record in records
    ]

    if not all(
        value == target_semantics[0]
        for value in target_semantics[1:]
    ):
        errors.append(
            "target semantic attributes "
            "differ across conditions"
        )

    if not all(
        value == target_centers[0]
        for value in target_centers[1:]
    ):
        errors.append(
            "target position differs "
            "across conditions"
        )

    positions = [
        [
            obj["center"]
            for obj in record["objects"]
        ]
        for record in records
    ]

    if not all(
        value == positions[0]
        for value in positions[1:]
    ):
        errors.append(
            "object positions differ "
            "across conditions"
        )

    queried_multisets = [
        queried_value_multiset(record)
        for record in records
    ]

    if not all(
        value == queried_multisets[0]
        for value in queried_multisets[1:]
    ):
        errors.append(
            "queried-attribute value multiset "
            "differs across paired conditions"
        )

    redundant_by_id = {
        obj["object_id"]: obj
        for obj in redundant["objects"]
    }

    single_by_id = {
        obj["object_id"]: obj
        for obj in single["objects"]
    }

    conjunctive_by_id = {
        obj["object_id"]: obj
        for obj in conjunctive["objects"]
    }

    if (
        set(redundant_by_id)
        != set(single_by_id)
        or set(redundant_by_id)
        != set(conjunctive_by_id)
    ):
        errors.append(
            "object IDs differ "
            "across conditions"
        )
        return errors

    for object_id in redundant_by_id:
        r_obj = redundant_by_id[
            object_id
        ]
        s_obj = single_by_id[
            object_id
        ]
        c_obj = conjunctive_by_id[
            object_id
        ]

        if object_id in (
            0,
            3,
        ):
            if (
                semantic_attrs(r_obj)
                != semantic_attrs(s_obj)
                or semantic_attrs(r_obj)
                != semantic_attrs(c_obj)
            ):
                errors.append(
                    f"object {object_id} should "
                    "be identical across all "
                    "three conditions"
                )

    r1 = redundant_by_id[1]
    s1 = single_by_id[1]
    c1 = conjunctive_by_id[1]

    r2 = redundant_by_id[2]
    s2 = single_by_id[2]
    c2 = conjunctive_by_id[2]

    if changed_semantic_attrs(
        r1,
        c1,
    ) != [primary]:
        errors.append(
            "object 1 should differ from "
            "redundant to conjunctive only "
            f"in '{primary}'"
        )

    if changed_semantic_attrs(
        r2,
        c2,
    ) != [secondary]:
        errors.append(
            "object 2 should differ from "
            "redundant to conjunctive only "
            f"in '{secondary}'"
        )

    if single_ambiguous == primary:
        if (
            semantic_attrs(s1)
            != semantic_attrs(c1)
        ):
            errors.append(
                "single condition should already "
                "contain the primary-match distractor"
            )

        if (
            semantic_attrs(s2)
            != semantic_attrs(r2)
        ):
            errors.append(
                "single condition should leave "
                "the secondary-match distractor "
                "unmodified"
            )

    elif single_ambiguous == secondary:
        if (
            semantic_attrs(s2)
            != semantic_attrs(c2)
        ):
            errors.append(
                "single condition should already "
                "contain the secondary-match distractor"
            )

        if (
            semantic_attrs(s1)
            != semantic_attrs(r1)
        ):
            errors.append(
                "single condition should leave "
                "the primary-match distractor "
                "unmodified"
            )

    cf_answers = []
    cf_targets = []
    cf_questions = []
    cf_queried_value_counts = []

    for record in records:
        cf_items = list(
            record[
                "counterfactuals"
            ].values()
        )

        if len(cf_items) != 1:
            errors.append(
                f"{record['condition']} does not "
                "have exactly one counterfactual"
            )
            continue

        cf = cf_items[0]

        cf_answers.append(
            cf["answer"]
        )

        cf_targets.append(
            semantic_attrs(
                cf["target_object"]
            )
        )

        cf_questions.append(
            cf["question"]
        )

        cf_queried_value_counts.append(
            cf.get(
                "queried_value_counts"
            )
        )

    if (
        len(cf_answers) == 3
        and not all(
            value == cf_answers[0]
            for value in cf_answers[1:]
        )
    ):
        errors.append(
            "counterfactual answers differ "
            "across paired conditions"
        )

    if (
        len(cf_targets) == 3
        and not all(
            value == cf_targets[0]
            for value in cf_targets[1:]
        )
    ):
        errors.append(
            "counterfactual target attributes "
            "differ across paired conditions"
        )

    if (
        len(cf_questions) == 3
        and not all(
            value == cf_questions[0]
            for value in cf_questions[1:]
        )
    ):
        errors.append(
            "counterfactual questions differ "
            "across paired conditions"
        )

    if (
        len(cf_queried_value_counts) == 3
        and not all(
            value
            == cf_queried_value_counts[0]
            for value
            in cf_queried_value_counts[1:]
        )
    ):
        errors.append(
            "counterfactual queried-value counts "
            "differ across paired conditions"
        )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--redundant",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--single",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--conjunctive",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    datasets = {
        "redundant_cues": load_records(
            args.redundant
        ),
        "single_cue_ambiguous": load_records(
            args.single
        ),
        "conjunctive_binding": load_records(
            args.conjunctive
        ),
    }

    family_sets = [
        set(records)
        for records in datasets.values()
    ]

    common = set.intersection(
        *family_sets
    )

    union = set.union(
        *family_sets
    )

    errors = []

    if not all(
        family_set == family_sets[0]
        for family_set
        in family_sets[1:]
    ):
        errors.append(
            "family ID sets differ across datasets: "
            f"common={len(common)}, "
            f"union={len(union)}"
        )

    family_errors = []

    for family_id in sorted(common):
        current = validate_family(
            family_id,
            datasets[
                "redundant_cues"
            ][family_id],
            datasets[
                "single_cue_ambiguous"
            ][family_id],
            datasets[
                "conjunctive_binding"
            ][family_id],
        )

        if current:
            family_errors.append(
                (
                    family_id,
                    current,
                )
            )

    print(
        f"Paired families checked: "
        f"{len(common)}"
    )

    print(
        "Families with pairing errors: "
        f"{len(family_errors)}"
    )

    if errors:
        for error in errors:
            print(f"- {error}")

    if family_errors:
        print(
            "\nFirst pairing errors:"
        )

        for (
            family_id,
            current,
        ) in family_errors[:20]:
            print(f"\n{family_id}")

            for error in current:
                print(f"  - {error}")

    if errors or family_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
