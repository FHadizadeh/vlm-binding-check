import re
from typing import List, Optional


DEFAULT_ANSWER_CANDIDATES = [
    "small",
    "large",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "square",
    "circle",
    "triangle",
]


def build_chat_prompt(question: str) -> str:
    return question.strip()


def normalize_answer(
    text: str,
    candidates: Optional[List[str]] = None,
) -> str:
    if candidates is None:
        candidates = DEFAULT_ANSWER_CANDIDATES

    normalized_candidates = [candidate.lower().strip() for candidate in candidates]

    t = text.lower().strip()
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if t in normalized_candidates:
        return t

    words = t.split()

    for candidate in normalized_candidates:
        if candidate in words:
            return candidate

    return t
