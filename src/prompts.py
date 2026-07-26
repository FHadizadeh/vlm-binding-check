from typing import List


def build_chat_prompt(question: str) -> str:
    """Simple text-only prompt wrapper. For a real VLM chat template, use the model processor."""
    return question.strip()


def answer_candidates() -> List[str]:
    return ["small", "medium", "large"]


def normalize_answer(text: str) -> str:
    t = text.lower().strip().replace(".", "").replace(",", "")
    words = t.split()
    for cand in answer_candidates():
        if cand in words or t == cand:
            return cand
    for cand in answer_candidates():
        if cand in t:
            return cand
    return t
