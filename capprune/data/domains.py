"""MMLU subject -> domain maps.

``DOMAINS`` is the 8-way capability split CAP-Prune protects during pruning
(finer than MMLU's 4 official categories so that, e.g., mathematics, science
and history are tracked separately, as in the proposal).  ``CATEGORIES`` is the
official 4-way grouping, reported for comparability with the base paper
(Table 1 of Kodathala & Vunnam).
"""
from __future__ import annotations

DOMAINS: dict[str, list[str]] = {
    "mathematics": [
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics",
    ],
    "physical_sciences": [
        "astronomy", "college_chemistry", "college_physics", "conceptual_physics",
        "high_school_chemistry", "high_school_physics",
    ],
    "life_sciences_medicine": [
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "high_school_biology", "human_aging", "medical_genetics", "nutrition",
        "professional_medicine", "virology",
    ],
    "computing_engineering": [
        "college_computer_science", "computer_security", "electrical_engineering",
        "high_school_computer_science", "machine_learning",
    ],
    "history": [
        "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "prehistory",
    ],
    "law_ethics_philosophy": [
        "formal_logic", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "professional_law", "world_religions",
    ],
    "social_sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "business_general": [
        "business_ethics", "global_facts", "management", "marketing", "miscellaneous",
        "professional_accounting",
    ],
}

CATEGORIES: dict[str, list[str]] = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging",
        "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}


def _invert(m: dict[str, list[str]]) -> dict[str, str]:
    out = {}
    for k, subs in m.items():
        for s in subs:
            if s in out:
                raise ValueError(f"subject {s} mapped twice")
            out[s] = k
    return out


SUBJECT_TO_DOMAIN = _invert(DOMAINS)
SUBJECT_TO_CATEGORY = _invert(CATEGORIES)
assert set(SUBJECT_TO_DOMAIN) == set(SUBJECT_TO_CATEGORY) and len(SUBJECT_TO_DOMAIN) == 57


def domain_of(subject: str, custom: dict[str, list[str]] | None = None) -> str:
    if custom:
        for d, subs in custom.items():
            if subject in subs:
                return d
    return SUBJECT_TO_DOMAIN.get(subject, subject)
