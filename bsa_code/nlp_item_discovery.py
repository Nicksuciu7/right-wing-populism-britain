# nlp audit for candidate attitude questions

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from .config import load_config
from .logging_utils import log_event, setup_logging
from .output_layout import build_diagnostics_dir
from .rtf_parser import load_rtf_text

_VARIABLE_RE = re.compile(
    r"Variable =\s*(?P<var>\S+)\s+Variable label =\s*(?P<label>.*?)(?=\s+(?:This\s+variable\s+is|Value\s+label\s+information\s+for|Pos\.\s*=|Variable\s*=|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PROFILE_SEED_FAMILY = {
    "anti_elite": "populism",
    "inequality": "populism",
    "trust": "populism",
    "authoritarianism": "right",
    "welfare": "right",
    "immigration": "right",
}
_PROFILE_SEED_ALIAS = {
    "corruption": "anti_elite",
    "tradition": "authoritarianism",
    "voice": "anti_elite",
}


@dataclass(frozen=True)
class SeedMatch:
    category: str
    score: float
    keyword_hits: tuple[str, ...]
    phrase_hits: tuple[str, ...]
    strong_phrase_hits: tuple[str, ...]
    exclude_hits: tuple[str, ...]


def discover_question_candidates(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    build_dir = Path(config["paths"]["derived_dir"])
    diagnostics_dir = build_diagnostics_dir(build_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(build_dir / "logs" / "nlp_discovery.log")
    active_years = _active_years_from_config(config)
    active_year_set = set(active_years)

    dictionary_dir = Path(config["paths"]["dictionary_dir"])
    attitude_map_dir = Path(config["paths"]["attitude_map"]) / "years"
    seed_path = Path(config["paths"].get("seed_lexicon", attitude_map_dir.parent.parent / "rwp_seed_lexicon.json"))

    lexicon = json.loads(seed_path.read_text(encoding="utf-8"))
    dictionary_df = _build_dictionary_candidate_table(
        dictionary_dir,
        lexicon,
        allowed_years=active_year_set,
    )
    mapped_df = _build_mapped_item_table(
        attitude_map_dir,
        dictionary_df,
        lexicon,
        allowed_years=active_year_set,
    )
    coverage_df = _build_item_coverage_table(mapped_df)
    shared_candidates_df = _build_profile_shared_candidates(mapped_df, active_years)
    shared_summary_df = _build_profile_shared_construct_summary(shared_candidates_df)
    immigration_df = mapped_df[mapped_df["item_group"].isin(["immigration", "nativism"])].copy()
    year_summary_df = _build_year_summary_table(immigration_df)
    overlap_df = _build_pairwise_overlap_table(immigration_df)

    dictionary_df.to_csv(
        diagnostics_dir / "nlp_dictionary_seed_category_candidates.csv", index=False
    )
    coverage_df.to_csv(diagnostics_dir / "nlp_shared_question_coverage.csv", index=False)
    shared_candidates_df.to_csv(
        diagnostics_dir / "nlp_profile_shared_candidates.csv",
        index=False,
    )
    shared_summary_df.to_csv(
        diagnostics_dir / "nlp_profile_shared_construct_summary.csv",
        index=False,
    )
    year_summary_df.to_csv(
        diagnostics_dir / "nlp_immigration_nativism_year_summary.csv", index=False
    )
    overlap_df.to_csv(
        diagnostics_dir / "nlp_immigration_nativism_pairwise_overlap.csv", index=False
    )

    top_year_row = year_summary_df.iloc[0].to_dict() if not year_summary_df.empty else {}
    top_overlap_row = overlap_df.iloc[0].to_dict() if not overlap_df.empty else {}

    log_event(
        logger,
        "nlp_discovery_complete",
        active_years=active_years,
        dictionary_candidates=int(len(dictionary_df)),
        mapped_rows=int(len(mapped_df)),
        profile_shared_candidates=int(len(shared_candidates_df)),
        immigration_rows=int(len(immigration_df)),
        top_year=int(top_year_row.get("year", 0)) if top_year_row else None,
        top_year_shared_count=int(top_year_row.get("shared_item_count", 0)) if top_year_row else None,
        top_overlap_year_a=int(top_overlap_row.get("year_a", 0)) if top_overlap_row else None,
        top_overlap_year_b=int(top_overlap_row.get("year_b", 0)) if top_overlap_row else None,
        top_overlap_shared_count=(
            int(top_overlap_row.get("shared_item_count", 0)) if top_overlap_row else None
        ),
        outputs=[
            "nlp_dictionary_seed_category_candidates.csv",
            "nlp_shared_question_coverage.csv",
            "nlp_profile_shared_candidates.csv",
            "nlp_profile_shared_construct_summary.csv",
            "nlp_immigration_nativism_year_summary.csv",
            "nlp_immigration_nativism_pairwise_overlap.csv",
        ],
    )


def _active_years_from_config(config: dict) -> list[int]:
    years = sorted(
        {
            int(wave.get("year"))
            for wave in (config.get("waves") or [])
            if wave.get("year") is not None
        }
    )
    if not years:
        raise ValueError("nlp_discovery requires at least one configured wave year.")
    return years


def _build_dictionary_candidate_table(
    dictionary_dir: Path,
    lexicon: dict,
    *,
    allowed_years: set[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dictionary_path in sorted(dictionary_dir.glob("*.rtf")):
        year = _year_from_dictionary_name(dictionary_path)
        if year is None:
            continue
        if allowed_years is not None and year not in allowed_years:
            continue
        for variable, label in _extract_variable_labels(dictionary_path):
            best_match, all_matches = _score_label(label, lexicon)
            for match in all_matches:
                rows.append(
                    {
                        "year": year,
                        "seed_category": match.category,
                        "variable": variable,
                        "label": label,
                        "seed_score": match.score,
                        "keyword_hits": "; ".join(match.keyword_hits),
                        "phrase_hits": "; ".join(match.phrase_hits),
                        "strong_phrase_hits": "; ".join(match.strong_phrase_hits),
                        "exclude_hits": "; ".join(match.exclude_hits),
                        "best_seed_category": best_match.category if best_match else None,
                        "best_seed_score": best_match.score if best_match else None,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "seed_category",
                "rank_within_year_category",
                "variable",
                "label",
                "seed_score",
                "keyword_hits",
                "phrase_hits",
                "strong_phrase_hits",
                "exclude_hits",
                "best_seed_category",
                "best_seed_score",
            ]
        )

    df = pd.DataFrame(rows)
    df = df[df["seed_score"] > 0].copy()
    df = df.sort_values(
        ["year", "seed_category", "seed_score", "best_seed_score", "variable"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    df["rank_within_year_category"] = (
        df.groupby(["year", "seed_category"]).cumcount() + 1
    )
    ordered_cols = [
        "year",
        "seed_category",
        "rank_within_year_category",
        "variable",
        "label",
        "seed_score",
        "keyword_hits",
        "phrase_hits",
        "strong_phrase_hits",
        "exclude_hits",
        "best_seed_category",
        "best_seed_score",
    ]
    return df.loc[:, ordered_cols]


def _build_mapped_item_table(
    attitude_map_dir: Path,
    dictionary_df: pd.DataFrame,
    lexicon: dict,
    *,
    allowed_years: set[int] | None = None,
) -> pd.DataFrame:
    dict_best = dictionary_df.sort_values(
        ["year", "variable", "best_seed_score", "seed_score"],
        ascending=[True, True, False, False],
    ).drop_duplicates(["year", "variable"])

    rows: list[dict[str, object]] = []
    for year_path in sorted(attitude_map_dir.glob("*.json")):
        payload = json.loads(year_path.read_text(encoding="utf-8"))
        year = int(payload["year"])
        if allowed_years is not None and year not in allowed_years:
            continue
        for construct, construct_payload in (payload.get("constructs") or {}).items():
            items = (construct_payload or {}).get("items") or {}
            for item, spec in items.items():
                variable = str((spec or {}).get("var") or "").strip()
                label = None
                best_category = None
                best_score = None
                keyword_hits = ""
                phrase_hits = ""
                strong_phrase_hits = ""
                exclude_hits = ""

                if variable:
                    match = dict_best[
                        (dict_best["year"] == year) & (dict_best["variable"].str.lower() == variable.lower())
                    ]
                    if not match.empty:
                        row = match.iloc[0]
                        label = row["label"]
                        best_category = row["best_seed_category"]
                        best_score = row["best_seed_score"]
                        keyword_hits = row["keyword_hits"]
                        phrase_hits = row["phrase_hits"]
                        strong_phrase_hits = row["strong_phrase_hits"]
                        exclude_hits = row["exclude_hits"]

                if label is None:
                    label = ""
                if best_category is None:
                    best_match, _ = _score_label(label, lexicon)
                    if best_match is not None:
                        best_category = best_match.category
                        best_score = best_match.score
                        keyword_hits = "; ".join(best_match.keyword_hits)
                        phrase_hits = "; ".join(best_match.phrase_hits)
                        strong_phrase_hits = "; ".join(best_match.strong_phrase_hits)
                        exclude_hits = "; ".join(best_match.exclude_hits)

                rows.append(
                    {
                        "year": year,
                        "construct": construct,
                        "item": item,
                        "item_group": _item_group(item),
                        "variable": variable,
                        "label": label,
                        "best_seed_category": best_category,
                        "best_seed_score": best_score,
                        "keyword_hits": keyword_hits,
                        "phrase_hits": phrase_hits,
                        "strong_phrase_hits": strong_phrase_hits,
                        "exclude_hits": exclude_hits,
                    }
                )
    return pd.DataFrame(rows)


def _build_profile_shared_candidates(
    mapped_df: pd.DataFrame,
    active_years: list[int],
) -> pd.DataFrame:
    columns = [
        "family",
        "construct",
        "item",
        "item_group",
        "primary_seed_category",
        "seed_categories",
        "year_count",
        "years_present",
        "variables_by_year",
        "labels_by_year",
        "max_seed_score",
    ]
    if mapped_df.empty or not active_years:
        return pd.DataFrame(columns=columns)

    required_years = sorted({int(year) for year in active_years})
    required_year_set = set(required_years)
    rows: list[dict[str, object]] = []
    for (construct, item), group in mapped_df.groupby(["construct", "item"], dropna=False):
        years_present = sorted({int(year) for year in group["year"].tolist()})
        if set(years_present) != required_year_set or len(years_present) != len(required_years):
            continue

        family = str(construct) if construct in {"populism", "right"} else None
        item_group = _item_group(str(item))
        allowed_seed_categories = {
            category
            for category, mapped_family in _PROFILE_SEED_FAMILY.items()
            if mapped_family == family
        }
        if not family or not allowed_seed_categories:
            continue

        category_rows: list[tuple[str, float]] = []
        for row in group.itertuples():
            category = str(row.best_seed_category).strip() if row.best_seed_category is not None else ""
            category = _PROFILE_SEED_ALIAS.get(category, category)
            if category == "identity" and item_group in {"immigration", "nativism"}:
                category = "immigration"
            if category not in allowed_seed_categories:
                continue
            score = float(row.best_seed_score) if pd.notna(row.best_seed_score) else 0.0
            category_rows.append((category, score))
        if not category_rows:
            fallback_category = _fallback_profile_seed_category(str(construct), str(item))
            if fallback_category not in allowed_seed_categories:
                continue
            category_rows.append((fallback_category, 0.0))

        counts: dict[str, int] = {}
        max_scores: dict[str, float] = {}
        for category, score in category_rows:
            counts[category] = counts.get(category, 0) + 1
            max_scores[category] = max(score, max_scores.get(category, float("-inf")))
        ordered_categories = sorted(
            counts,
            key=lambda category: (-counts[category], -max_scores.get(category, 0.0), category),
        )
        vars_by_year = [
            f"{int(row.year)}:{row.variable}"
            for row in group.sort_values("year").itertuples()
            if isinstance(row.variable, str) and row.variable
        ]
        labels_by_year = [
            f"{int(row.year)}:{row.label}"
            for row in group.sort_values("year").itertuples()
            if isinstance(row.label, str) and row.label
        ]
        rows.append(
            {
                "family": family,
                "construct": construct,
                "item": item,
                "item_group": item_group,
                "primary_seed_category": ordered_categories[0],
                "seed_categories": ";".join(ordered_categories),
                "year_count": int(len(years_present)),
                "years_present": ";".join(str(year) for year in years_present),
                "variables_by_year": "; ".join(vars_by_year),
                "labels_by_year": "; ".join(labels_by_year),
                "max_seed_score": float(max(max_scores.values())) if max_scores else None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["family", "primary_seed_category", "item"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _fallback_profile_seed_category(construct: str, item: str) -> str | None:
    if construct == "populism":
        if item in {"gov_trust", "mp_trust"}:
            return "trust"
        return "inequality"
    if construct == "right":
        if item.startswith("welfare_"):
            return "welfare"
        if item.startswith("immigration_") or item.startswith("nativism_"):
            return "immigration"
        return "authoritarianism"
    return None


def _build_profile_shared_construct_summary(shared_candidates_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["family", "seed_category", "item_count", "items"]
    if shared_candidates_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (family, seed_category), group in shared_candidates_df.groupby(
        ["family", "primary_seed_category"],
        dropna=False,
    ):
        items = sorted({str(item) for item in group["item"].tolist()})
        rows.append(
            {
                "family": family,
                "seed_category": seed_category,
                "item_count": int(len(items)),
                "items": ";".join(items),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family", "item_count", "seed_category"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _build_item_coverage_table(mapped_df: pd.DataFrame) -> pd.DataFrame:
    if mapped_df.empty:
        return pd.DataFrame(
            columns=[
                "construct",
                "item",
                "item_group",
                "year_count",
                "years_present",
                "variables_by_year",
                "labels_by_year",
                "best_seed_categories",
                "max_seed_score",
            ]
        )

    rows: list[dict[str, object]] = []
    for (construct, item, item_group), group in mapped_df.groupby(
        ["construct", "item", "item_group"], dropna=False
    ):
        group = group.sort_values("year")
        years = [str(int(year)) for year in group["year"].tolist()]
        var_bits = [
            f"{int(row.year)}:{row.variable}"
            for row in group.itertuples()
            if isinstance(row.variable, str) and row.variable
        ]
        label_bits = [
            f"{int(row.year)}:{row.label}"
            for row in group.itertuples()
            if isinstance(row.label, str) and row.label
        ]
        seed_categories = sorted(
            {
                str(value)
                for value in group["best_seed_category"].dropna().tolist()
                if str(value).strip()
            }
        )
        seed_scores = pd.to_numeric(group["best_seed_score"], errors="coerce")
        rows.append(
            {
                "construct": construct,
                "item": item,
                "item_group": item_group,
                "year_count": int(group["year"].nunique()),
                "years_present": ";".join(years),
                "variables_by_year": "; ".join(var_bits),
                "labels_by_year": "; ".join(label_bits),
                "best_seed_categories": ";".join(seed_categories),
                "max_seed_score": (
                    float(seed_scores.max()) if seed_scores.notna().any() else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["year_count", "construct", "item"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _build_year_summary_table(immigration_df: pd.DataFrame) -> pd.DataFrame:
    if immigration_df.empty:
        return pd.DataFrame(
            columns=["year", "item_count", "shared_item_count", "shared_items", "unique_items"]
        )

    coverage = immigration_df.groupby("item")["year"].nunique().to_dict()
    rows: list[dict[str, object]] = []
    for year, group in immigration_df.groupby("year"):
        items = sorted(set(group["item"].tolist()))
        shared = [item for item in items if coverage.get(item, 0) > 1]
        unique = [item for item in items if coverage.get(item, 0) == 1]
        rows.append(
            {
                "year": int(year),
                "item_count": int(len(items)),
                "shared_item_count": int(len(shared)),
                "shared_items": ";".join(shared),
                "unique_items": ";".join(unique),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["shared_item_count", "item_count", "year"], ascending=[False, False, True]
    ).reset_index(drop=True)


def _build_pairwise_overlap_table(immigration_df: pd.DataFrame) -> pd.DataFrame:
    if immigration_df.empty:
        return pd.DataFrame(columns=["year_a", "year_b", "shared_item_count", "shared_items"])

    per_year = {
        int(year): set(group["item"].tolist()) for year, group in immigration_df.groupby("year")
    }
    rows: list[dict[str, object]] = []
    for year_a, year_b in combinations(sorted(per_year), 2):
        shared = sorted(per_year[year_a] & per_year[year_b])
        if not shared:
            continue
        rows.append(
            {
                "year_a": int(year_a),
                "year_b": int(year_b),
                "shared_item_count": int(len(shared)),
                "shared_items": ";".join(shared),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["shared_item_count", "year_a", "year_b"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _extract_variable_labels(path: Path) -> list[tuple[str, str]]:
    text = load_rtf_text(path)
    rows: list[tuple[str, str]] = []
    for match in _VARIABLE_RE.finditer(text):
        variable = match.group("var").strip()
        label = _clean_label(match.group("label"))
        if variable and label:
            rows.append((variable, label))
    return rows


def _clean_label(value: str) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value)).strip()
    text = text.replace(" :", ":").replace(" ,", ",")
    return text


def _normalise_text(value: str) -> str:
    lowered = str(value).lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _score_label(label: str, lexicon: dict) -> tuple[SeedMatch | None, list[SeedMatch]]:
    text = _normalise_text(label)
    matches: list[SeedMatch] = []
    for category, payload in lexicon.items():
        keyword_hits = _find_hits(text, payload.get("keywords", []))
        phrase_hits = _find_hits(text, payload.get("phrases", []))
        strong_hits = _find_hits(text, payload.get("strong_phrases", []))
        exclude_hits = _find_hits(
            text,
            (payload.get("exclude_keywords", []) or []) + (payload.get("exclude_phrases", []) or []),
        )
        score = float(len(keyword_hits) + 2 * len(phrase_hits) + 3 * len(strong_hits))
        if exclude_hits:
            score = max(0.0, score - (2.0 * len(exclude_hits)))
        if score <= 0:
            continue
        matches.append(
            SeedMatch(
                category=str(category),
                score=score,
                keyword_hits=tuple(keyword_hits),
                phrase_hits=tuple(phrase_hits),
                strong_phrase_hits=tuple(strong_hits),
                exclude_hits=tuple(exclude_hits),
            )
        )
    matches.sort(
        key=lambda match: (
            -match.score,
            -len(match.strong_phrase_hits),
            -len(match.phrase_hits),
            -len(match.keyword_hits),
            match.category,
        )
    )
    best_match = matches[0] if matches else None
    return best_match, matches


def _find_hits(text: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    for term in terms or []:
        normalised = _normalise_text(term)
        if not normalised:
            continue
        if f" {normalised} " in f" {text} ":
            hits.append(normalised)
    return hits


def _item_group(item: str) -> str:
    if item.startswith("immigration_"):
        return "immigration"
    if item.startswith("nativism_"):
        return "nativism"
    return "other"


def _year_from_dictionary_name(path: Path) -> int | None:
    match = re.search(r"(\d{2})(?!.*\d)", path.stem)
    if match is None:
        return None
    suffix = int(match.group(1))
    return 2000 + suffix
