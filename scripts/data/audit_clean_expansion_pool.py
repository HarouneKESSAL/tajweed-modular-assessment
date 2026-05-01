from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter


NOISY_LABELS = {"not_related_quran", "multiple_aya", "not_match_aya", "in_complete"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalized_text(row: dict) -> str:
    raw = (
        row.get("normalized_text")
        or row.get("aya_text_norm")
        or row.get("text")
        or row.get("chunk_text")
        or row.get("target_text")
        or ""
    )
    return " ".join(str(raw).split())


def compact_text(row: dict) -> str:
    return normalized_text(row).replace(" ", "")


def verse_id(row: dict) -> tuple[str | None, str | None]:
    return (
        row.get("quranjson_surah_number") or row.get("surah_name") or row.get("surah"),
        row.get("quranjson_verse_key") or row.get("verse_key"),
    )


def has_verse_id(row: dict) -> bool:
    surah, verse = verse_id(row)
    return bool(surah or verse)


def sample_id(row: dict) -> str:
    return str(row.get("sample_id") or row.get("parent_id") or row.get("id") or "")


def summarize_rows(rows: list[dict]) -> dict:
    verse_ids = {verse_id(row) for row in rows if has_verse_id(row)}
    texts = {compact_text(row) for row in rows if compact_text(row)}
    surahs = {verse_id(row)[0] for row in rows if verse_id(row)[0]}
    reciters = {row.get("reciter_id") for row in rows if row.get("reciter_id")}
    return {
        "rows": len(rows),
        "unique_verse_ids": len(verse_ids),
        "unique_texts": len(texts),
        "unique_surahs": len(surahs),
        "unique_reciters": len(reciters),
    }


def extract_duration_rules(rule_spans: list[dict]) -> list[str]:
    labels = []
    for span in rule_spans or []:
        rule = str(span.get("rule", "")).strip().lower()
        if rule == "ghunnah":
            labels.append("ghunnah")
        elif rule.startswith("madd"):
            labels.append("madd")
    return labels


def extract_transition_rules(rule_spans: list[dict]) -> list[str]:
    labels = []
    for span in rule_spans or []:
        rule = str(span.get("rule", "")).strip().lower()
        if "ikhfa" in rule:
            labels.append("ikhfa")
        elif "idgham" in rule or "idghaam" in rule:
            labels.append("idgham")
    return sorted(set(labels))


def has_qalqalah(rule_spans: list[dict]) -> bool:
    return any(str(span.get("rule", "")).strip().lower() == "qalqalah" for span in rule_spans or [])


def clean_pool(rows: list[dict]) -> tuple[list[dict], Counter]:
    skipped = Counter()
    clean = []
    for row in rows:
        if row.get("match_status") != "matched_unique":
            skipped[f"match_status:{row.get('match_status') or 'missing'}"] += 1
            continue
        if row.get("final_label") in NOISY_LABELS:
            skipped[f"label:{row.get('final_label')}"] += 1
            continue
        if not row.get("audio_path"):
            skipped["missing_audio_path"] += 1
            continue
        if not row.get("rule_spans"):
            skipped["missing_rule_spans"] += 1
            continue
        clean.append(row)
    return clean, skipped


def rule_candidate_summary(clean_rows: list[dict]) -> dict:
    duration_row_count = 0
    duration_rule_counts = Counter()
    transition_row_counts = Counter()
    transition_ambiguous = 0
    burst_counts = Counter()

    for row in clean_rows:
        duration_rules = extract_duration_rules(row.get("rule_spans", []))
        if duration_rules:
            duration_row_count += 1
            duration_rule_counts.update(duration_rules)

        transition_rules = extract_transition_rules(row.get("rule_spans", []))
        if len(transition_rules) > 1:
            transition_ambiguous += 1
        elif transition_rules:
            transition_row_counts[transition_rules[0]] += 1
        else:
            transition_row_counts["none"] += 1

        burst_counts["qalqalah" if has_qalqalah(row.get("rule_spans", [])) else "none"] += 1

    return {
        "duration": {
            "candidate_rows_with_duration_rules": duration_row_count,
            "rule_positions": dict(duration_rule_counts),
        },
        "transition": {
            "candidate_rows_by_label": dict(transition_row_counts),
            "ambiguous_multi_transition_rows": transition_ambiguous,
        },
        "burst": {
            "candidate_rows_by_label": dict(burst_counts),
        },
    }


def module_manifest_summary(path: Path) -> dict:
    rows = load_jsonl(path)
    label_counts = Counter()
    for row in rows:
        if "burst_label" in row:
            label_counts["qalqalah" if int(row.get("burst_label", 0)) == 1 else "none"] += 1
        elif row.get("canonical_rules"):
            label_counts[str((row.get("canonical_rules") or ["none"])[0])] += 1
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        **summarize_rows(rows),
        "label_counts": dict(label_counts),
    }


def build_report(args: argparse.Namespace) -> dict:
    raw_rows = load_jsonl(PROJECT_ROOT / args.input)
    clean_rows, skipped = clean_pool(raw_rows)

    official = {
        "duration": module_manifest_summary(PROJECT_ROOT / "data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl"),
        "transition": module_manifest_summary(PROJECT_ROOT / "data/manifests/retasy_transition_subset.jsonl"),
        "burst": module_manifest_summary(PROJECT_ROOT / "data/manifests/retasy_burst_subset.jsonl"),
        "content_chunks": module_manifest_summary(PROJECT_ROOT / "data/manifests/retasy_content_chunks.jsonl"),
    }

    official_ids = {
        name: {sample_id(row) for row in load_jsonl(PROJECT_ROOT / summary["path"]) if sample_id(row)}
        for name, summary in official.items()
    }
    clean_ids = {str(row.get("id")) for row in clean_rows if row.get("id")}

    content_alignment_artifacts = {
        "scaled250": {
            "manifest": "data/manifests/retasy_content_alignment_corpus_scaled250.jsonl",
            "summary": summarize_rows(load_jsonl(PROJECT_ROOT / "data/manifests/retasy_content_alignment_corpus_scaled250.jsonl")),
        },
        "correct200": {
            "manifest": "data/manifests/retasy_content_alignment_corpus_correct200.jsonl",
            "summary": summarize_rows(load_jsonl(PROJECT_ROOT / "data/manifests/retasy_content_alignment_corpus_correct200.jsonl")),
        },
    }

    expansion_room = {}
    for name, ids in official_ids.items():
        if name == "content_chunks":
            continue
        expansion_room[name] = {
            "official_parent_rows_in_clean_pool": len(ids & clean_ids),
            "clean_pool_rows_not_in_official_manifest": len(clean_ids - ids),
        }

    report = {
        "input": str((PROJECT_ROOT / args.input).relative_to(PROJECT_ROOT)),
        "clean_filter": {
            "match_status": "matched_unique",
            "excluded_final_labels": sorted(NOISY_LABELS),
        },
        "raw_summary": summarize_rows(raw_rows),
        "clean_summary": summarize_rows(clean_rows),
        "skipped": dict(skipped),
        "candidate_rule_summary": rule_candidate_summary(clean_rows),
        "official_manifest_summary": official,
        "expansion_room": expansion_room,
        "content_alignment_artifacts": content_alignment_artifacts,
        "recommendation": {
            "decision": "expand_with_gates_not_blind_training",
            "why": [
                "Transition and burst already cover most of the clean matched rule pool.",
                "Content has the largest need for more text diversity, but previous scaled alignment experiments did not beat the tuned chunked baseline.",
                "Duration is already strong, so expansion should be verse-held-out and promotion-gated before replacing the baseline.",
            ],
            "next_experiment": [
                "Create a stricter content expansion split from clean matched rows.",
                "Train content with a capped/curriculum mix so diverse aligned chunks do not overpower the stable baseline chunks.",
                "Compare only against the tuned chunked open baseline.",
                "Promote only if strict text-held-out exact and char accuracy improve.",
            ],
        },
    }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Clean Expansion Readiness",
        "",
        "## Clean Pool",
        "",
        f"- Raw rows: `{report['raw_summary']['rows']}`",
        f"- Clean matched rows: `{report['clean_summary']['rows']}`",
        f"- Clean unique verse IDs: `{report['clean_summary']['unique_verse_ids']}`",
        f"- Clean unique surahs: `{report['clean_summary']['unique_surahs']}`",
        "",
        "## Candidate Rule Counts",
        "",
        f"- Duration rows with duration rules: `{report['candidate_rule_summary']['duration']['candidate_rows_with_duration_rules']}`",
        f"- Duration rule positions: `{report['candidate_rule_summary']['duration']['rule_positions']}`",
        f"- Transition rows by label: `{report['candidate_rule_summary']['transition']['candidate_rows_by_label']}`",
        f"- Transition ambiguous rows skipped by current builder: `{report['candidate_rule_summary']['transition']['ambiguous_multi_transition_rows']}`",
        f"- Burst rows by label: `{report['candidate_rule_summary']['burst']['candidate_rows_by_label']}`",
        "",
        "## Official Manifests",
        "",
        "| Module | Rows | Unique Verse IDs | Unique Texts | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, summary in report["official_manifest_summary"].items():
        notes = summary.get("label_counts") or {}
        lines.append(
            f"| {name} | `{summary['rows']}` | `{summary['unique_verse_ids']}` | `{summary['unique_texts']}` | `{notes}` |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Decision: `{report['recommendation']['decision']}`",
            "",
        ]
    )
    for item in report["recommendation"]["why"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Next experiment:")
    for item in report["recommendation"]["next_experiment"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_quranjson_train.jsonl")
    parser.add_argument("--output-json", default="data/analysis/clean_expansion_readiness.json")
    parser.add_argument("--output-md", default="data/analysis/clean_expansion_readiness.md")
    args = parser.parse_args()

    report = build_report(args)
    json_path = PROJECT_ROOT / args.output_json
    md_path = PROJECT_ROOT / args.output_md
    write_json(json_path, report)
    write_text(md_path, markdown(report))

    print(f"Clean matched rows : {report['clean_summary']['rows']}")
    print(f"Clean verse IDs    : {report['clean_summary']['unique_verse_ids']}")
    print(f"Duration candidates: {report['candidate_rule_summary']['duration']['candidate_rows_with_duration_rules']}")
    print(f"Transition labels  : {report['candidate_rule_summary']['transition']['candidate_rows_by_label']}")
    print(f"Burst labels       : {report['candidate_rule_summary']['burst']['candidate_rows_by_label']}")
    print(f"Saved JSON         : {json_path}")
    print(f"Saved Markdown     : {md_path}")


if __name__ == "__main__":
    main()
