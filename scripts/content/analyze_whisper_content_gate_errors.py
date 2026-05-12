from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MUQATTAAT = {
    "الم", "الر", "المر", "المص", "كهيعص", "طه", "طسم", "طس",
    "يس", "ص", "حم", "عسق", "ق", "ن"
}


MUQATTAAT_SPOKEN_TO_SCRIPT = {
    "الفلامميم": "الم",
    "الفلاميم": "الم",
    "الافلاميم": "الم",
    "الافلامميم": "الم",
    "الاخلامميم": "الم",
    "الاخلاميم": "الم",
    "الف لام ميم": "الم",
    "اليف لام ميم": "الم",
    "الفلام ميم": "الم",
    "الاخلام ميم": "الم",

    "ياسين": "يس",
    "يا سين": "يس",
    "ياسن": "يس",

    "طسم": "طسم",
    "طاسيم": "طسم",
    "طاسينميم": "طسم",
    "طا سين ميم": "طسم",
    "باسيم": "طسم",
    "قاسيمميم": "طسم",
    "قاسيم ميم": "طسم",

    "عينسينقاف": "عسق",
    "عين سين قاف": "عسق",
    "عين سينقاف": "عسق",
    "عنسنقاف": "عسق",

    "حاميم": "حم",
    "حا ميم": "حم",

    "طاها": "طه",
    "طا ها": "طه",


    "الميم": "الم",
    "اللاميم": "الم",
    "الاميم": "الم",
    "طاسيميم": "طسم",
    "طاسين ميم": "طسم",
    "طا سيميم": "طسم",
    "قاف": "ق",
    "نون": "ن",
    "صاد": "ص",
}


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def compact(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def normalize_arabic(text: str) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_muqattaat_prediction(gold_norm: str, pred_norm: str) -> str:
    gold_c = compact(gold_norm)
    pred = str(pred_norm or "").strip()
    pred_c = compact(pred)

    if gold_c not in MUQATTAAT:
        return pred_norm

    if pred in MUQATTAAT_SPOKEN_TO_SCRIPT:
        return MUQATTAAT_SPOKEN_TO_SCRIPT[pred]
    if pred_c in MUQATTAAT_SPOKEN_TO_SCRIPT:
        return MUQATTAAT_SPOKEN_TO_SCRIPT[pred_c]

    return pred_norm


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (a[i - 1] != b[j - 1]),
            )
            prev = old
    return dp[n]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0
    ed = levenshtein(gold, pred)
    return max(0.0, 1.0 - ed / max(1, len(gold)))


def reciter_from_id(sample_id: str) -> str:
    m = re.match(r"^hf_quran_md_ayah_route_(.+?)_\d{3}_\d{3}_", str(sample_id))
    return m.group(1) if m else "unknown"


def surah_from_id(sample_id: str) -> str:
    m = re.match(r"^hf_quran_md_ayah_route_.+?_(\d{3})_(\d{3})_", str(sample_id))
    return m.group(1) if m else "unknown"


def length_bucket(n: int) -> str:
    if n <= 10:
        return "001_010"
    if n <= 20:
        return "011_020"
    if n <= 40:
        return "021_040"
    if n <= 60:
        return "041_060"
    return "061_plus"


def edit_char_stats(gold: str, pred: str) -> dict[str, Counter]:
    deletes = Counter()
    inserts = Counter()
    replaces = Counter()

    sm = SequenceMatcher(a=gold, b=pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            deletes.update(gold[i1:i2])
        elif tag == "insert":
            inserts.update(pred[j1:j2])
        elif tag == "replace":
            replaces.update(gold[i1:i2])
            inserts.update(pred[j1:j2])

    return {
        "deleted_gold_chars": deletes,
        "inserted_pred_chars": inserts,
        "replaced_gold_chars": replaces,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    total_gold = sum(int(r["gold_len_after"]) for r in rows)
    total_ed = sum(int(r["edit_distance_after"]) for r in rows)

    return {
        "samples": n,
        "exact_after_rate": sum(1 for r in rows if r["exact_after"]) / max(1, n),
        "avg_char_accuracy_after": sum(float(r["char_accuracy_after"]) for r in rows) / max(1, n),
        "cer_after": total_ed / max(1, total_gold),
        "avg_gold_len": total_gold / max(1, n),
        "avg_pred_len": sum(int(r["pred_len_after"]) for r in rows) / max(1, n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows_raw = [
        json.loads(x)
        for x in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    rows = []
    changed_by_muqattaat = []

    for r in rows_raw:
        sample_id = str(r.get("id") or r.get("sample_id") or "")
        gold_norm = normalize_arabic(r.get("gold_norm") or r.get("gold") or "")
        pred_norm_raw = normalize_arabic(r.get("pred_norm") or r.get("pred") or "")

        pred_norm_after = normalize_muqattaat_prediction(gold_norm, pred_norm_raw)

        gold_c = compact(gold_norm)
        pred_c_raw = compact(pred_norm_raw)
        pred_c_after = compact(pred_norm_after)

        ed_before = levenshtein(gold_c, pred_c_raw)
        ed_after = levenshtein(gold_c, pred_c_after)

        item = {
            "id": sample_id,
            "audio_path": r.get("audio_path", ""),
            "reciter": reciter_from_id(sample_id),
            "surah": surah_from_id(sample_id),
            "gold_norm": gold_norm,
            "pred_norm_raw": pred_norm_raw,
            "pred_norm_after": pred_norm_after,
            "gold_compact": gold_c,
            "pred_compact_raw": pred_c_raw,
            "pred_compact_after": pred_c_after,
            "exact_before": pred_c_raw == gold_c,
            "exact_after": pred_c_after == gold_c,
            "char_accuracy_before": char_accuracy(gold_c, pred_c_raw),
            "char_accuracy_after": char_accuracy(gold_c, pred_c_after),
            "edit_distance_before": ed_before,
            "edit_distance_after": ed_after,
            "gold_len_after": len(gold_c),
            "pred_len_after": len(pred_c_after),
            "len_delta_after": len(pred_c_after) - len(gold_c),
            "length_bucket": length_bucket(len(gold_c)),
            "muqattaat_gold": gold_c in MUQATTAAT,
            "muqattaat_changed": pred_c_raw != pred_c_after,
        }

        rows.append(item)
        if item["muqattaat_changed"]:
            changed_by_muqattaat.append(item)

    errors = [r for r in rows if not r["exact_after"]]
    near_misses = [
        r for r in errors
        if r["char_accuracy_after"] >= 0.95
    ]
    strong_near_misses = [
        r for r in errors
        if r["char_accuracy_after"] >= 0.98
    ]

    by_reciter = defaultdict(list)
    by_length = defaultdict(list)
    by_surah = defaultdict(list)

    deleted_chars = Counter()
    inserted_chars = Counter()
    replaced_chars = Counter()

    for r in rows:
        by_reciter[r["reciter"]].append(r)
        by_length[r["length_bucket"]].append(r)
        by_surah[r["surah"]].append(r)

    for r in errors:
        stats = edit_char_stats(r["gold_compact"], r["pred_compact_after"])
        deleted_chars.update(stats["deleted_gold_chars"])
        inserted_chars.update(stats["inserted_pred_chars"])
        replaced_chars.update(stats["replaced_gold_chars"])

    error_types = {
        "single_edit_errors": [r for r in errors if r["edit_distance_after"] == 1],
        "two_edit_errors": [r for r in errors if r["edit_distance_after"] == 2],
        "short_predictions_len_delta_le_minus2": [r for r in errors if r["len_delta_after"] <= -2],
        "long_predictions_len_delta_ge_2": [r for r in errors if r["len_delta_after"] >= 2],
        "muqattaat_remaining_errors": [r for r in errors if r["muqattaat_gold"]],
    }

    report = {
        "overall_after_muqattaat": summarize(rows),
        "errors_after_muqattaat": summarize(errors),
        "num_errors_after_muqattaat": len(errors),
        "num_near_misses_ge_095": len(near_misses),
        "num_strong_near_misses_ge_098": len(strong_near_misses),
        "num_muqattaat_changed": len(changed_by_muqattaat),
        "error_type_counts": {k: len(v) for k, v in error_types.items()},
        "by_length": {k: summarize(v) for k, v in sorted(by_length.items())},
        "by_reciter": {
            k: summarize(v)
            for k, v in sorted(
                by_reciter.items(),
                key=lambda kv: summarize(kv[1])["cer_after"],
                reverse=True,
            )
        },
        "by_surah_worst_30": {
            k: summarize(v)
            for k, v in sorted(
                by_surah.items(),
                key=lambda kv: summarize(kv[1])["cer_after"],
                reverse=True,
            )[:30]
        },
        "top_deleted_gold_chars": deleted_chars.most_common(30),
        "top_inserted_pred_chars": inserted_chars.most_common(30),
        "top_replaced_gold_chars": replaced_chars.most_common(30),
        "muqattaat_changed_examples": changed_by_muqattaat,
        "strong_near_misses_ge_098": sorted(
            strong_near_misses,
            key=lambda r: r["edit_distance_after"],
        )[:100],
        "near_misses_ge_095": sorted(
            near_misses,
            key=lambda r: (-r["char_accuracy_after"], r["edit_distance_after"]),
        )[:100],
        "worst_examples": sorted(
            errors,
            key=lambda r: (r["char_accuracy_after"], -r["edit_distance_after"]),
        )[:100],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Whisper content gate error report")
    lines.append("")
    lines.append("This report analyzes Whisper ASR errors after Quran normalization and muqattaat normalization.")
    lines.append("")
    lines.append("## Overall after muqattaat normalization")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k, v in report["overall_after_muqattaat"].items():
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines.append(f"| num_errors_after_muqattaat | {report['num_errors_after_muqattaat']} |")
    lines.append(f"| num_near_misses_ge_095 | {report['num_near_misses_ge_095']} |")
    lines.append(f"| num_strong_near_misses_ge_098 | {report['num_strong_near_misses_ge_098']} |")
    lines.append(f"| num_muqattaat_changed | {report['num_muqattaat_changed']} |")

    lines.append("")
    lines.append("## Error type counts")
    lines.append("")
    lines.append("| type | count |")
    lines.append("|---|---:|")
    for k, v in report["error_type_counts"].items():
        lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## By length")
    lines.append("")
    lines.append("| bucket | samples | exact_after | char_acc | CER | avg_gold_len |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bucket, s in report["by_length"].items():
        lines.append(
            f"| {bucket} | {s['samples']} | {s['exact_after_rate']:.3f} | "
            f"{s['avg_char_accuracy_after']:.3f} | {s['cer_after']:.3f} | {s['avg_gold_len']:.2f} |"
        )

    lines.append("")
    lines.append("## By reciter, worst CER first")
    lines.append("")
    lines.append("| reciter | samples | exact_after | char_acc | CER |")
    lines.append("|---|---:|---:|---:|---:|")
    for reciter, s in report["by_reciter"].items():
        lines.append(
            f"| {reciter} | {s['samples']} | {s['exact_after_rate']:.3f} | "
            f"{s['avg_char_accuracy_after']:.3f} | {s['cer_after']:.3f} |"
        )

    lines.append("")
    lines.append("## Character-level clues")
    lines.append("")
    lines.append(f"- top deleted gold chars: `{report['top_deleted_gold_chars']}`")
    lines.append(f"- top inserted pred chars: `{report['top_inserted_pred_chars']}`")
    lines.append(f"- top replaced gold chars: `{report['top_replaced_gold_chars']}`")

    lines.append("")
    lines.append("## Muqattaat changed examples")
    lines.append("")
    for r in report["muqattaat_changed_examples"][:50]:
        lines.append(f"### {r['id']}")
        lines.append(f"- gold: `{r['gold_norm']}`")
        lines.append(f"- pred_raw: `{r['pred_norm_raw']}`")
        lines.append(f"- pred_after: `{r['pred_norm_after']}`")
        lines.append(f"- exact_after: {r['exact_after']}")
        lines.append("")

    lines.append("")
    lines.append("## Strong near misses, char accuracy >= 0.98")
    lines.append("")
    for r in report["strong_near_misses_ge_098"][:60]:
        lines.append(f"### {r['id']}")
        lines.append(f"- reciter: `{r['reciter']}`")
        lines.append(f"- gold: `{r['gold_norm']}`")
        lines.append(f"- pred: `{r['pred_norm_after']}`")
        lines.append(f"- char_accuracy: {r['char_accuracy_after']:.4f}")
        lines.append(f"- edit_distance: {r['edit_distance_after']}")
        lines.append(f"- len_delta: {r['len_delta_after']}")
        lines.append("")

    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for r in report["worst_examples"][:60]:
        lines.append(f"### {r['id']}")
        lines.append(f"- reciter: `{r['reciter']}`")
        lines.append(f"- gold: `{r['gold_norm']}`")
        lines.append(f"- pred: `{r['pred_norm_after']}`")
        lines.append(f"- char_accuracy: {r['char_accuracy_after']:.4f}")
        lines.append(f"- edit_distance: {r['edit_distance_after']}")
        lines.append(f"- len_delta: {r['len_delta_after']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8")[:8000])


if __name__ == "__main__":
    main()
