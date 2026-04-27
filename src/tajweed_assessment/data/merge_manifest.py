from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

from .quranjson_rules import load_quranjson_rule_records


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


SURAH_NAME_TO_NUMBER = {
    "al_faatihah": 1, "al_fatiha": 1, "alfaatihah": 1, "alfatiha": 1,
    "al_baqarah": 2, "aal_imran": 3, "ali_imran": 3, "an_nisa": 4,
    "al_maidah": 5, "al_anam": 6, "al_araf": 7, "al_anfal": 8,
    "at_tawbah": 9, "al_tawbah": 9, "yunus": 10, "hud": 11, "yusuf": 12,
    "ar_rad": 13, "ibrahim": 14, "al_hijr": 15, "an_nahl": 16, "al_isra": 17,
    "bani_israil": 17, "al_kahf": 18, "maryam": 19, "ta_ha": 20, "taha": 20,
    "al_anbiya": 21, "al_hajj": 22, "al_muminun": 23, "an_nur": 24,
    "al_furqan": 25, "ash_shuara": 26, "an_naml": 27, "al_qasas": 28,
    "al_ankabut": 29, "ar_rum": 30, "luqman": 31, "as_sajdah": 32,
    "al_ahzab": 33, "saba": 34, "fatir": 35, "ya_sin": 36, "yasin": 36,
    "as_saffat": 37, "sad": 38, "az_zumar": 39, "ghafir": 40, "fussilat": 41,
    "ash_shura": 42, "az_zukhruf": 43, "ad_dukhan": 44, "al_jathiyah": 45,
    "al_ahqaf": 46, "muhammad": 47, "al_fath": 48, "al_hujurat": 49,
    "qaf": 50, "adh_dhariyat": 51, "at_tur": 52, "an_najm": 53, "al_qamar": 54,
    "ar_rahman": 55, "al_waqiah": 56, "al_hadid": 57, "al_mujadilah": 58,
    "al_hashr": 59, "al_mumtahanah": 60, "as_saff": 61, "al_jumuah": 62,
    "al_munafiqun": 63, "at_taghabun": 64, "at_talaq": 65, "at_tahrim": 66,
    "al_mulk": 67, "al_qalam": 68, "al_haqqah": 69, "al_maarij": 70,
    "nuh": 71, "al_jinn": 72, "al_muzzammil": 73, "al_muddaththir": 74,
    "al_qiyamah": 75, "al_insan": 76, "al_mursalat": 77, "an_naba": 78,
    "an_naziat": 79, "abasa": 80, "at_takwir": 81, "al_infitar": 82,
    "al_mutaffifin": 83, "al_inshiqaq": 84, "al_buruj": 85, "at_tariq": 86,
    "al_ala": 87, "al_ghashiyah": 88, "al_fajr": 89, "al_balad": 90,
    "ash_shams": 91, "al_layl": 92, "ad_duha": 93, "ash_sharh": 94,
    "al_inshirah": 94, "at_tin": 95, "al_alaq": 96, "al_qadr": 97,
    "al_bayyinah": 98, "az_zalzalah": 99, "al_adiyat": 100, "al_qariah": 101,
    "at_takathur": 102, "al_asr": 103, "al_humazah": 104, "al_fil": 105,
    "quraysh": 106, "al_maun": 107, "al_kawthar": 108, "al_kafirun": 109,
    "an_nasr": 110, "al_masad": 111, "al_ikhlas": 112, "al_falaq": 113,
    "an_nas": 114,
}


def surah_name_to_number(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    return SURAH_NAME_TO_NUMBER.get(_slug(name))


def merge_retasy_with_quranjson(
    *,
    retasy_manifest_path: str | Path,
    quranjson_repo_root: str | Path,
    out_jsonl: str | Path,
) -> Path:
    retasy_rows = load_jsonl(retasy_manifest_path)
    q_records = load_quranjson_rule_records(quranjson_repo_root)

    by_surah_and_text: Dict[tuple[int, str], List] = {}
    by_text_global: Dict[str, List] = {}

    for rec in q_records:
        if rec.verse_text_norm:
            by_text_global.setdefault(rec.verse_text_norm, []).append(rec)
            if rec.surah_number is not None:
                by_surah_and_text.setdefault((rec.surah_number, rec.verse_text_norm), []).append(rec)

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched_unique = 0
    matched_ambiguous = 0
    unmatched = 0

    with out_path.open("w", encoding="utf-8") as f:
        for base in retasy_rows:
            base = dict(base)

            hf_surah_number = surah_name_to_number(base.get("surah_name"))
            base["hf_surah_number"] = hf_surah_number

            aya_text_norm = base.get("aya_text_norm", "")
            candidates = []

            if hf_surah_number is not None and aya_text_norm:
                candidates = by_surah_and_text.get((hf_surah_number, aya_text_norm), [])

            if not candidates and aya_text_norm:
                candidates = by_text_global.get(aya_text_norm, [])

            if len(candidates) == 1:
                match = candidates[0]
                matched_unique += 1
                base["match_status"] = "matched_unique"
                base["match_count"] = 1
                base["quranjson_source_json_path"] = match.source_json_path
                base["quranjson_surah_number"] = match.surah_number
                base["quranjson_verse_key"] = match.verse_key
                base["quranjson_verse_index"] = match.verse_index
                base["quranjson_verse_text"] = match.verse_text
                base["rule_spans"] = [asdict(x) for x in match.rule_spans]
            elif len(candidates) > 1:
                matched_ambiguous += 1
                base["match_status"] = "matched_ambiguous"
                base["match_count"] = len(candidates)
                base["quranjson_source_json_path"] = None
                base["quranjson_surah_number"] = None
                base["quranjson_verse_key"] = None
                base["quranjson_verse_index"] = None
                base["quranjson_verse_text"] = None
                base["rule_spans"] = []
            else:
                unmatched += 1
                base["match_status"] = "unmatched"
                base["match_count"] = 0
                base["quranjson_source_json_path"] = None
                base["quranjson_surah_number"] = None
                base["quranjson_verse_key"] = None
                base["quranjson_verse_index"] = None
                base["quranjson_verse_text"] = None
                base["rule_spans"] = []

            f.write(json.dumps(base, ensure_ascii=False) + "\n")

    print(f"wrote: {out_path}")
    print(f"matched_unique   = {matched_unique}")
    print(f"matched_ambiguous= {matched_ambiguous}")
    print(f"unmatched        = {unmatched}")

    return out_path


if __name__ == "__main__":
    merge_retasy_with_quranjson(
        retasy_manifest_path="data/manifests/retasy_train.jsonl",
        quranjson_repo_root="external/quranjson-tajwid",
        out_jsonl="data/manifests/retasy_quranjson_train.jsonl",
    )