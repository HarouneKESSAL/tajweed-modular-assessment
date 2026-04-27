from dataclasses import dataclass
from typing import Dict, List

BLANK = "<blank>"
PHONEMES = [BLANK, "a", "i", "u", "m", "n", "q", "k", "l", "b", "y"]
RULES = ["none", "madd", "ghunnah", "ikhfa", "idgham", "qalqalah"]
TRANSITION_RULES = ["none", "ikhfa", "idgham"]

phoneme_to_id: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}
rule_to_id: Dict[str, int] = {r: i for i, r in enumerate(RULES)}
transition_rule_to_id: Dict[str, int] = {r: i for i, r in enumerate(TRANSITION_RULES)}

id_to_phoneme: Dict[int, str] = {v: k for k, v in phoneme_to_id.items()}
id_to_rule: Dict[int, str] = {v: k for k, v in rule_to_id.items()}

BLANK_ID = phoneme_to_id[BLANK]
IGNORE_INDEX = -100

@dataclass
class RuleTarget:
    position: int
    rule: str
    detail: str = ""

def encode_phonemes(symbols: List[str]) -> List[int]:
    return [phoneme_to_id[s] for s in symbols]

def encode_rules(symbols: List[str]) -> List[int]:
    return [rule_to_id[s] for s in symbols]

def normalize_rule_name(symbol: str) -> str:
    symbol = str(symbol).strip().lower()
    if not symbol:
        return "none"
    if symbol in rule_to_id:
        return symbol
    if symbol.startswith("madd"):
        return "madd"
    if "ghunnah" in symbol:
        return "ghunnah"
    if "ikhfa" in symbol:
        return "ikhfa"
    if "idgham" in symbol or "idghaam" in symbol:
        return "idgham"
    if "qalqalah" in symbol:
        return "qalqalah"
    return "none"
