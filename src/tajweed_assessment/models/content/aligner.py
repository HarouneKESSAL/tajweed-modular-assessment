from typing import List
from tajweed_assessment.data.labels import id_to_phoneme

def align_sequences(reference: List[int], hypothesis: List[int]) -> list[dict]:
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = m, n
    ops: list[dict] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops.append({"type": "match" if cost == 0 else "substitution", "ref": reference[i - 1], "hyp": hypothesis[j - 1]})
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append({"type": "deletion", "ref": reference[i - 1], "hyp": None})
            i -= 1
            continue
        ops.append({"type": "insertion", "ref": None, "hyp": hypothesis[j - 1]})
        j -= 1

    return list(reversed(ops))

def human_readable_alignment(reference: List[int], hypothesis: List[int]) -> list[dict]:
    out = []
    for step in align_sequences(reference, hypothesis):
        out.append({
            "type": step["type"],
            "reference": None if step["ref"] is None else id_to_phoneme[step["ref"]],
            "hypothesis": None if step["hyp"] is None else id_to_phoneme[step["hyp"]],
        })
    return out
