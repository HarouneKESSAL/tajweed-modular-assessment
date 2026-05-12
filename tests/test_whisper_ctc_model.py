import torch

from tajweed_assessment.models.content.whisper_ctc import (
    collapse_ctc_ids,
    ctc_greedy_decode,
)


def test_collapse_ctc_ids_removes_blanks_and_repeats():
    assert collapse_ctc_ids([0, 1, 1, 0, 2, 2, 0, 3], blank_id=0) == [1, 2, 3]


def test_ctc_greedy_decode_shape():
    log_probs = torch.full((1, 5, 4), -10.0)
    log_probs[0, 0, 0] = 0.0
    log_probs[0, 1, 1] = 0.0
    log_probs[0, 2, 1] = 0.0
    log_probs[0, 3, 2] = 0.0
    log_probs[0, 4, 0] = 0.0

    decoded = ctc_greedy_decode(log_probs, blank_id=0)

    assert decoded == [[1, 2]]