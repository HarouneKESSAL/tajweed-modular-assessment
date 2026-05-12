from tajweed_assessment.inference.transition_multilabel import labels_to_combo


def test_labels_to_combo():
    assert labels_to_combo([]) == "none"
    assert labels_to_combo(["ikhfa"]) == "ikhfa"
    assert labels_to_combo(["idgham"]) == "idgham"
    assert labels_to_combo(["ikhfa", "idgham"]) == "ikhfa+idgham"
    assert labels_to_combo(["idgham", "ikhfa"]) == "ikhfa+idgham"
