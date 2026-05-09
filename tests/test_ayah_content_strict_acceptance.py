from scripts.system.run_ayah_content_inference import (
    ayah_acceptance_verdict,
    ayah_quality_label,
)


def test_ayah_content_exact_is_accepted():
    quality = ayah_quality_label(char_acc=1.0, edit_dist=0, exact_match=True)
    verdict = ayah_acceptance_verdict(char_acc=1.0, edit_dist=0, exact_match=True)

    assert quality == "content_verified_exact"
    assert verdict == "accepted_exact"


def test_ayah_content_high_similarity_is_not_auto_accepted():
    quality = ayah_quality_label(char_acc=0.7931, edit_dist=6, exact_match=False)
    verdict = ayah_acceptance_verdict(char_acc=0.7931, edit_dist=6, exact_match=False)

    assert quality == "same_ayah_candidate_review_required"
    assert verdict == "not_accepted"


def test_ayah_content_near_exact_is_review_recommended():
    quality = ayah_quality_label(char_acc=0.985, edit_dist=1, exact_match=False)
    verdict = ayah_acceptance_verdict(char_acc=0.985, edit_dist=1, exact_match=False)

    assert quality == "content_verified_near_exact"
    assert verdict == "accepted_near_exact_review_recommended"


def test_ayah_content_95_percent_is_still_not_accepted():
    quality = ayah_quality_label(char_acc=0.95, edit_dist=2, exact_match=False)
    verdict = ayah_acceptance_verdict(char_acc=0.95, edit_dist=2, exact_match=False)

    assert quality == "almost_correct_review_required"
    assert verdict == "not_accepted"
