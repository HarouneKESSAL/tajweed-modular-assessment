import torch

from tajweed_assessment.models.transition.multilabel_transition_module import (
    TransitionMultiLabelModule,
    logits_to_multilabel_predictions,
    transition_rules_to_multihot,
)


def test_transition_rules_to_multihot():
    assert transition_rules_to_multihot([]) == [0.0, 0.0]
    assert transition_rules_to_multihot(["ikhfa"]) == [1.0, 0.0]
    assert transition_rules_to_multihot(["idgham"]) == [0.0, 1.0]
    assert transition_rules_to_multihot(["ikhfa", "idgham"]) == [1.0, 1.0]
    assert transition_rules_to_multihot(["ikhfa_shafawi", "idgham_ghunnah"]) == [1.0, 1.0]


def test_transition_multilabel_forward_shape():
    model = TransitionMultiLabelModule(
        mfcc_dim=39,
        ssl_dim=64,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        num_labels=2,
    )

    mfcc = torch.randn(3, 25, 39)
    ssl = torch.randn(3, 25, 64)
    lengths = torch.tensor([25, 20, 10])

    logits = model(mfcc, ssl, lengths)

    assert logits.shape == (3, 2)


def test_logits_to_multilabel_predictions():
    logits = torch.tensor(
        [
            [3.0, -3.0],
            [-3.0, 3.0],
            [3.0, 3.0],
            [-3.0, -3.0],
        ]
    )

    assert logits_to_multilabel_predictions(logits) == [
        ["ikhfa"],
        ["idgham"],
        ["ikhfa", "idgham"],
        [],
    ]
