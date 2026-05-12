import torch

from tajweed_assessment.models.routing.learned_router import (
    LearnedRoutingModule,
    logits_to_routing_plan,
    routing_label_names,
)


def test_learned_router_forward_shape():
    model = LearnedRoutingModule(input_dim=12, hidden_dim=8, num_outputs=3)
    x = torch.randn(4, 12)

    logits = model(x)

    assert logits.shape == (4, 3)


def test_routing_label_names():
    assert routing_label_names() == ["use_duration", "use_transition", "use_burst"]


def test_logits_to_routing_plan():
    logits = torch.tensor([
        [3.0, -3.0, -3.0],
        [3.0, 3.0, -3.0],
        [-3.0, -3.0, -3.0],
    ])

    plans = logits_to_routing_plan(logits)

    assert plans == [
        {"use_duration": True, "use_transition": False, "use_burst": False},
        {"use_duration": True, "use_transition": True, "use_burst": False},
        {"use_duration": False, "use_transition": False, "use_burst": False},
    ]
