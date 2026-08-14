import numpy as np

from decision_sparse_rl.metrics.exp7 import antithetic_asymmetry, margin_class, relative_error, transition_category


def test_margin_classes_are_frozen_at_boundaries():
    assert margin_class(0.00001, 0.00002, 0.001) == "ambiguous"
    assert margin_class(-0.0005, 0.00002, 0.001) == "near_boundary"
    assert margin_class(0.002, 0.00002, 0.001) == "interior"


def test_transition_categories_do_not_delete_modes():
    ref = ("a",)
    assert transition_category(ref, ref, ref).startswith("A")
    assert transition_category(ref, ("b",), ("b",)).startswith("B")
    assert transition_category(ref, ("b",), ("c",)).startswith("C")
    assert transition_category(ref, ref, ("c",)).startswith("D")


def test_vector_metrics():
    assert antithetic_asymmetry(np.array([1.0]), np.array([-1.0])) == 0.0
    assert relative_error(np.array([1.0]), np.array([1.0])) == 0.0
