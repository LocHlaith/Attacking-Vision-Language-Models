import pytest

from avlm_or.model import load_reference_labels, resolve_reference_class


CATEGORIES = ["washbasin", "Persian cat", "sports car"]


def test_reference_class_name_override() -> None:
    labels = {"2.png": "persian cat"}
    assert resolve_reference_class("datasets/2.png", 0, CATEGORIES, labels) == 1
    assert resolve_reference_class("datasets/1.png", 2, CATEGORIES, labels) == 2


def test_reference_class_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown reference class"):
        resolve_reference_class("2.png", 0, CATEGORIES, {"2.png": "cat"})


def test_load_reference_labels() -> None:
    assert load_reference_labels("datasets/reference_labels.json") == {}
