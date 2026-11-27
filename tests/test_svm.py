"""Teste pentru clasa SVM (Pegasos, one-vs-rest) din src/svm/svm.py.

Fiecare test folosește un set de date mic, construit de mână, pentru care
răspunsul corect este evident. Rulare (din rădăcina proiectului):

    python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Facem clasa importabilă fără a modifica structura proiectului.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "svm"))
from svm import SVM  # noqa: E402


# ---------------------------------------------------------------------------
# Seturi de date mici, făcute de mână
# ---------------------------------------------------------------------------

def two_clusters_2d():
    """Două grupuri de puncte, clar separabile printr-o dreaptă (x + y = 0)."""
    X = np.array([
        [3.0, 3.0], [4.0, 3.0], [3.0, 4.0], [4.0, 4.0],          # clasa +1
        [-3.0, -3.0], [-4.0, -3.0], [-3.0, -4.0], [-4.0, -4.0],  # clasa -1
    ])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    return X, y


def three_clusters_2d():
    """Trei grupuri așezate la 120° unul de altul, în jurul originii."""
    centers = np.array([[5.0, 0.0], [-2.5, 4.33], [-2.5, -4.33]])
    offsets = np.array([[0.0, 0.0], [0.3, 0.3], [-0.3, 0.3], [0.3, -0.3], [-0.3, -0.3]])
    X = np.vstack([c + offsets for c in centers])
    y = np.repeat([0, 1, 2], len(offsets))
    return X, y


# ---------------------------------------------------------------------------
# Caz binar, separabil liniar
# ---------------------------------------------------------------------------

def test_binary_separable_classifies_training_points_correctly():
    X, y = two_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_binary_separable_classifies_new_points_on_the_right_side():
    X, y = two_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)
    new_points = np.array([[5.0, 5.0], [-5.0, -5.0]])
    assert np.array_equal(model.predict(new_points), [1, -1])


def test_fit_returns_self_so_calls_can_be_chained():
    X, y = two_clusters_2d()
    model = SVM(C=1.0)
    assert model.fit(X, y) is model


# ---------------------------------------------------------------------------
# Caz cu 3 clase (one-vs-rest)
# ---------------------------------------------------------------------------

def test_three_classes_classifies_training_points_correctly():
    X, y = three_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_three_classes_learned_attributes_have_expected_shapes():
    X, y = three_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)

    assert np.array_equal(model.classes_, [0, 1, 2])
    assert model.W_.shape == (3, 2)   # un vector de ponderi per clasă


def test_decision_function_gives_one_score_per_class():
    X, y = three_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(X), 3)


def test_predict_picks_the_class_with_the_highest_score():
    X, y = three_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X, y)
    scores = model.decision_function(X)
    assert np.array_equal(model.predict(X), model.classes_[scores.argmax(axis=1)])


# ---------------------------------------------------------------------------
# Etichete de tip text
# ---------------------------------------------------------------------------

def test_string_labels_are_returned_as_strings():
    X, y_numeric = two_clusters_2d()
    y = np.where(y_numeric == 1, "WALKING", "SITTING")

    model = SVM(C=1.0, seed=0).fit(X, y)

    assert set(model.classes_) == {"WALKING", "SITTING"}
    assert np.array_equal(model.predict(X), y)


def test_accepts_plain_python_lists():
    X, y = two_clusters_2d()
    model = SVM(C=1.0, seed=0).fit(X.tolist(), y.tolist())
    assert np.array_equal(model.predict(X.tolist()), y)


# ---------------------------------------------------------------------------
# Reproductibilitate
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_model():
    X, y = three_clusters_2d()
    a = SVM(C=1.0, seed=42).fit(X, y)
    b = SVM(C=1.0, seed=42).fit(X, y)
    assert np.array_equal(a.W_, b.W_)


# ---------------------------------------------------------------------------
# Date invalide
# ---------------------------------------------------------------------------

def test_single_class_raises_value_error():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array(["WALKING", "WALKING", "WALKING"])
    with pytest.raises(ValueError):
        SVM(C=1.0).fit(X, y)
