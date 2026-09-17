"""The multinomial summary prints each class's coefficient norm from the right slice.

``MultinomialSavedModel::summary_text`` renders ``‖β_a‖₂`` for every active class
from the saved row-major ``(P, K-1)`` coefficient matrix, so class ``a`` is every
``m``-th entry starting at ``a``. A contiguous slice (the first ``P`` entries, then
the next ``P``) prints norms that mix classes. The summary is rendered in Rust
from the model bytes, so the check reads the published coefficient layout back
from the model and compares it with the norms the summary printed.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_multinomial_summary_norm_strides_correctly() -> None:
    rng = np.random.default_rng(42)
    n = 300
    x = rng.uniform(-1.0, 1.0, n)
    # Class A rises steeply with x, class B sits below the reference C with a
    # weak slope, so the two classes' coefficient columns differ in magnitude.
    eta = np.column_stack([2.0 * x, -1.0 + 0.2 * x, np.zeros(n)])
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    cls = np.array(["A", "B", "C"])[[rng.choice(3, p=row) for row in prob]]
    model = gamfit.fit(pd.DataFrame({"x": x, "y": cls}), "y ~ x", family="multinomial")

    flat = np.asarray(model._metadata["coefficients_flat"], dtype=float)
    m = len(model._metadata["class_levels"]) - 1
    p = flat.size // m
    assert m >= 2 and p * m == flat.size, (m, flat.size)
    strided = np.array([np.linalg.norm(flat[a::m]) for a in range(m)])
    contiguous = np.array([np.linalg.norm(flat[a * p : (a + 1) * p]) for a in range(m)])
    # The check has power only where the two layouts print different norms.
    assert np.max(np.abs(strided - contiguous) / np.maximum(strided, contiguous)) > 1.0e-2, (
        strided,
        contiguous,
    )

    printed = np.array(
        [float(value) for value in re.findall(r"‖β_a‖₂ = ([-+0-9.eE]+)", model.summary())]
    )
    assert printed.size == m, printed
    # Printed to four significant digits.
    np.testing.assert_allclose(printed, strided, rtol=5.0e-4, atol=0.0)
