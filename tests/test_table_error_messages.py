"""Missing values are refused where a formula consumes them, with the column
and the 1-based row named.

Table normalization preserves a missing cell (``NaN`` / ``None``) so that a
column the formula never references cannot block a fit (#2775); the refusal
lives one layer down, in the model-aware validator that knows which columns a
term consumes, and it must be as actionable there as it ever was here.
"""

from __future__ import annotations

import pytest


def _frame(bad: object) -> dict[str, list[object]]:
    return {
        "x": [0.1, 0.2, bad, 0.4, 0.5, 0.6, 0.7, 0.8],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }


def test_normalize_table_preserves_a_missing_numeric_cell() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("gamfit._rust")
    from gamfit._tables import normalize_table

    headers, table, _ = normalize_table(_frame(float("nan")))
    assert headers == ["x", "y"]
    assert table.shape == (8, 2)
    assert np.isnan(float(table[2][0]))


@pytest.mark.parametrize("bad", [float("nan"), None])
def test_fit_names_the_column_and_the_one_based_row(bad: object) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("gamfit._rust")
    import gamfit

    with pytest.raises(ValueError) as exc_info:
        gamfit.fit(_frame(bad), "y ~ x", family="gaussian")
    message = str(exc_info.value)
    assert "x" in message, f"expected column name 'x' in error: {message}"
    assert "row 3" in message, f"expected the 1-based row 3 in error: {message}"
