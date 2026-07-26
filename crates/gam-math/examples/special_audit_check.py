#!/usr/bin/env python3
"""Arbitrary-precision reference check for the `special_audit_dump` example.

    cargo run -p gam-math --release --example special_audit_dump > /tmp/dump.tsv
    python3 crates/gam-math/examples/special_audit_check.py /tmp/dump.tsv [channel...]

Prints, per channel, the worst relative and worst absolute error against a
60-digit `mpmath` reference, with the argument each was attained at. Not wired
into CI; it is a hand-run audit tool and it needs `mpmath`.

References are chosen so that the reference itself is not the thing under test:
inverse functions are obtained by high-precision root-finding rather than by an
`erfinv` that loses digits near its endpoint, and the log-CDF derivatives use
the closed-form Mills recurrence rather than numerical differentiation.
"""

import sys
from collections import defaultdict

import mpmath as mp

mp.mp.dps = 80

# Below/above these the f64 channel has underflowed or overflowed and no
# relative statement is meaningful.
MIN_MAGNITUDE = mp.mpf("1e-300")
MAX_MAGNITUDE = mp.mpf("1e300")


def log_ncdf(x):
    """`log Phi(x)`, written so it never asks the working precision to resolve
    `1 - Phi(-x)` against `1`. `mp.log(mp.ncdf(x))` silently loses every digit
    once `Phi(-x)` falls below the mpf epsilon."""
    if x > 0:
        return mp.log1p(-mp.erfc(x / mp.sqrt(2)) / 2)
    return mp.log(mp.erfc(-x / mp.sqrt(2)) / 2)


def ref_bessel_ratio(x):
    return mp.besseli(1, x) / mp.besseli(0, x)


def ref_bessel_d2(x):
    r = ref_bessel_ratio(x)
    return -x + x * x * (1 - r * r)


def mills(x):
    """`lambda(x) = phi(x) / Phi(x)` at working precision."""
    return mp.npdf(x) / mp.ncdf(x)


def ref_logcdf_derivative(order):
    """Closed-form derivatives of `log Phi(x)` via the Mills recurrence.

    `f' = lambda`, and `lambda' = -lambda(x + lambda)`, so every higher
    derivative is a polynomial in `(x, lambda)`. At 60 digits the
    cancellations these expressions suffer in `f64` are irrelevant.
    """

    def inner(x):
        if order == 0:
            return log_ncdf(x)
        lam = mills(x)
        if order == 1:
            return lam
        if order == 2:
            return -lam * (x + lam)
        if order == 3:
            return lam * (x * x - 1 + 3 * x * lam + 2 * lam * lam)
        return -lam * (
            (x**3 - 3 * x)
            + (7 * x * x - 4) * lam
            + 12 * x * lam**2
            + 6 * lam**3
        )

    return inner


def ref_normal_quantile(p):
    """`Phi^-1(p)` by bisection-safeguarded root finding on `log Phi`."""
    target = mp.log(mp.mpf(p))
    guess = -mp.sqrt(2 * -target) if target < -20 else mp.mpf(0)
    return mp.findroot(lambda z: log_ncdf(z) - target, guess, solver="mnewton")


def ref_normal_quantile_from_log(log_p):
    target = mp.mpf(log_p)
    guess = -mp.sqrt(2 * -target) if target < -20 else mp.mpf(0)
    return mp.findroot(lambda z: log_ncdf(z) - target, guess, solver="mnewton")


CHANNELS = {
    "bessel_centered_log": lambda x: mp.log(mp.besseli(0, x)) - x,
    "bessel_ratio": ref_bessel_ratio,
    "bessel_d1": lambda x: x * (ref_bessel_ratio(x) - 1),
    "bessel_d2": ref_bessel_d2,
    "digamma": lambda x: mp.psi(0, x),
    "trigamma": lambda x: mp.psi(1, x),
    "tetragamma": lambda x: mp.psi(2, x),
    "pentagamma": lambda x: mp.psi(3, x),
    "normal_pdf": mp.npdf,
    "normal_cdf": mp.ncdf,
    "normal_logcdf": log_ncdf,
    "normal_logsf": lambda x: log_ncdf(-x),
    "probit_logcdf": log_ncdf,
    "probit_mills": mills,
    "erfcx": lambda x: mp.exp(x * x) * mp.erfc(x),
    "log1mexp": lambda a: mp.log(1 - mp.exp(-a)),
    "normal_quantile": ref_normal_quantile,
    "normal_quantile_from_log": ref_normal_quantile_from_log,
    "logcdf_d0": ref_logcdf_derivative(0),
    "logcdf_d1": ref_logcdf_derivative(1),
    "logcdf_d2": ref_logcdf_derivative(2),
    "logcdf_d3": ref_logcdf_derivative(3),
    "logcdf_d4": ref_logcdf_derivative(4),
}

# `mpmath`'s Bessel routines get impractically slow long before the f64 channel
# runs out of domain; cap where the asymptotic branch is already stationary.
BESSEL_CAP = 1e11
DOMAIN_CAP = {
    "bessel_centered_log": lambda x: x < BESSEL_CAP,
    "bessel_ratio": lambda x: x < BESSEL_CAP,
    "bessel_d1": lambda x: x < BESSEL_CAP,
    "bessel_d2": lambda x: x < BESSEL_CAP,
}


def legendre_refine(n, seed):
    """Newton-refine an f64 Gauss-Legendre node to working precision.

    Seeding from the f64 node (already correct to ~1e-15) makes this land on
    the intended root, which a fixed analytic guess plus a secant solver does
    not reliably do past `n = 5`.
    """
    z = mp.mpf(seed)
    if z == 0:
        derivative = n * (z * mp.legendre(n, z) - mp.legendre(n - 1, z)) / (z * z - 1)
        return z, 2 / ((1 - z * z) * derivative * derivative)
    for _ in range(200):
        value = mp.legendre(n, z)
        derivative = n * (z * value - mp.legendre(n - 1, z)) / (z * z - 1)
        step = value / derivative
        z = z - step
        if abs(step) < mp.mpf(10) ** (-mp.mp.dps + 5):
            break
    value = mp.legendre(n, z)
    derivative = n * (z * value - mp.legendre(n - 1, z)) / (z * z - 1)
    return z, 2 / ((1 - z * z) * derivative * derivative)


def errors(got, want):
    got = mp.mpf(got)
    absolute = abs(got - want)
    relative = absolute / abs(want) if want != 0 else absolute
    return relative, absolute


def beta_quantile_reference(a, b, p):
    """`Beta^-1(p; a, b)` by bisection in `ln x`.

    Newton and secant solvers (including `mp.findroot`'s defaults) do not find
    these roots. `I_x(a,b)` is flat to more than a hundred digits over most of
    the bracket, and for `Beta(0.1, 0.1)` at `p = 1e-8` the root sits at
    `8.9e-78`; no method that steps in `x` reaches it. Bisecting in `ln x`
    turns a search over hundreds of orders of magnitude into a search over a
    bounded interval, and `I` is monotone, so bisection cannot fail.

    Returns `0` when the root is below `f64`'s smallest subnormal, which is a
    real answer rather than a failure: it says the correctly rounded `f64`
    quantile is zero.
    """
    A, B, P = mp.mpf(a), mp.mpf(b), mp.mpf(p)
    # `exp(-1490)` is below `f64::MIN_POSITIVE`; `exp(0)` is the support ceiling.
    low, high = mp.mpf(-1490), mp.mpf(0)
    for _ in range(230):
        middle = (low + high) / 2
        value = mp.betainc(A, B, 0, mp.e ** middle, regularized=True)
        value = mp.mpf(value.real) if not isinstance(value, mp.mpf) else value
        if value < P:
            low = middle
        else:
            high = middle
    root = mp.e ** ((low + high) / 2)
    return mp.mpf(0) if root < mp.mpf("5e-324") else root


def main():
    path = sys.argv[1]
    only = set(sys.argv[2:]) or None
    worst_relative = defaultdict(lambda: (mp.mpf(0), None))
    worst_absolute = defaultdict(lambda: (mp.mpf(0), None))
    gl_rows = defaultdict(dict)
    binomial_rows = []
    beta_rows = []
    skipped = defaultdict(int)

    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            channel = parts[0]
            if only and channel not in only and not channel.startswith("gl_"):
                continue
            if channel in ("gl_node", "gl_weight"):
                if only and "gl" not in only:
                    continue
                n, index, value = int(parts[1]), int(parts[2]), float(parts[3])
                gl_rows[n].setdefault(channel, {})[index] = value
                continue
            if channel == "binomial":
                binomial_rows.append((int(parts[1]), int(parts[2]), float(parts[3])))
                continue
            if channel == "beta_shape":
                continue
            if channel == "beta_quantile":
                if only and channel not in only:
                    continue
                a, b, p, value = (float(t) for t in parts[1:5])
                want = beta_quantile_reference(a, b, p)
                if want == 0:
                    # The true quantile is below MIN_POSITIVE. Zero is then the
                    # only correct f64 answer, and anything positive is a floor.
                    if value != 0.0:
                        beta_rows.append((float("inf"), (a, b, p)))
                    continue
                if not (MIN_MAGNITUDE < abs(want) < MAX_MAGNITUDE):
                    skipped[channel] += 1
                    continue
                beta_rows.append((float(abs((mp.mpf(value) - want) / want)), (a, b, p)))
                continue
            reference = CHANNELS.get(channel)
            if reference is None:
                continue
            arg, value = float(parts[1]), float(parts[2])
            cap = DOMAIN_CAP.get(channel)
            if cap is not None and not cap(arg):
                continue
            try:
                want = reference(mp.mpf(arg))
            except (ValueError, ZeroDivisionError, OverflowError):
                skipped[channel] += 1
                continue
            if not mp.isfinite(want):
                skipped[channel] += 1
                continue
            if want != 0 and not (MIN_MAGNITUDE < abs(want) < MAX_MAGNITUDE):
                # The f64 channel has underflowed/overflowed; nothing to say.
                skipped[channel] += 1
                continue
            relative, absolute = errors(value, want)
            if relative > worst_relative[channel][0]:
                worst_relative[channel] = (relative, arg)
            if absolute > worst_absolute[channel][0]:
                worst_absolute[channel] = (absolute, arg)

    for channel in sorted(worst_relative):
        relative, rel_arg = worst_relative[channel]
        absolute, abs_arg = worst_absolute[channel]
        note = f"  (skipped {skipped[channel]})" if skipped[channel] else ""
        print(
            f"{channel:26s} rel={mp.nstr(relative, 3):>9s} @ {rel_arg!r:<24s}"
            f" abs={mp.nstr(absolute, 3):>9s} @ {abs_arg!r}{note}"
        )

    if beta_rows:
        beta_rows.sort(reverse=True)
        worst, where = beta_rows[0]
        median = sorted(row[0] for row in beta_rows)[len(beta_rows) // 2]
        print(
            f"{'beta_quantile':26s} rel={mp.nstr(worst, 3):>9s} @ "
            f"(a,b,p)={where!r}  median={median:.3e}  n={len(beta_rows)}"
        )
        for relative, where in beta_rows[1:4]:
            print(f"{'':26s}     {mp.nstr(relative, 3):>9s} @ (a,b,p)={where!r}")

    for n in sorted(gl_rows):
        nodes = gl_rows[n]["gl_node"]
        weights = gl_rows[n]["gl_weight"]
        node_error = mp.mpf(0)
        weight_error = mp.mpf(0)
        weight_arg = None
        for i in range(n):
            ref_node, ref_weight = legendre_refine(n, nodes[i])
            node_error = max(node_error, errors(nodes[i], ref_node)[1])
            this_weight = errors(weights[i], ref_weight)[0]
            if this_weight > weight_error:
                weight_error, weight_arg = this_weight, i
        weight_sum_error = abs(sum(weights.values()) - 2)
        print(
            f"gauss_legendre n={n:<5d} node_abs={mp.nstr(node_error, 3):>9s} "
            f"weight_rel={mp.nstr(weight_error, 3):>9s} @ i={weight_arg} "
            f"sum-2={weight_sum_error:.3e}"
        )

    bad = [
        (n, k, got)
        for n, k, got in binomial_rows
        if mp.mpf(got) != mp.binomial(n, k) and mp.binomial(n, k) < 2**53
    ]
    if binomial_rows:
        print(f"binomial exact-within-2^53 failures: {len(bad)}")
        for row in bad[:10]:
            print("   ", row, "want", mp.binomial(row[0], row[1]))


if __name__ == "__main__":
    main()
