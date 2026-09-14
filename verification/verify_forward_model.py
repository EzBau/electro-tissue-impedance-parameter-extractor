"""
Phase 1 forward-model verification (Task 7).

Hand-checks the Randles-CPE forward model against three cases:
  1. alpha = 1.0 (CPE degenerates to an ideal capacitor) -> reduces to a
     textbook parallel-RC circuit; compared against the standard closed-form
     parallel-RC formula computed independently of forward_model.py.
  2. alpha = 0.5 (ideal Warburg-like element) -> checked against the known
     closed-form limit Z_CPE(f) = 1/(Q*sqrt(j*omega)) and, in the Rct -> inf
     limit, the classic 45-degree-line Warburg phase behavior.
  3. One randomly sampled case from the degenerate slice -> plotted on a
     Nyquist diagram and checked by eye for a depressed semicircle shape
     (no crossovers, no negative-real-part artifacts, single arc).

Run: python verify_forward_model.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from forward_model import FREQ_GRID_HZ, randles_cpe_impedance  # noqa: E402

import generate_dataset as gen  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


def check(label: str, got: complex, expected: complex, tol: float = 1e-6) -> bool:
    err = abs(got - expected) / max(abs(expected), 1e-30)
    status = PASS if err < tol else FAIL
    print(f"    {label}: got={got:.6g}  expected={expected:.6g}  rel_err={err:.3e}  [{status}]")
    return status == PASS


# ---------------------------------------------------------------------------
# Case 1: alpha = 1.0 -> parallel RC circuit
# ---------------------------------------------------------------------------
def case_1_parallel_rc() -> bool:
    print("\n[Case 1] alpha = 1.0: Randles-CPE must reduce to standard parallel RC")
    Rs, Rct, Q, alpha = 2_000.0, 50_000.0, 2.0e-7, 1.0
    # At alpha = 1, Z_CPE(f) = 1/(Q * j*omega) i.e. a pure capacitor with
    # C = Q farads. Independent closed-form for a parallel RC in series
    # with Rs (standard textbook formula, computed without touching
    # forward_model.py):
    #   Z(f) = Rs + Rct / (1 + j*omega*Rct*C)
    C = Q  # at alpha=1, Q has units of Farads
    freqs_to_check = [10.0, 1_000.0, 50_000.0]
    all_ok = True
    for f in freqs_to_check:
        omega = 2 * math.pi * f
        expected = Rs + Rct / (1 + 1j * omega * Rct * C)
        got = complex(randles_cpe_impedance(f, Rs, Rct, Q, alpha))
        all_ok &= check(f"f={f:>8.1f} Hz", got, expected)

    # Sanity limits: at f->0, Z -> Rs+Rct (capacitor branch open, all
    # current through Rct). At f->inf, Z -> Rs (capacitor branch shorts).
    z_low = complex(randles_cpe_impedance(1e-3, Rs, Rct, Q, alpha))
    z_high = complex(randles_cpe_impedance(1e9, Rs, Rct, Q, alpha))
    all_ok &= check("f->0 limit (expect Rs+Rct)", z_low, Rs + Rct, tol=1e-3)
    all_ok &= check("f->inf limit (expect Rs)", z_high, Rs, tol=1e-3)
    return all_ok


# ---------------------------------------------------------------------------
# Case 2: alpha = 0.5 -> Warburg-like element
# ---------------------------------------------------------------------------
def case_2_warburg() -> bool:
    print("\n[Case 2] alpha = 0.5: CPE must reduce to Warburg-like closed form")
    Rs, Q = 5_000.0, 3.0e-8
    all_ok = True

    # Independent closed form for a bare CPE at alpha=0.5:
    #   Z_CPE(f) = 1 / (Q * sqrt(j*omega)) = 1/(Q*sqrt(omega)) * (1 - j)/sqrt(2)
    # i.e. equal real and (negative) imaginary parts -> phase angle exactly
    # -45 degrees at every frequency, independent of f. This is the
    # signature "Warburg" 45-degree line on a Nyquist plot.
    for f in [1.0, 1_000.0, 100_000.0]:
        omega = 2 * math.pi * f
        expected_zcpe = 1.0 / (Q * complex(0, omega) ** 0.5)
        from forward_model import z_cpe

        got_zcpe = complex(z_cpe(f, Q, 0.5))
        all_ok &= check(f"Z_CPE f={f:>8.1f} Hz", got_zcpe, expected_zcpe)
        phase_deg = math.degrees(math.atan2(got_zcpe.imag, got_zcpe.real))
        ok = abs(phase_deg - (-45.0)) < 1e-6
        print(f"      phase(Z_CPE) = {phase_deg:.6f} deg (expected exactly -45.0)  [{PASS if ok else FAIL}]")
        all_ok &= ok

    # With Rct -> infinity (open charge-transfer branch), the full Randles
    # circuit collapses to Rs + Z_CPE, so it should reproduce the same
    # -45 degree Warburg line offset by Rs on the real axis.
    Rct_effectively_inf = 1e18
    for f in [1.0, 10_000.0]:
        omega = 2 * math.pi * f
        expected = Rs + 1.0 / (Q * complex(0, omega) ** 0.5)
        got = complex(randles_cpe_impedance(f, Rs, Rct_effectively_inf, Q, 0.5))
        all_ok &= check(f"Rs+Z_CPE f={f:>8.1f} Hz (Rct->inf)", got, expected, tol=1e-4)
    return all_ok


# ---------------------------------------------------------------------------
# Case 3: random degenerate-slice sample -> Nyquist eyeball check
# ---------------------------------------------------------------------------
def case_3_degenerate_nyquist(out_dir: str) -> bool:
    print("\n[Case 3] Random degenerate-slice sample: Nyquist-plot shape check")
    rng = np.random.default_rng(4242)
    params = gen.sample_degenerate_params(rng, 1)
    Rs, Rct, Q, alpha = (params[k][0] for k in ("Rs", "Rct", "Q", "alpha"))
    print(f"    sampled params: Rs={Rs:.4g} ohm, Rct={Rct:.4g} ohm, Q={Q:.4g} S*s^a, alpha={alpha:.4f}")

    Z = randles_cpe_impedance(FREQ_GRID_HZ, Rs, Rct, Q, alpha)
    re, im = Z.real, -Z.imag  # Nyquist convention: plot -Im(Z) vs Re(Z)

    all_ok = True

    # Numeric shape checks (not just "looks right"):
    # (a) Re(Z) must be monotonically non-increasing as frequency increases
    #     (classic depressed-semicircle behavior: real part decreases from
    #     Rs+Rct at low f down toward Rs at high f).
    order = np.argsort(FREQ_GRID_HZ)
    re_sorted = re[order]
    monotonic = np.all(np.diff(re_sorted) <= 1e-6)
    print(f"    Re(Z) monotonically non-increasing with frequency: [{PASS if monotonic else FAIL}]")
    all_ok &= bool(monotonic)

    # (b) -Im(Z) must be >= 0 everywhere (capacitive/depressed arc sits
    #     above the real axis in this convention; no computational sign
    #     artifact flipping it below).
    non_negative_im = np.all(im >= -1e-9)
    print(f"    -Im(Z) >= 0 at all frequencies (no artifact): [{PASS if non_negative_im else FAIL}]")
    all_ok &= bool(non_negative_im)

    # (c) Endpoints should bracket the arc: high-frequency real part should
    #     approach Rs, low-frequency real part should approach Rs+Rct (for
    #     a band wide enough to reach both asymptotes; here we just check
    #     the ordering, which holds regardless of band width).
    endpoints_ok = re_sorted[0] >= re_sorted[-1]
    print(f"    Low-f Re(Z) >= high-f Re(Z) (arc points the right way): [{PASS if endpoints_ok else FAIL}]")
    all_ok &= bool(endpoints_ok)

    # (d) The arc must be a *single* depressed semicircle, not a
    #     computational artifact with multiple lobes: check Im(Z) rises
    #     then falls at most once (unimodal) as frequency sweeps low->high.
    im_sorted = im[order]
    d = np.diff(im_sorted)
    sign_changes = np.sum(np.diff(np.sign(d[d != 0])) != 0)
    unimodal = sign_changes <= 1
    print(f"    -Im(Z) is unimodal across the sweep (single arc, {sign_changes} sign change(s)): [{PASS if unimodal else FAIL}]")
    all_ok &= bool(unimodal)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(re, im, "o-")
        ax.set_xlabel("Re(Z) [ohm]")
        ax.set_ylabel("-Im(Z) [ohm]")
        ax.set_title(
            f"Degenerate-slice sample Nyquist plot\nRs={Rs:.3g}, Rct={Rct:.3g}, "
            f"Q={Q:.3g}, alpha={alpha:.3f}"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, "case3_degenerate_nyquist.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Nyquist plot saved to {plot_path} (visual confirmation: single depressed semicircle)")
    except ImportError:
        print("    matplotlib not available -- skipped plot, numeric shape checks above stand on their own")

    return all_ok


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {
        "Case 1 (alpha=1.0, parallel RC)": case_1_parallel_rc(),
        "Case 2 (alpha=0.5, Warburg)": case_2_warburg(),
        "Case 3 (degenerate slice, Nyquist shape)": case_3_degenerate_nyquist(out_dir),
    }
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        print(f"  {name}: {PASS if ok else FAIL}")
        all_pass &= ok
    print("=" * 60)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
