"""Property tests for Particle2Grid (Eq. 16-17), cns/midend/prob_match.py.

P is the method -- "probabilistic correspondence encoded" is the paper's title -- and
until now it had NO test at all. cnsv2_implementation_spec.md Sec. 11 ships a
companion `test_p2g.py` that was never ported; this is that gate, written against our
implementation.

The core test is a LITERAL Eq. 16 reference: loop every anchor explicitly, with no
floor/offset trick, and diff against the optimized scatter. Everything else checks a
specific numerical claim the spec makes, so a mismatch tells us whether the spec's
analysis actually describes our code.

Run: python tests/test_p2g.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cns.midend.prob_match import (ParticleToGrid, bspline_quadratic,
                                   patch_grid_coords)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def reference_p2g(S, H16, W16, K, eps=1e-6):
    """Literal Eq. 16/17. Anchors enumerated explicitly; no floor, no offset window.

    num[i,j] = sum_k S[i,k] * kappa((x_d^k - x_c^i - x_a^j)/g)
    den[i,j] = same with the all-ones indicator instead of S
    x_a^j    = origin + (j + 0.5) * g          (cell-CENTRED, spec Sec. 1.1)
    """
    B, N, _ = S.shape
    grid = patch_grid_coords(H16, W16, S.device)           # [N,2] as (x,y)
    f = grid[None, :, :] - grid[:, None, :]                # [N,N,2] = x_d - x_c
    gx, gy = 2.0 * W16 / K, 2.0 * H16 / K
    num = S.new_zeros(B, N, K * K)
    den = S.new_zeros(N, K * K)
    for ay in range(K):
        for ax in range(K):
            xa = -W16 + (ax + 0.5) * gx
            ya = -H16 + (ay + 0.5) * gy
            w = (bspline_quadratic((f[..., 0] - xa) / gx)
                 * bspline_quadratic((f[..., 1] - ya) / gy))     # [N,N]
            j = ay * K + ax
            num[:, :, j] = (S * w[None]).sum(-1)
            den[:, j] = w.sum(-1)
    return (num / (den[None] + eps)).reshape(B, H16, W16, K * K), den


def main():
    torch.manual_seed(0)
    dev = "cpu"

    # ---- 1. literal reference, small grid, exhaustive -----------------------
    H = W = 8
    K = 4                       # g = (4,4): integer, so anchors land on integers
    N = H * W
    p2g = ParticleToGrid(K=K)
    S = torch.rand(2, N, N)
    S = S / S.sum(-1, keepdim=True)                # rows are distributions (Eq. 14)
    got = p2g(S, H, W)
    want, den_ref = reference_p2g(S, H, W, K)
    err = float((got - want).abs().max())
    check("matches literal Eq. 16 reference (8x8, K=4)", err < 1e-5, f"max|diff| {err:.2e}")

    # ---- 2. the {-1,0,1} offset window IS the full kernel support -----------
    # prob_match.py claims o in {-2,2} contributes exactly zero. The reference above
    # enumerates ALL anchors, so test 1 passing already proves it -- state it.
    check("offset window {-1,0,1} loses nothing", err < 1e-5,
          "reference enumerates all K^2 anchors and agrees")

    # ---- 3. denominator is CONTENT-independent ------------------------------
    S2 = torch.rand(1, N, N); S2 = S2 / S2.sum(-1, keepdim=True)
    _, den_ref2 = reference_p2g(S2, H, W, K)
    check("denominator independent of S", torch.allclose(den_ref, den_ref2, atol=1e-6))

    # ---- 4. denominator VARIES per patch (spec: P not translation-equivariant)
    live = (den_ref > 1e-8).sum(1)                       # live anchors per patch
    spread = float(den_ref[den_ref > 1e-8].max() / den_ref[den_ref > 1e-8].min())
    check("denominator varies by patch position", int(live.min()) != int(live.max()),
          f"live anchors {int(live.min())}..{int(live.max())}, value spread {spread:.0f}x")

    # ---- 5. structural sparsity, AT THE CONFIG THE SPEC MEASURES ------------
    # Spec Sec. 3.4: "58.7% of P's entries are structurally zero, and the live-anchor
    # count varies with patch position (81 for a corner patch, 100 for a centre patch,
    # up to 121)". Those numbers are for the PRODUCTION grid (32x32, K=16), not the
    # 8x8/K=4 grid above -- checking them on the small grid was my own error. The
    # denominator needs no S, so read it straight off the plan.
    big = ParticleToGrid(K=16)
    big._build_plan(32, 32, torch.device(dev))
    den_big = big._plan[(32, 32, torch.device(dev))][4][0]        # [N, K*K]
    zero_frac = float((den_big <= 1e-8).float().mean())
    check("58.7% of P structurally zero at 32x32/K=16 (spec Sec. 3.4)",
          abs(zero_frac - 0.587) < 0.02, f"measured {100*zero_frac:.1f}%")
    live_big = (den_big > 1e-8).sum(1)
    corner = int(live_big[0])                                     # patch (0,0)
    centre = int(live_big[16 * 32 + 16])                          # patch (16,16)
    check("live-anchor count 81 corner / 100 centre / <=121 max (spec Sec. 3.4)",
          corner == 81 and centre == 100 and int(live_big.max()) <= 121,
          f"corner {corner}, centre {centre}, max {int(live_big.max())}")

    # ---- 6. no NaN/Inf where the denominator is zero ------------------------
    check("no NaN/Inf from empty anchors",
          bool(torch.isfinite(got).all()), "eps guard in the num/den divide")

    # ---- 7. P's peak is NOT at the true displacement (spec Sec. 3.4) --------
    # Do not debug by checking where P peaks. Multiply den back out and check the
    # SPLAT instead -- that is the quantity with a meaningful argmax.
    one = torch.zeros(1, N, N)
    i0 = (H // 2) * W + (W // 2)
    k0 = (H // 2 + 1) * W + (W // 2 + 3)                 # a specific true match
    one[0, i0, k0] = 1.0
    P1, den1 = reference_p2g(one, H, W, K)
    flat = P1.reshape(-1, K * K)[i0]
    splat = flat * (den1[i0] + 1e-6)                     # undo the normalization
    fx, fy = (k0 % W) - (i0 % W), (k0 // W) - (i0 // W)
    gx, gy = 2.0 * W / K, 2.0 * H / K
    ax_t = int(round(((fx + W) / gx) - 0.5)); ay_t = int(round(((fy + H) / gy) - 0.5))
    j_true = ay_t * K + ax_t
    check("splat (num) peaks at the true displacement",
          int(splat.argmax()) == j_true,
          f"argmax {int(splat.argmax())} == true anchor {j_true}")
    check("P itself does NOT necessarily peak there (spec warning holds)",
          True, f"P argmax {int(flat.argmax())} vs true {j_true}"
                f"{' (differs, as the spec predicts)' if int(flat.argmax()) != j_true else ' (coincides here)'}")

    # ---- 8. batch independence --------------------------------------------
    solo = p2g(S[:1], H, W)
    check("batch element independence", torch.allclose(solo, got[:1], atol=1e-6))

    # ---- 9. gradient flows through P w.r.t. S (spec Sec. 11 asks for this) --
    Sg = S[:1].clone().requires_grad_(True)
    p2g(Sg, H, W).sum().backward()
    g = Sg.grad
    check("gradient flows to S", g is not None and bool(torch.isfinite(g).all())
          and float(g.abs().max()) > 0, f"max|dP/dS| {float(g.abs().max()):.3e}")

    # ---- 10. the production config, sampled rows ----------------------------
    H = W = 16
    K = 16                      # g = (2,2): integer -> valid per spec Sec. 1.1
    N = H * W
    p2g2 = ParticleToGrid(K=K)
    S3 = torch.rand(1, N, N); S3 = S3 / S3.sum(-1, keepdim=True)
    got3 = p2g2(S3, H, W)
    want3, _ = reference_p2g(S3, H, W, K)
    e3 = float((got3 - want3).abs().max())
    check("matches reference at K=16 (16x16 grid)", e3 < 1e-5, f"max|diff| {e3:.2e}")

    # ---- 11. spec Sec. 1.1 hard constraint: g must be EVEN -----------------
    # "For the efficient implementation to be exact, anchor spacing g = [2W16/K,
    # 2H16/K] must be even, so cell-centred anchors land on integers." Our code does
    # not check this. Demonstrate the failure mode rather than assert correctness.
    H = W = 12
    K = 8                       # g = (3,3): ODD -> anchors at half-integers
    N = H * W
    S4 = torch.rand(1, N, N); S4 = S4 / S4.sum(-1, keepdim=True)
    got4 = ParticleToGrid(K=K)(S4, H, W)
    want4, _ = reference_p2g(S4, H, W, K)
    e4 = float((got4 - want4).abs().max())
    # Our implementation floors per-sample rather than assuming integer anchors, so it
    # should STILL agree. If it ever does not, that is the spec's warning biting.
    check("odd-g grid still matches reference (12x12, K=8, g=3)", e4 < 1e-5,
          f"max|diff| {e4:.2e}; spec flags this config as the one that breaks "
          f"implementations that assume integer anchors")
    print(f"\n  NOTE: nothing in prob_match.py VALIDATES K against H16/W16. Spec Sec. 1.1 "
          f"restricts K to {{2,4,8,16,32}} for a 32x32 grid; an invalid K is silently "
          f"accepted.")

    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
