# CNSv2 — Implementation Specification

Working spec for building CNSv2 (arXiv:2503.00132) from scratch. **No official code release exists** — verified by checking `hhcaz/CNSv2`, `CNS-v2`, `CNS_v2`; only v1's `hhcaz/CNS` is public.

Everything below is tagged:

- **[P]** specified in the paper
- **[V1]** specified in the CNS paper (arXiv:2309.09047), reasonable to inherit
- **[C]** chosen here, with rationale — the paper is silent
- **[!]** verified numerically during analysis, contradicts a naive reading of the paper

Target config throughout: **512×512 input, ViT-B/16, so H16 = W16 = 32, N16 = 1024.**

---

## 1. Hyperparameter decisions

| Symbol | Value | Tag | Rationale |
|---|---|---|---|
| Input resolution | 512×512 | [P] | stated in Table I baseline env |
| Backbone | AM-RADIOv2.5 ViT-B/16 | [P] | |
| `C` (token width) | 768 | [P] | fixed by ViT-B |
| RADIO frozen? | **frozen** | [C] | RoMa freezes DINOv2 and is cited approvingly; also halves memory and removes a large source of instability. Revisit only if matching quality is the measured bottleneck |
| Matching transformer depth | 4 blocks, alternating self/cross | [C] | LoFTR uses 4; "several" in the paper |
| Matching transformer width | 256 | [C] | project 768→256 first; full-width attention over 1024 tokens ×2 images is wasteful |
| Positional encoding | 2D axial RoPE | [P] | |
| **`K`** (anchor grid) | **16** → 256 channels | [C] | see §1.1 |
| `C_f` (fine CNN width) | 64 | [C] | keeps `K²+C_f` = 320 manageable |
| Controller depth | 4 self-attn blocks + 1 cross-attn | [C] | |
| Controller width | 256 | [C] | |
| λ (control gain) | 1.0 | [C] | see §1.2 |
| Δt | 0.02 s (50 Hz) | [C] | v1 uses 0.04 s; 50 Hz matches v2's 35 FPS claim better |
| Loss weighting | `L_dir + 0.1·L_norm` | [V1] | v2 doesn't state one; v1 does |
| Batch size | 16 | [P] | stated in §IV-A |
| Optimizer | AdamW, lr 1e-4, cosine | [C] | |
| `d*` | 1.0 m | [P] | baseline env |
| Canonical intrinsics | fx=fy=512, cx=cy=256 | [P] | |

### 1.1 Choosing K — there is a hard constraint

**[!]** For the efficient implementation in §3.4 to be exact, the anchor spacing `g = [2W16/K, 2H16/K]` must be **even** in both axes, so cell-centred anchors `x_a = −W16 + (j+0.5)·g` land on integers.

Verified: `12×16, K=8` (gy=3, odd) → anchors at half-integers → fast path is wrong by 2.4e-2. `16×24, K=8` (gy=4, gx=6) → exact to 7e-18.

For H16 = W16 = 32: valid `K ∈ {2, 4, 8, 16, 32}`.

| K | g | channels K² | displacement resolution |
|---|---|---|---|
| 8 | 8 | 64 | 8 patches = 128 px — too coarse |
| **16** | **4** | **256** | 4 patches = 64 px |
| 32 | 2 | 1024 | 2 patches = 32 px, but 1024 channels/token |

**K=16.** K=32 quadruples channel count for a factor-2 gain in displacement resolution, and `P` is already not the precision path (that's the fine CNN branch). Section IV-A's remark that `φ_S` bilinearly samples `S`'s last dim "from H16×W16 to K×K" implies K < 32, consistent.

### 1.2 λ consistency

λ is a free scalar that sets label magnitude. It must be **identical** between label generation and the simulator's integration step, otherwise DAgger rollouts systematically diverge from what the labels imply. Assert it in one place; do not let it appear as a literal twice.

---

## 2. Data structures

```python
Sample = {
    "I_c":  uint8  [H, W, 3],       # current image
    "I_d":  uint8  [H, W, 3],       # desired image
    "v_tilde_star": float32 [6],    # normalized expert velocity [nu/d*; omega]
}
```

That is the whole training tuple. `d*` is **not** stored — it is baked into the label and only reappears at deployment.

---

## 3. Module specifications

### 3.1 Backbone `[P]`

```python
F_c = radio(I_c).spatial     # [B, 1024, 768] -> reshape [B, 32, 32, 768]
F_d = radio(I_d).spatial
```

Load via `torch.hub` or their `hf_model`. **[!] Verify `model.patch_generator.cpe_mode is True` after loading.** `enable_cpe()` is a monkey-patch on a stock timm ViT; loading by any other route silently gives you the backbone *without* Cropped Position Embedding, which is what makes RADIO's absolute positional embeddings resolution-robust. No error is raised.

Discard the summary/CLS token. Do not use the teacher adaptors. L2-normalize channel-wise before correlating.

### 3.2 Matching transformer `[P]` / `[C]`

```python
proj:      768 -> 256                          # [C]
for 4 blocks:
    self-attn  (within each image, RoPE)
    cross-attn (Ic <-> Id, RoPE)
returns F_c_tilde, F_d_tilde                   # [B, 32, 32, 256]
```

Cross-attention makes `F̃c` depend on `Id`. **Desired-image features cannot be cached across frames** — this is per-frame work at deployment.

2D axial RoPE: split channels in half, apply RoPE with row index to one half and column index to the other.

### 3.3 Correlation `[P]`

```python
Fc_flat = F_c_tilde.reshape(B, 1024, 256)
S = softmax(Fc_flat @ Fd_flat.transpose(-1,-2) / sqrt(256), dim=-1)   # [B, 1024, 1024]
```

Row-wise softmax. Also compute the reverse direction for the confidence:

```python
S_dc = softmax(same_logits, dim=-2)            # column-wise
C = (S * S_dc).sum(-1)                         # [B, 1024]  dual-softmax confidence, Eq. 25
```

`C` is needed only by the hybrid controller (§3.8), but see §10.1.

### 3.4 Particle2Grid — efficient formulation `[!]`

The literal Eq. 16 is a scatter: 1024 particles × 1024 patches × 9 anchors ≈ 9.4M atomic adds per image. Unnecessary.

**Key identity (verified exact to 1e-17):** because particle positions `x_d^k` lie on a regular grid and the kernel argument is `(x_d^k − x_c^i − x_a^j)/g`, the weight depends only on an offset. So the numerator is a **separable correlation** of the score map with the B-spline kernel, evaluated at centres `x_c^i + x_a^j`:

```
num[i,j] = (S_map[i]  ⊛  κ_g)(x_c^i + x_a^j)
```

And the denominator is the *same* correlation applied to an all-ones indicator map — **content-independent, precompute once**.

```python
def particle_to_grid(S, H16, W16, K, den_lut, taps_x, taps_y, centres):
    """S: [B, N, N] -> P: [B, H16, W16, K*K]"""
    B, N, _ = S.shape
    smap = S.reshape(B * N, 1, H16, W16)
    smap = F.pad(smap, (pad_x, pad_x, pad_y, pad_y))       # zero pad
    blur = depthwise_conv1d(smap, taps_x, dim=-1)          # separable
    blur = depthwise_conv1d(blur, taps_y, dim=-2)
    num  = gather(blur, centres)                           # [B, N, K*K]
    return (num / den_lut).reshape(B, H16, W16, K * K)
```

**Precomputed once at startup:**

- `taps_x`, `taps_y` — B-spline evaluated at integer offsets in `[−⌊1.5g⌋, ⌊1.5g⌋]`. For g=4 that is 13 taps.
- `centres[i,j] = x_c^i + x_a^j + pad` — integer indices, shape `[N, K²]`
- `den_lut[i,j]` — shape `[N, K²]`, the indicator-map correlation. Content-independent.
- `pad_x = W16 + ⌊1.5·gx⌋ + 1`, likewise y. For 32/g=4: pad = 39, padded map 110×110.

Cost: two depthwise 1D convs (one cuDNN call each) plus a gather. No atomics, fully differentiable, trivially batched.

**Sparsity `[!]`:** 58.7% of `P`'s entries are structurally zero, and the live-anchor count varies with patch position (81 for a corner patch, 100 for a centre patch, up to 121). The zero *pattern* therefore encodes absolute patch position. Two consequences: (a) do not normalize `P` per-token in a way that divides by zero; (b) the network can in principle recover absolute position from the sparsity pattern, which slightly undercuts the pure-equivariance framing.

**Two properties that contradict a naive reading `[!]`:**

1. **`P` is not exactly translation-equivariant.** The splat (numerator) is; the denominator is not, because the reachable displacement set `{x_d − x_c}` depends on where patch *i* sits. Measured denominators for an 8×8 grid, K=4: patch (2,2) → `[0.02, 0.37, 0.60, ... 14.54]`, patch (3,3) → `[0.19, 1.34, 1.56, ... 12.69]`. ~65× spread, different per patch. The modulation is **content-independent** so the network can absorb it — but the claim is narrower than the paper states.

2. **`P` is a local weighted average, not a density. Its peak is not at the true displacement.** For a one-hot match at `f=(1,3)`: nearest anchor gives `num=0.473, den=12.69, P=0.037`; a far anchor gives `num=0.193, den=3.34, P=0.058`. The far anchor wins because fewer particles are near it.

   **Do not debug by checking whether `P` peaks in the right place.** It won't. Multiply the denominator back out and check the splat. `test_p2g.py` does this.

### 3.5 Fine CNN branch `[P]` — mostly unspecified `[C]`

Paper gives two sentences: "several low-cost convolution layers to fuse fine-grained features," and Fig. 2's caption, "fine-grained features from CNN are also fused to capture the pixel-wise error to improve the servo precision."

```python
x = cat([I_c, I_d], dim=1)                     # [B, 6, 512, 512]  -- EARLY fusion
# 4 stride-2 blocks: 6 -> 32 -> 64 -> 64 -> 64
F_fine = cnn(x)                                # [B, 64, 32, 32]
```

**Early channel-wise concatenation is the load-bearing choice.** A siamese-then-subtract design would prevent the first layer from computing cross-image derivatives. Stacking at the input lets a filter learn to be a difference operator between the `Ic` and `Id` channels.

Rationale for why this recovers sub-patch precision (Lucas–Kanade): for small displacement, `Ic(x) − Id(x) ≈ ∇I·Δx`. Photometric difference is linear in gradient times offset, so sub-pixel displacement is regressable from brightness differences with no correspondence at all. `P` cannot do this — below one 16×16 tile it is blind by construction.

**Caveat `[!]`: the paper contains no ablation of this branch.** The claim that it (rather than `P`) supplies the 0.474 mm is inference from one figure caption plus the resolution argument. It is the cheapest missing experiment: train with and without, read the TE difference. Do that early — it tells you where precision actually comes from.

### 3.6 Controller head `[P]` / `[C]`

```python
tok = cat([P.reshape(B,1024,256), F_fine.reshape(B,1024,64)], -1)   # [B,1024,320]
tok = linear(320 -> 256)(tok)
for 4 blocks: tok = self_attn(tok)                                  # RoPE
a   = cross_attn(q=action_token, k=tok, v=tok)                      # [B, 1, 256]
l_tilde, v_dir = mlp(a).split([1, 6])
v_tilde = sigma(l_tilde) * v_dir / v_dir.norm()
```

`sigma(x) = exp(x−1)` for `x ≤ 1`, else `x`. C¹ at the seam.

Note Eq. 10 writes `NeuralController(F_c, S)` but **Fig. 2 is authoritative** and routes `P` plus the CNN branch.

The action token is a single learned query — the aggregation bottleneck. Memoryless: no recurrence, unlike v1's GConvGRU.

### 3.7 Velocity denormalization `[P]` — inference only

```python
def denormalize(v_tilde, K_real, d_star):
    R_t, t_t = pbvs_inverse(v_tilde)                 # Eq. 18
    nu  = d_star * v_tilde[:3]                       # Eq. 20: only LINEAR scales
    om  = v_tilde[3:]
    if K_real.f != f_canonical:                      # Eq. 21
        s = K_real.f / f_canonical
        Sm = diag([s, s, 1])
        E_hat = Sm.T @ skew(t_t) @ R_t @ Sm
        R_h, t_h = decompose_essential(E_hat)        # SVD, Eqs. 3-4
        nu, om = pbvs(R_h, t_h)
        nu *= d_star
    return concat([nu, om])
```

**Gotcha:** `Sᵀ[t]ₓRS` is generally *not* a valid essential matrix (unequal singular values), so the SVD in Eqs. 3–4 doubles as a projection back onto the essential manifold. It's an approximation, not an identity. Fine in practice; know that it's there.

**Asymmetric failure mode `[V1]`:** underestimating `d*` costs convergence time but not precision (v1 Table VI: 433 vs 208 steps at d̂=0.25 m against true 0.711 m, RE/TE unchanged). Overestimating causes overshoot and feature loss (SR drops to 80% at d̂=2 m). **If unsure, err low.**

### 3.8 Hybrid controller `[P]`

```python
Xg_c = (C[:,None] * coords_c).sum(0) / C.sum()             # Eq. 25
Xg_d = (C[:,None] * (coords_c + flow)).sum(0) / C.sum()
if norm(Xg_c - Xg_d) > 0.1 * sqrt(N16):                    # 0.1*32 = 3.2 at 512^2
    v = -lam * inv(J) @ concat([t_hat, Xg_c - Xg_d, theta*u_z])   # Eq. 24, 2.5D
else:
    v = pbvs_output                                        # directly supervised
```

`N16 = H16·W16`. **Express the threshold as `0.1·sqrt(N16)`, never as the literal 3.2** — it must scale with grid size.

`J` is the 2½D Jacobian; the paper defers to Malis et al. 1999 [13] and does not write it out. This is the least-specified part of the method.

Table I row (7): PBVS-only drops SR 20/20 → 18/20, precision on successes unchanged. So the hybrid buys robustness only. **Ship without it in v0** and add later.

---

## 4. Data generation

### 4.1 Scene `[P]`

- IsaacSim, photorealistic
- 1–6 objects scattered on a ground plane
- 6852 meshes (GSO + OmniObject3D); see §4.4 on reducing this
- 32k background textures/materials, 733 HDR ambient maps
- Randomize object sizes and poses

### 4.2 Pose sampling `[V1]`

Upper hemisphere over the scene centre, parametrized `(d, θ)`:

| | d | θ (elevation) | perturbation `a_max` |
|---|---|---|---|
| initial | 0.5–0.9 m | 30°–90° | [10°, 10°, 60°] |
| desired | 0.5–0.9 m | 70°–90° | [5°, 5°, 15°] |

Camera z-axis points at the scene centre, x-axis parallel to world xOy, then perturb by the axis-angle above. Yields ~30°–172° initial rotation error, mean ~87° — matches v2's real-world test distribution.

### 4.3 The two collection processes `[P]`

Concurrent, both writing one shared buffer, weights periodically synced to the DAgger collector. **Not** sequential.

**Offline (uniform):**
```
sample (T_c, T_d) independently from the hemisphere
render both
label = pbvs(ground_truth_relative_pose) / d*
store
```
One frame. No rollout, no episode.

**DAgger (on-policy):**
```
render I_d once from T_d
loop:
    render I_c from current pose
    (R, t) <- SIMULATOR GROUND TRUTH          # not from the images
    v_star <- pbvs(R, t);  v_tilde_star = [nu/d*; omega]
    STORE (I_c, I_d) -> v_tilde_star
    v_pred <- policy(I_c, I_d)
    T_c <- T_c * exp(v_pred * dt)             # POLICY drives, not the expert
    if converged or ill_posed: resample
```

**The three most common reimplementation bugs, in order:**

1. **Moving the camera by `v_star`.** That is plain behaviour cloning — the dataset only ever covers the expert's straight Cartesian line, and you get the compounding-drift failure DAgger exists to prevent.
2. **Unit mismatch on integration.** Either denormalize by `d*` before integrating, or run the sim in the unit world with `d* = 1` (Eq. 19). Mixing makes the camera crawl at `1/d*` of intended speed and silently corrupts the visited-state distribution.
3. **Labelling from the images.** The expert uses simulator ground truth. An image-based expert inherits frontend error and you train the network to imitate a flawed teacher.

**Resample criterion `[V1]`:** reset when the camera reaches the desired pose, or is ill-posed — too close, too far, or most of the scene out of frame. This is a *data-balance* mechanism, orthogonal to DAgger: it stops converged episodes flooding the buffer with near-zero-velocity samples, and prunes states where no meaningful correspondence exists.

**Cold start `[C]`:** at iteration 0 the network is random but still returns six numbers, so DAgger rollouts happen — they are just garbage, and they terminate fast because the camera goes out of bounds quickly. The offline stream carries training in the early phase. Canonical DAgger uses a β mixing schedule instead; v2 mentions none, and the offline half appears to serve that role structurally. Suggested offline:DAgger sampling ratio 1:1, revisit once the policy converges in rollout.

**Underappreciated `[!]`:** uniformly sampled pose pairs give almost no near-goal data (two random points on a sphere are rarely close). **Converging DAgger rollouts are the only source of the terminal regime**, which is exactly what terminal precision depends on. See §8.3 for the diagnostic.

### 4.4 Reducing the mesh library

Object count is probably not the binding constraint: the policy conditions on displacement distributions rather than appearance, test objects are unseen either way, scene combinatorics dwarf mesh count, and textures/HDRs carry the appearance-robustness load.

**Composition matters more than count.** GSO alone skews matte, well-textured, medium, blobby. If nearly all training objects are well-textured, the policy rarely sees genuinely bimodal rows of `S` and never learns to resolve ambiguity — training out the exact capability the method claims.

- Curate deliberately for textureless, specular, near-planar
- Add real CAD of your target parts
- Cut meshes before cutting HDR maps or background textures
- Hold out a **difficulty-stratified** eval set (textured / textureless / specular) — aggregate SR will hide which competence degraded
- If you fine-tune RADIO rather than freeze it, diversity matters substantially more

No object-count ablation exists in the paper.

---

## 5. Loss

```python
L_norm = l1(sigma_inv(v_star.norm()), l_tilde)
L_dir  = 1 - cosine(v_star, v_tilde)
L      = L_dir + 0.1 * L_norm                  # weight from v1; v2 states none
```

L1 on the *log* magnitude makes the norm loss a **relative** error, so sub-millimetre terminal commands are not drowned out by gross-motion samples. This split is inherited from v1 (which used `T = 1+ELU` and MSE); v2 reparameterized to `σ` and L1.

Convergence quality lives in the direction term; precision lives in the norm term. They can be traded independently.

---

## 6. Deployment path

```
Ic, Id  ->  RADIO  ->  matching transformer  ->  S
                                              -> P            (Particle2Grid)
                                              -> C            (dual softmax)
Ic, Id  ->  fine CNN                          -> F_fine
                    concat -> controller -> v_tilde
                    denormalize(v_tilde, K_real, d*)  -> [v; omega]  m/s, rad/s
                    hand-eye transform -> robot Cartesian velocity
```

Budget at 35 FPS ≈ 28 ms: RADIO ViT-B on two images dominates. Run both images in one batched forward pass. Correlation is 1024×1024×256 ≈ 268M MACs — trivial. Particle2Grid via §3.4 is two depthwise convs. Use mixed precision — the paper explicitly relies on it to hit real-time.

---

## 7. Build order with validation gates

Do not proceed past a failing gate.

| # | Build | Gate |
|---|---|---|
| 1 | RADIO loader | `cpe_mode is True`; features at 384/512/640 stay consistent (cosine sim of the same physical patch across resolutions > 0.9) |
| 2 | Particle2Grid | `test_p2g.py` 12/12, including diff vs the literal Eq. 16 reference on random `S` |
| 3 | Correlation + `C` | on a synthetic homography pair, `argmax` of `S` rows recovers the true warp for textured patches; `C` is low on blank regions |
| 4 | Sim + PBVS expert | integrating the *expert's own* velocity converges to < 0.1 mm in < 200 steps. If not, λ/units are wrong — fix before any learning |
| 5 | Offline data only, no DAgger | overfit 100 samples to near-zero loss. Catches label-pipeline bugs |
| 6 | Full offline training | SR > 50% in sim from moderate initial error. Establishes the pipeline works at all |
| 7 | Add DAgger | SR climbs; `‖ṽ*‖` histogram grows a small-magnitude tail (§8.3) |
| 8 | Add fine CNN branch | **measure the TE delta** — this is the paper's missing ablation |
| 9 | Add hybrid control | SR improves ~2/20; precision unchanged. If precision changes, the switch threshold is wrong |

---

## 8. Diagnostics

### 8.1 `P` looks wrong

Check the splat, not `P`. Multiply `den_lut` back out. See §3.4 — `P`'s peak is genuinely not at the true displacement, so peak-location is a false alarm.

### 8.2 Trains but doesn't converge in rollout

Almost always a units or λ mismatch between label generation and integration (§4.3 bugs 1–2). Test: feed the *ground-truth* label into the integrator instead of the policy. If that doesn't converge, the bug is in the loop, not the network.

### 8.3 Poor terminal precision

Log the histogram of `‖ṽ*‖` across the buffer. An empty small-magnitude tail means DAgger rollouts are not yet reaching the goal, so the terminal regime is unrepresented — no amount of further training fixes it. This is the single most informative training-time metric.

### 8.4 Good in sim, poor in real

Check `d*` estimation first (overestimate is the dangerous direction), then RGB–depth extrinsic if you're sourcing `d*` from a depth camera — a systematic 3D bias doesn't average over N points and is usually larger than quoted depth noise.

### 8.5 Metrics to log

`SR`, `TE`, `RE`, `TT` (convergence time), `FPS`, plus: entropy of `S` rows, mean `C`, fraction of episodes hitting the ill-posed resample, and the `‖ṽ*‖` histogram.

---

## 9. Known unknowns

Not specified anywhere, and no code to check:

- `K` — chosen here as 16, constrained to give even `g`
- Frozen vs fine-tuned RADIO
- λ, loss weighting, CNN branch depth/width/schedule
- Total pose pairs, scenes, dataset size, training duration. The only number is "first 40k iterations, batch 16" — explicitly the *ablation* window, not the full run. Under DAgger "total pairs" isn't well-defined anyway
- Database size and eviction policy — Fig. 3 says only "Database"
- Offline : DAgger sampling ratio
- The 2½D Jacobian `J` in Eq. 24, deferred to [13]

---

## 10. Deviations worth making

### 10.1 Feed `C` to the network

`C^i` is already computed for Eq. 25 and then thrown away — the controller never sees an explicit confidence signal and must infer reliability from `P`'s shape. Concatenating `C` as one extra channel is free. **Cheapest likely-positive change; do it in v1 of your build.**

### 10.2 Replace Eq. 24 with a receding-horizon QP

The hybrid law is a one-step proportional controller with a hand-picked switching threshold, and Table I row (7) shows the failure mode is FoV loss. A QP over the same error stack with `Xg` kept in-frame as a hard constraint, plus joint and velocity limits, fixes that principledly and removes the `0.1√N16` heuristic. Warm-start from the policy output so you refine a globally-reasonable guess rather than solving cold.

Prior art: **DeepMPCVS** (Katara et al., CoRL 2021), flow-based visual servoing with MPC — v1's ref [9]. Read before building.

### 10.3 Consider restoring recurrence

v2 silently dropped v1's GConvGRU. Justified if dense matching has no intermittency to filter — but v2 has no temporal smoothing at all, so any transient in `S` propagates straight to the commanded velocity. If you see jitter near convergence, this is the first thing to revisit.

### 10.4 Skip the hybrid controller initially

Worth 2/20 SR and nothing on precision. Not where the method's value is.

---

## 11. Companion file

`test_p2g.py` — 12 property tests plus a literal reference implementation of Eq. 16. Set `IMPL = your_function`, signature `f(S, H16, W16, K) -> (H16, W16, K*K)`, `S` rows summing to 1, row index `i = h·W16 + w`.

Still to add on your side: gradient check through `P` w.r.t. `S`; GPU/CPU consistency if you write a fused kernel; and a timing gate, since the naive scatter is unusable at 35 FPS.
