# CNSv2 — paper-faithful recreation on the 5090 / Ubuntu 24 box

**Audience:** a fresh Claude Code instance on the 5090 machine.
**Goal:** reproduce CNSv2 (arXiv 2503.00132, *Probabilistic Correspondence Encoded
Neural Image Servo*) faithfully — Table I baseline config, §G safe velocity
control, §H training details, Fig. 3 pipeline. The 4060/8GB box this handoff
comes from ran a deliberately compromised version; every compromise is listed in
§4 so you can undo it rather than inherit it.

Paper PDF: `/home/mowito/Downloads/2503.00132v1.pdf` (copy it over).
Original prototype plan doc: `mw_ws/CNSv2_5090_SETUP.md` — written *for* a 5090
box, so its "full pipeline" sections apply directly to you. Treat §1.5 / §2
("deltas for a smaller box") as **not applicable**.

---

## 1. Get the code

```bash
git clone git@github.com:mowito/mw_CNS.git
cd mw_CNS && git checkout sv_first_draft
```

⚠️ **At handoff time ~16 files were uncommitted on the 4060 box.** Confirm with
the user that the CNSv2 work (`train_cnsv2.py`, `cns/models/hybrid_control.py`,
`cns/sim/pose_sampling.py`, `cns/sim/pose_perturb.py`, `cns/render/bproc_gen.py`,
`cns/render/bproc_eval_server.py`, `eval_servo_bproc.py`, `merge_scenes.py`,
`collect_dagger.py`, `scripts/`) was pushed before you rely on the remote.
Upstream of this repo is `hhcaz/CNS` (CNS v1) — v1 code under `cns/` is the
reference for environment/sampling/supervisor behaviour and is worth reading.

---

## 2. What the paper actually specifies (verified against the PDF, not paraphrase)

**Table I baseline (row 1) — target = SR 20/20, TE 0.948±0.606 mm, RE 0.075±0.048°**
- canonical camera `fx=fy=512, cx=cy=256, H1=W1=512`; scene scale `d* = 1 m`
- 1–6 objects randomly scattered on a ground plane
- controller φ_P (probabilistic grid), **batch size 16**, ~40k iterations
- servo uses **hybrid control initially → switches to PBVS** when the two image
  gravity centres are close

**§G Safe Velocity Control**
```
v = -λ Ĵ⁻¹ e,   e = [ d t̂_c ; cX̂_g − dX̂_g ; θ̂û_z ]                    (24)
C^i = Σ_j S^{i,j}_{c→d} · S^{i,j}_{d→c},  N16 = H16×W16                    (25)
cX̂_g = Σ C^i x^i_c / Σ C^i,   dX̂_g = Σ C^i (x^i_c + F^i) / Σ C^i
use hybrid while ‖cX̂_g − dX̂_g‖ > 0.1·√N16, else PBVS
```
- Jacobian details are deferred to **[13] Malis, Chaumette & Boudet, "2½D visual
  servoing", IEEE T-RA 15(2):238–250, 1999** — get this paper; our Jacobian is a
  reconstruction (see §4).
- **PBVS is what is "directly supervised"** — the hybrid law drives the *rollout*,
  it is NOT the regression target. (Easy to misread; we did.)
- Row (7): always-PBVS drops SR to 18/20. "The failed cases are with large initial
  viewpoint deviation inducing feature loss problem."

**§H Training Details**
> "We launch two sampling processes, one uniformly samples current and desired
> pose pairs for rendering **in the upper hemisphere** offline; Another uniformly
> samples the initial and desired poses, and adopts the **DAgger** scheme which
> **updates current poses online** for rendering with actions from the **current
> training neural policy**."

**Fig. 3 pipeline — DAgger is CONCURRENT with training, not a phase 2:**
```
Simulation Process #1 (uniform)  ─┐
Simulation Process #2 (DAgger,    ├─> Database {a, a*} ─> Training
   Env 1..N, uses pred_action)   ─┘          ^
                     Synchronize Weights Periodically ──┘
```
Renderer in the paper is **NVIDIA IsaacSim**.

**Assets (§H):** 6852 models (GSO + OmniObject3D), **32k background texture
images**, **733 HDR maps**, randomised object size/pose, background
textures/materials, ambient light.

---

## 3. Environment setup (Ubuntu 24 + 5090)

1. **Driver.** 5090 (Blackwell, sm_120) needs a recent driver + **CUDA 12.8+**
   PyTorch build. `pip install torch --index-url https://download.pytorch.org/whl/cu128`
   or newer. Verify `torch.cuda.get_device_capability()` is `(12, 0)` and that a
   real matmul runs — a torch too old for sm_120 imports fine but fails at kernel
   launch.
2. **Driver/module mismatch is a known trap.** If `nvidia-smi` says
   "Driver/library version mismatch" and `torch.cuda.is_available()` is False,
   the loaded kernel module differs from userspace NVML → **reboot** (or
   `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia_uvm`).
   This cost hours on the 4060 box and masqueraded as OOM/job-limit errors.
3. **IsaacSim.** Installed on the 4060 box at `~/isaacsim`, `~/isaacsim-6`.
   Install on the 5090 too — see §5, this is the single biggest lever.
4. **Disk.** Budget **≥1.5 TB**: OmniObject3D is ~1.2 TB, plus 32k textures,
   733 HDRIs, rendered scenes, and feature caches (~3.1 MB per training pair).
5. **RAM.** ≥64 GB strongly preferred. The 4060 box had 31 GB and that alone
   caused two separate OOM kills (§6).
6. Python deps as per the existing repo + `blenderproc` only if you keep Blender
   as a fallback renderer.

---

## 4. What is "jugaad" on the 4060 box — REPLACE these for fidelity

| # | Compromise made | Paper-faithful target |
|---|---|---|
| 1 | **BlenderProc/Cycles renderer** (48 samples) | **IsaacSim** (`render/isaac_scene.py`, see §5) |
| 2 | **DAgger as phase-2** (`collect_dagger.py`: roll out with a frozen ckpt, save, retrain) | **Concurrent** process + periodic weight sync (§6 of plan doc, Fig. 3). On 8 GB VRAM Blender (6.1 GB) + training (7 GB) cannot coexist, so the 4060 plan was *time-slicing*. With 32 GB, run it properly. |
| 3 | **batch 8** | **batch 16** |
| 4 | **fp32 training** | Paper uses mixed precision. Note: fp16 overflows the feature correlation → NaN, so `score_matrix` and particle-to-grid are **forced fp32** in `prob_match.py`; keep those islands fp32 under AMP. |
| 5 | **944 GSO models, 155 HDRIs, ZERO background textures** | 6852 models, 733 HDRIs, 32k background textures |
| 6 | **Hybrid Jacobian reconstructed** from standard VS blocks in `cns/models/hybrid_control.py` (Eq. 25 + switch threshold ARE exact) | Transcribe from [13] Malis 1999; cross-check vs **ViSP `vpServo`/`vpAdaptiveGain`** (built at `mw_ws/install/VISP` on the laptop) — the plan doc explicitly warns not to trust a from-scratch PBVS without a second source |
| 7 | **dt = 0.10 s** for eval/rollout (render-cost driven) | CNS v1 uses **1/50 s**; paper runs 30 s episodes |
| 8 | **`--min-vel` filter** to drop sub-patch samples | With correct data generation this should be unnecessary — but keep the *reason* in mind (§6.3) |
| 9 | Feature cache **mmap'd** from disk | With enough RAM, load normally |

---

## 5. The IsaacSim port — highest-value work item

**Measured Blender baseline to beat: 0.702 s per 512×512 render** (48 Cycles
samples, GSO meshes + HDRI, on a 4060). Consequences at dt=0.02 (273 expert
steps/episode):

| dt | steps/ep | 20-ep eval | 500-ep DAgger round |
|---|---|---|---|
| 0.15 | 34 | 8 min | 3.3 h |
| 0.10 | 53 | 12 min | 5.2 h |
| 0.02 (paper) | 273 | **64 min** | **26.6 h** |

Rendering is ~95% of per-step cost (ViT forward is ~30 ms). A 5090 gives maybe
3–5× on Cycles — still 6–9 h per DAgger round. **IsaacSim's RTX renderer should
be 1–2 orders of magnitude faster for this workload; BENCHMARK IT FIRST** (this
is an expectation, not a measured fact — nobody has timed IsaacSim here).

Build `cns/render/isaac_scene.py` to the same sample contract the two existing
renderers use, so the whole pipeline is renderer-agnostic:
- per-scene `.npz`: `images[M+1,H,W,3] uint8` (view 0 = desired),
  `poses[M+1]` = **wcT, OpenCV camera-to-world**, `wP[.,3]` = object centres
- `cns/sim/cnsv2_data.py: load_bproc_samples()` labels these with
  `supervisor_vel(Policy.PBVS_Straight, ...)` → `(vel_si, tPo_norm)`; it needs
  **no changes** if you match the contract
- IsaacSim also gives instance masks + intrinsics, which Fig. 3 lists as
  observation inputs (`o: rgb, instance mask, camera intrinsic`) — we never used
  masks; the paper does

---

## 6. Hard-won gotchas — do not rediscover these

### 6.1 Learning rate
`--lr 3e-4` **collapses this head onto the constant (mean-velocity) solution**:
output becomes identical for every input (pairwise cos = 1.0000) and `l_dir`
pins to the best-constant baseline. Measured at N=64: 3e-4 → 0.653 (== constant
0.653); **5e-5 → 0.016**. Same at N=256: 0.702 (== const 0.698) vs 0.120. Current
default is 5e-5. Re-tune for batch 16 but verify against the constant baseline.

### 6.2 Always compare against the BEST-CONSTANT predictor
A model that has learned nothing scores `l_dir ≈ 1 − cos` of the mean direction,
which is **not** 1.0. Measured baselines: **0.681** on a far-only val split,
**0.7191** on a 24%-near-goal split. Always recompute for your val mix; comparing
to the wrong one will make a dead model look alive. Also check
**output-varies-with-input** (pairwise cosine of outputs) and **ablate each input**
(zero/shuffle it, measure output change) — that is how we found §6.5.

### 6.3 Correspondence grid resolution — patches are 16 px
`512/32 = 16 px` per patch. A 3 mm translation error at `d*≈1 m` with `fx=512` is
~1.5 px = **0.4 patch**; measured paired `cos(Fc,Fd) = 0.9295` there (vs 0.59 far
out) — the views are near-identical in feature space and the label is
**unlearnable**. Worse, `sigma_inv(0.007) = −4.15` vs a median target of 1.49, so
those **11% of samples produced 32% of the L1 magnitude loss** and stalled
direction learning entirely (a 16k-iter run sat at `l_dir` 0.69 for 44
validations). **Implication: the paper's TE ≈ 0.95 mm is far below one patch.**
Understand how they achieve sub-patch precision (soft grid expectation? the
hybrid→PBVS handover? finer `H16`?) before trusting any near-goal data floor.
`cns/sim/pose_perturb.py` has `TE_MIN = 3 mm` — too small; ≥30 mm is ~1 patch.

### 6.4 Schedule length must scale with dataset size
`--iters` sets cosine `T_max`, but the learning need is in **epochs**. This head
needed **~12 epochs at LR ≥1.7e-5** to escape a long pre-breakout plateau. A run
that reached only 11.9 epochs total never escaped. Rule of thumb
`iters ≈ 3 × n_train` (≈24 epochs at batch 8). **Do not fix this by resuming with
a bigger `--iters`** — `sched.load_state_dict` restores the OLD `T_max` from the
checkpoint and the LR stays starved; move `*_iter*.pth` aside and start fresh.

### 6.5 The controller silently ignored the probability grid
`P` is a distribution over `K²=256` cells (entries ~1e-3) fused against
LayerNorm'd features at std ~1, so its contribution was **6.6e-4** of the feature
signal and buried under `grid_proj`'s own bias — zeroing `P` changed the output by
**2.3e-07**. Fixed by `self.grid_norm = nn.LayerNorm(grid_dim)` before
`grid_proj` in `cns/models/controller.py`. **If you rewrite the controller,
re-run the P-ablation test.** Probabilistic correspondence is the whole method;
it must measurably affect the output.

### 6.6 Long plateaus are NOT overfitting — do not early-stop through them
Val `l_dir` went 0.589 → 0.645 (looked like textbook overfitting for 6
validations) → **0.227**. A run was killed on that misreading. Genuine
overfitting does not reverse. Use generous patience (or 0) plus
best-checkpoint tracking. `train_cnsv2.py` has `--patience` and saves
`*_best.pth`; best/iter values persist across resume so a resumed run cannot
clobber the optimum with its first validation.

### 6.7 Memory
- Full-split validation OOMs: 185 val samples × 1024 tokens of cross-attention
  ≈ 6 GB. `validate()` is chunked via `--val-batch`; metrics are batch means so a
  size-weighted mean is exact.
- `load_bproc_samples` returns **uint8** images (float32 was 12.6 GB per tensor at
  4000 pairs → 38 GB peak → OOM on 31 GB); `precompute_backbone_features` scales
  per chunk on GPU.
- The feature cache is `torch.load(..., mmap=True)` — an 18.8 GB cache read into
  RAM was OOM-killed (exit 137) while its own dirty pages were still flushing.
  Peak RSS 0.56 GB vs 18.8 GB.
- Cache has a **scene-count guard**: rendering more scenes then reusing a stale
  cache silently trains on the old samples. It hard-errors on mismatch.

### 6.8 Blender leaks ~80 MB/scene
`bproc_gen` deletes objects each iteration but Blender never frees mesh/texture
data: 27.3 GB RSS by scene 348. **Render in batches of ~50 in fresh processes**
(see `scripts/`-adjacent batch scripts referenced in git history). Moot if you
port to IsaacSim.

### 6.9 `--oracle` proves almost nothing
The oracle uses ground-truth poses and never looks at an image, so 100% oracle SR
does **not** validate the image path. We chased a policy bug for hours because of
that false confidence — the real problem was that eval rendered with PyBullet
while training used BlenderProc (frozen-feature cosine only **0.36** across
renderers; the policy scored `l_dir` 0.85 there, worse than the 0.72 constant).
**Always evaluate in the domain you trained in.** Still run the oracle — it
isolates loop/integrator/pose-convention bugs from policy quality — just don't
over-read it.

### 6.10 CNS v1 sampling parameters (`cns/sim/environment.py:52-66`)
Now wrapped in `cns/sim/pose_sampling.py`. **Desired**: r∈[0.5,0.9] m, φ∈[70,90]°,
drz 15°, dry/drx 5°. **Initial**: r∈[0.5,0.9] m, φ∈[30,90]°, **drz 60°**,
dry/drx 10°. dt 1/50 s, `dist_eps` 2 mm, `angle_eps` 1°.
Three things our earlier sampler got wrong: symmetric elevation (the
desired/initial asymmetry *is* the task), **zero in-plane roll** (the paper cites
large initial in-plane rotation as where rivals fail), and scene-relative
distance. Verified: framing 0.56–0.60 frame-fill, upper hemisphere only.

---

## 7. Verification gates (run in this order)

1. **Expert sanity.** With CNS v1 sampling + dt=0.02, pure PBVS must converge:
   measured **100%, TE 1.97 mm, RE 0.37°, ~273 steps**. If your expert doesn't hit
   this, fix it before training anything.
2. **Renderer contract.** Load your IsaacSim scenes with `load_bproc_samples()`
   and check the `‖vel_si‖` histogram + `paired cos(Fc,Fd)` by bin (§6.3).
3. **P-ablation** (§6.5) — zero/shuffle `P`, confirm the output moves.
4. **Overfit test with >1 distinct direction.** A single sample is meaningless — a
   constant predictor scores ~1.0 on it. Use N≥32 with low target pairwise
   cosine; a healthy head reaches `l_dir` ~1e-4 on N=4 and should fit 32.
5. **Closed-loop in the training domain**, oracle first then policy. Report SR
   **and** the median final/initial TE ratio — SR alone hides a policy that halves
   the error but misses a 2 mm gate.
6. **Table I row 1 target**: SR 20/20, TE 0.948±0.606 mm, RE 0.075±0.048°.

---

## 8. State of results at handoff (4060 box, compromised pipeline)

| run | data | best val `l_dir` | closed-loop |
|---|---|---|---|
| 370 scenes, far-only | 1850 pairs | 0.1521 | not validly measured |
| **800 scenes, far-only** | 3570 pairs | **0.1175** (cos ≈0.88) | **0% SR**, TE ratio 1.094, RE ratio 0.554 |
| 1400 scenes, +near-goal (TE_MIN 3 mm) | 5970 pairs | 0.6748 — **failed**, see §6.3 | — |

Checkpoints preserved under `checkpoints/baseline_370scenes/`,
`baseline_800scenes/`, `aborted_iters8000/`, `aborted_16k/`.

**The 0% SR is real but its cause is understood**: the policy is accurate far from
the goal (`l_dir` 0.035 at ‖vel_si‖∈[2,2.5)) and at chance near it (translation
cosine **0.51** below 0.5, only ~6% of offline data lives there). That is exactly
the gap DAgger is supposed to fill — which is why doing DAgger *properly* (§4.2)
matters more than any other single change. Harness is proven correct (oracle 100%,
ratio 0.003).

---

## 9. Open questions to resolve on the 5090

1. **How does the paper reach TE ≈0.95 mm with 16 px patches?** (§6.3) This is the
   most important unknown — it determines whether near-goal data, the
   hybrid→PBVS handover, or something else supplies sub-patch precision.
2. **Exact Eq. 24 Jacobian** from [13]; validate against ViSP.
3. **32k background textures — we have ZERO.** `bproc_gen: rand_material()` only
   randomises Principled BSDF base colour/roughness/metallic (procedural, no
   images). The paper randomises over 32k background texture IMAGES *and*
   materials. Cheapest large win available: BlenderProc ships a `cc_textures`
   downloader (ambientCG, CC0) — `blenderproc download cc_textures <dir>`. Wire
   into the ground-plane/background material. Untouched axis, unlike GSO.
4. **Does `instance mask` (Fig. 3) feed the policy or only data generation?** We
   never used masks.
5. **Two-stage schedule** ("short-sequence then extended-sequence refinement",
   plan doc §6) — never implemented.
6. **OmniObject3D acquisition** (~1.2 TB, openxlab). User plans GSO first.
