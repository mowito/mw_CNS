# CNSv2 — paper-faithful recreation on the 5090 / Ubuntu 24 box

> ## STATUS — updated on the 5090 box, 2026-07-29
>
> Most of §3–§5 below is now **done**, and several claims in the original text are
> **wrong or superseded**. Read this block before trusting anything downstream.
>
> **Environment (all verified):** conda env `cnsv2`, torch 2.11.0+cu128, **2×
> RTX 5090** (32 GB each, both `cap=(12,0)`, real matmul + bf16 checked),
> Threadripper PRO 9975WX, 62 GB RAM, driver 595.58.03. IsaacSim 6.0.1 at
> `~/isaacsim` (bundled python 3.12.13, **no torch** — hence the IPC split below).
>
> **Assets:** 944 GSO models copied byte-exact and converted to USD
> (`data/gso_usd`); **980 HDRIs** (§4 said 155 — wrong, and the paper only asks
> 733); **1989 ambientCG CC0 background textures** (§9.3 said ZERO). AM-RADIOv2.5
> weights cached and verified: `(1,32,32,768)`, 98.2 M frozen params.
>
> **§5 IsaacSim port: DONE and measured.** `cns/render/isaac_scene.py` +
> `cns/render/isaac_eval_server.py`. The "expectation, not a measured fact" is now
> a fact: **113 ms/frame** for full randomized scenes vs BlenderProc's 702 ms
> (~6×), and **18 ms/frame** for repeated renders of one scene. `rt_subframes`
> turned out not to matter (PSNR 47–52 dB at every setting 1→32) and there is **no
> scene-change staleness** even at `warmup=0` — both measured by
> `scripts/bench_isaac.py`. RSS is flat across scenes, so §6.8's Blender leak is
> gone. This makes the paper's `dt=1/50` affordable, so `eval_servo*.py` and
> `collect_dagger.py` now default to **dt=1/50 and 1500-step episodes**.
>
> **A 10th compromise, missing from §4 entirely: the Fig. 2 fine-grained CNN
> branch did not exist.** `controller.py` fused only ViT/16 coarse features with
> `P`. Fig. 2's caption and §I are explicit that a low-cost conv branch on the raw
> image pair is fused in "to capture the pixel-wise error to improve the servo
> precision" — that is the paper's mechanism for sub-patch precision and it bears
> directly on §6.3 and §8. Now `cns/models/fine_cnn.py` (0.187 M params), fused as
> a third stream; gradient flow and ablation effect both verified.
>
> **§4 item 6 / §9.2 RESOLVED — with a correction (2026-07-30): Malis Eq. 23 is
> NOT paper Eq. 24.** Malis 1999 is at `/home/mowito/Downloads/Malis2-1-2DTRA99.pdf`
> and is transcribed exactly in `hybrid_control.py` (`hybrid_velocity`), verified
> against the numeric inverse of his Eq. 21 to 1.8e-15. But an earlier version of
> this block claimed the two laws coincide with the gravity centre as reference
> point — **wrong, and measurably so**. The paper's own e-vector (§2 below) is
> `[d t̂_c ; cX̂_g − dX̂_g ; θ̂û_z]`: translation from the Cartesian estimate, image
> error steering rotation only. Malis is the opposite split. Both converge on
> ground truth; on NETWORK estimates the Malis form let far-field phantom matching
> drive translation and diverged to 10.3 m in closed loop. Deployment and rollouts
> use `hybrid_velocity_eq24` (the paper's composition, with Malis's interaction
> blocks); [13] is the source of the *blocks*, not the composition of e. Full
> story in the `hybrid_control.py` docstring. **ViSP is still not needed.**
>
> **§4 item 2 / Fig. 3 concurrent DAgger: implemented.**
> `scripts/run_concurrent_dagger.sh` runs N render servers + N collectors (Fig. 3's
> "Env 1..N") on GPU 1 and training on GPU 0, with `<out>_sync.pth` as the
> "Synchronize Weights Periodically" channel and `cns/sim/dagger_pool.py` as a
> growable half of the database mixed in at a fixed `--dagger-frac`.
>
> **New gotchas found here — see §6.11–6.14 at the end of this document.** The
> worst one silently collapsed appearance diversity across all 944 objects.
>
> Verification gates 1–4 all pass; see `scripts/run_gates.py` and
> `scripts/verify_isaac_data.py`.

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

### 6.3 Correspondence grid resolution — and why it is NOT the blocker
`512/32 = 16 px` per patch, so one patch of image displacement is
`16 * depth / fx` metres:

| d* (desired cam -> object) | 1 patch | 1 px |
|---|---|---|
| 0.475 m (our min) | 14.8 mm | 0.93 mm |
| 0.703 m | 22.0 mm | 1.37 mm |
| 1.0 m (paper canonical) | 31.2 mm | 1.95 mm |
| 2.8 m | 87.8 mm | 5.49 mm |

**The paper's TE = 0.948 mm at d*=1m is 0.485 px = ~1/33 of a patch.** No single
-frame patch correspondence resolves that, so precision does NOT come from
per-step spatial resolution. It comes from CLOSED-LOOP INTEGRATION: the paper
runs 30 s episodes at 50 Hz = **1500 control steps**. A proportional law with an
*unbiased* direction estimate decays error geometrically, so sub-pixel final
accuracy is reachable from coarse per-step estimates. Our evals used **40** steps
(~38x less integration), which is a far bigger handicap than the patch size.

⇒ The question that matters near the goal is **bias, not precision**. Zero-mean
noise still converges; a systematic bias stalls at the bias. MEASURE THIS: take a
trained policy, put it at a range of small pose errors, and check whether the
predicted direction is zero-mean about the true one. (An earlier version of this
doc claimed "the 5 mm gate is below the architecture's resolution" -- that was too
strong and is retracted.)

Still true: samples far below one patch are near-unlearnable (measured paired
`cos(Fc,Fd) = 0.93` in the `[0,0.05)` ||vel_si|| bin) AND carry extreme
`sigma_inv` targets, so they distort the magnitude loss. Use `--min-vel` /
`--balance` rather than letting them dominate; do not set `pose_perturb.TE_MIN`
below ~1 patch at your working depth.

### 6.3b DUPLICATED SAMPLERS — check every renderer entry point
`cns/render/bproc_eval_server.py` carried its OWN copy of the old
adaptive-frame-fill pose sampler. `bproc_gen.py` was migrated to
`cns/sim/pose_sampling` and the server was missed, so for a full DAgger campaign
**every rollout and every closed-loop eval silently ran on the OLD distribution**:
measured d* 0.78-2.81 m and in-plane roll spread **0.0 deg**, versus 0.47-0.93 m /
17.1 deg from bproc_gen. That put ~80% of the training data on the wrong
distribution and meant the evals never tested in-plane roll at all -- the very
thing the sampling fix was for. Both now import `pose_sampling`. If you add an
IsaacSim renderer, import the sampler, never re-implement it, and verify with a
d*/roll histogram per data source before trusting any number.

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

> **2026-07-30: THAT HYPOTHESIS HAS NOW BEEN TESTED AND IS NOT SUFFICIENT.** Fig. 3
> concurrent DAgger ran correctly end to end — ring evicting, 727 scenes cycled,
> 66.5% of the DAgger half near-goal, offline val l_dir 0.2456 -> **0.1993** on a
> scene-disjoint split. Gate 6 on that checkpoint: **SR 0/20**, median final TE
> **426 mm**, median TE ratio 0.667 (`logs/eval_gate6.log`, §6.17). Doing DAgger
> properly moved the offline number and did not move closed loop off zero. Do not
> spend more effort on the DAgger half expecting it to close this.

---

## 9. Open questions to resolve on the 5090

1. **RESOLVED (mostly), see §6.3**: sub-patch precision comes from closed-loop
   integration over ~1500 steps, not per-frame resolution. The remaining question
   is whether the policy's near-goal direction estimate is UNBIASED — that is what
   decides if the loop converges to sub-mm or stalls. Measure it directly.
2. **CLOSED — exact Eq. 24 Jacobian.** Malis 1999 is on the box and transcribed in
   `cns/models/hybrid_control.py`; see the STATUS block. **ViSP is not needed**: the
   reason the plan doc wanted a second source was that a from-scratch PBVS is easy
   to get wrong, and transcribing the cited paper directly removes that risk.
   `tests/test_malis_hybrid.py` checks the closed form against the numeric inverse
   of Malis's own Eq. 21 matrix, which is a stronger check than agreeing with
   another library's conventions.
3. **PARTLY CLOSED — background textures: 1989, not zero.** `data/cc_textures`
   holds ambientCG's entire CC0 material set, wired into the ground-plane material
   in `isaac_scene.py` (`_make_textured_material`, randomized per scene along with
   roughness/metallic and UV tiling). Still short of the paper's 32k: ambientCG is
   *exhausted* at 2005 assets, and the paper never names its source. Closing the
   rest needs a different corpus (e.g. DTD, 5640 images) — an open question about
   what the paper actually used, not a missing download.
4. **Does `instance mask` (Fig. 3) feed the policy or only data generation?** Still
   open. The Isaac renderer now SAVES masks (`masks` key, uint32) so the experiment
   is cheap to run, but nothing consumes them yet.
5. **NOT APPLICABLE — two-stage schedule.** This is not a CNSv2 idea. It comes from
   `CNSv2_5090_SETUP.md` §6, which attributes it to "CNS v1's own README
   precedent", and in v1 it is two scripts (`cns/train_gvs_short_seq.py` →
   `cns/train_gvs_long_seq.py`, with `dataset.py` → `dataset_long.py`).
   **v1 needs it because v1 is recurrent**: `cns/models/graph_vs.py:155` defines a
   `PEConvGRUCell` used as `self.temporal_aggr` (line 239), both v1 trainers thread
   `hidden_train`/`hidden_valid` through an episode, and the short-sequence trainer
   uses `steps_for_update=8` for truncated BPTT. Short-then-long is the standard
   curriculum for that.
   **CNSv2 is stateless**, so there is nothing to curriculum: `controller.py:122`
   returns the `hidden` it was passed untouched — it exists only to satisfy the CNS
   v1 trainer contract — and no recurrent cell appears anywhere in the v2 path.
   That matches the paper: Fig. 2 is feedforward, the loss (Eq. 22–23) is a
   per-sample regression, and §IV-A trains i.i.d. at batch 16 for ~40k iterations.
   The role a long-sequence stage would play — exposing the policy to its own
   compounding long-horizon error — is filled by DAgger, which the paper *does*
   specify (§H, Fig. 3) and which is now implemented.
   ⚠️ Revisit only if recurrence is ever added to the v2 controller; the `hidden`
   passthrough is a deliberate hook for that.
6. **OmniObject3D acquisition** (~1.2 TB, openxlab). Still open and now
   disk-blocked: this box has ~277 GB free on a single 915 GB NVMe, so the
   944/6852 model gap cannot be closed without another drive.

---

## 6.11–6.14 New gotchas found on the 5090 box (2026-07-29)

### 6.11 One shared texture across all 944 GSO models — the worst bug found
`omni.kit.asset_converter` writes extracted textures to
`<output_dir>/materials/textures/<basename>`. **Every GSO model's texture is named
`meshes/texture.png`**, so converting all 944 into one flat output directory made
all 944 `.usd` files reference a single shared `materials/textures/texture.png` —
whichever model converted last. Renders came back with every object in every scene
wearing the same skin, destroying the appearance-diversity axis of the paper's
domain randomization.

**Nothing statistical caught this.** Pose distributions, `‖vel_si‖` histograms,
image std, and frame-difference checks were all clean. It was visible only in a
contact sheet. `scripts/convert_gso_to_usd.py` now writes one directory per model,
and `IsaacSceneGen` hard-refuses a flat conversion that has a shared
`materials/textures/`. **Look at your images.**

### 6.12 DAgger rollout states pile up at the goal and are unlearnable
Saving every visited state of a converging rollout is wrong. Convergence is
geometric, so at `dt=1/50` most of the ~265 steps sit within millimetres of the
goal: a measured round gave **median `‖vel_si‖` 0.027 (~0.45 of a 16 px patch)
with 97.3 % below 0.5**. Those samples are individually unlearnable (paired
`cos(Fc,Fd) = 0.93` in that bin) and their `sigma_inv` targets dominate the
magnitude loss — the same mechanism as §6.3.

Two independent guards, because this is easy to reintroduce:
`collect_dagger.py --states-per-ep` keeps states **log-uniform in translation
error** (48 per episode), and `DaggerPool(min_vel=0.05)` rejects sub-patch pairs at
ingest whatever the collector did. After both: median 0.278, min 0.052.

Relatedly, **`pose_perturb.TE_MIN` was 3 mm**, which is 1/7 of a patch at the
working depth (one patch = 16·0.7/512 = 21.9 mm). That is the recorded cause of
the failed run in §8. It is now 25 mm. §6.3 already said not to do this.

### 6.13 Camera poses must be sampled about the SCENE centre
`bproc_gen.py:120-123` computes the object bounding-box centre and samples poses
about it. An IsaacSim port that samples about the world origin instead gets
different framing and a different `d*` for the same nominal radius, because
objects scatter over ±0.18 m. Renderers must agree on this or §6.9's
"evaluate in the domain you trained in" is violated by construction.

### 6.14 Small traps that cost real time
- **`np.savez_compressed` appends `.npz`** if the filename lacks it. A temp file
  named `x.npz.tmp` becomes `x.npz.tmp.npz` and the subsequent `os.replace` fails.
  Temp names must already end in `.npz`. (Needed at all because the trainer scans
  the DAgger directory while the collector writes it — half-written files read as
  `BadZipFile`, which `DaggerPool.ingest` skips *without* marking them ingested.)
- **IsaacSim `multi_gpu` defaults on** and warns "CUDA Peer Memory Copies from
  device[0] to device[1] is NOT possible... copies across GPUs will go through main
  memory". Turn it off and pin one GPU; that is also what frees the other card for
  training.
- **Randomized dome intensity produces unusable frames.** 7 of 2800 came back
  near-black, i.e. a good pose label attached to an image with no signal.
  `IsaacSceneGen.auto_expose()` rescales against the desired view.
- **Inference scripts must not hardcode the architecture.** `fine_dim` is a real
  switch (`--fine-dim 0` is the coarse-only ablation), so
  `build_from_checkpoint()` reads shapes from the state dict. Three scripts
  previously worked only because a constructor default happened to match.
- **`ParticleToGrid` needs 9 scatter offsets, not 16.** Offsets `{-1,0,1}` per axis
  are exactly the `|a|<1.5` B-spline support; `2` and `-2` contribute identically
  zero.

### 6.15 THE VAL SPLIT WAS LEAKY — val l_dir measured memorization
`build_feature_cache` split **pairs**, not **scenes**:
```python
perm = torch.randperm(n); tr, va = perm[nv:], perm[:nv]      # WRONG
```
Every pair in a scene shares that scene's desired view, objects, background
texture, HDRI and exposure. So a "held-out" pair sat inside a scene the model had
trained on: measured **100.0% of val pairs came from scenes also in train**.

What that hid, on the first full 40k run (checkpoint `cnsv2_best.pth`,
best val l_dir 0.1878):

| split | l_dir | best-constant |
|---|---|---|
| SEEN scenes (`isaac_train`) | **0.038** | 0.746 |
| UNSEEN scenes (`isaac_pilot`) | **0.243** | 0.742 |

A **6.4x** generalization gap the val curve could not see, while the reported
0.1878 sat between the two. Both still beat the constant baseline, so the model
did learn something real — but the number being optimized and early-stopped on
was mostly memorization, and closed loop diverged from **80 mm** starts on fresh
scenes even in a brightness-matched domain.

Fixed: the split is now scene-disjoint, and `train_cnsv2.py` **hard-errors** on a
cache whose val scenes overlap train (old caches must be deleted).

**Diagnostic order that found it** — worth reusing, since three plausible
suspects were eliminated by measurement before the real one:
1. oracle closed loop → 20/20, TE ratio 0.008 ⇒ harness/integrator fine
2. GT label → `postprocess` → `integrate` → 10/10 at 2.0 mm ⇒ deployment path fine
3. brightness histograms ⇒ found a REAL train/eval gap (the eval server did not
   `auto_expose` like the generator; fixed) — but it was **not** the cause
4. scene overlap between tr/va ⇒ the cause

Also: `--patience` and best-checkpoint tracking were both keyed to the leaky
metric, so "best" was selected on memorization. Re-run before trusting any
checkpoint selected this way.

### 6.16 THE DAGGER POOL SEALED ITSELF 800 ITERS IN — Fig. 3 silently became phase 2
`DaggerPool.ingest()` checked a `full()` high-water mark and `break`ed, with no
eviction. The reserve is not a limit to stop at, it is a **ring**. On the first
concurrent run (2026-07-30):

```
iter 800:  [dagger] +1815 pairs -> 8000 pairs / 249 scenes
           [dagger] pool FULL (8000/8000 pairs, 249/1600 scenes); stopping ingest
...every ingest for the remaining 39200 iters was a no-op.
```

`--dagger-reserve 8000` against ~32 pairs/scene binds at 249 scenes, so the pool
sealed four minutes into a two-hour run. Nothing announced itself as broken:
weight syncs kept publishing, collectors kept reloading (39 times), the
beta 1.0→0.3 schedule kept advancing, and 923 fresh scenes / 16 GB accumulated in
`data/dagger_live` — while 35% of every batch was drawn from the *same 249 scenes
produced by an 800-iter-old policy*. That is exactly the phase-2 algorithm §6.12
exists to replace, wearing Fig. 3's process layout. Both collectors and both
Isaac servers burned a GPU each for output nothing read.

**Why the symptom is easy to misread:** the loss curve looks healthy. The
aborted run reached best val l_dir **0.2456 @ iter 20200** on the scene-disjoint
split and was still improving. There is no offline metric that reports "your
on-policy data stopped arriving" — only the grep for `pool FULL`.

Fixed: eviction is FIFO over whole **scenes**, oldest first
(`_evict_scene`/`_alloc_scene`). By scene and not by pair because pairs address
their goal image indirectly via `scene_id` into `fd`/`id` — recycling a scene slot
under a live pair silently repoints that pair at a different scene's goal, a
corrupt label with no symptom but a worse number. `tests/test_dagger_pool_evict.py`
stamps every synthetic scene with a unique pixel value and asserts
`ic[r] == id[scene_id[r]]` for every live row, so that class of corruption fails
the test instead of the run. Verified by mutation: dropping the pair-invalidation
half of `_evict_scene` makes the test fail.

Two consequences for callers:
- Live pair rows are **no longer the prefix `[0, n_pairs)`**. Sample through
  `pool.sample(n)`; `randint(0, pool.n_pairs)` now draws dead slots.
- `meta.pt` stores a validity mask (`pair_valid`, `scene_seq`, `seq`). Old
  prefix-format meta still loads and is widened into the slot layout.

**Result of the re-run with the ring (2026-07-30, 40k iters, 1.96 h):**

| run | pool | best val l_dir (scene-disjoint) |
|---|---|---|
| leaky-split run | frozen | 0.1878 — **invalid**, §6.15; 0.243 on unseen scenes |
| aborted static run | frozen at iter 800 | 0.2456 @ 20200 (killed at 21800) |
| **ring run** | **evicting, 727 scenes cycled** | **0.1993 @ 37800** |

The ring run's 0.1993 is measured on held-out scenes, so it beats the leaky
checkpoint's *unseen-scene* number (0.243) outright: the model now generalizes
better than the memorizing one did on fresh scenes. Pool health over the run —
200 ingests, **zero** `pool FULL`, steady at ~7995/8000 pairs / 260 scenes, ring
turned over 2.8x, and near-goal held at **66.5%** with no drift, i.e. §6.12's
pile-up did not recur. 17057 pairs rejected below `min_vel`.

Caveat on attributing the gain: one run each, no seed control, and the two runs
also differ in that the aborted one had a full pool from iter 800 while the ring
run grew from empty. But a frozen pool plausibly *hurts* offline l_dir and not
just closed loop — 5-6 of every 16 samples drawn from a fixed 249 scenes for 20k
iters revisits each of those pairs ~14 times per 1000 iters. Gate 6's closed-loop
TE ratio is still the number that decides it.

Also fixed alongside: `scripts/run_concurrent_dagger.sh` invoked bare `python`,
which resolves only inside an activated `cnsv2` env. Launched detached it failed
with `python: command not found` — *after* four minutes of IsaacSim startup,
because the two render servers spawn before the trainer. The interpreter is now
`PYTHON=${PYTHON:-python}`, checked (including `import torch`) before anything is
spawned. And the `cleanup` trap only sent TERM, which the kit python behind
`python.sh` ignores — every run so far left two render servers alive after the
trainer exited, each holding GPU memory and ~300 W until spotted by hand. It now
escalates to KILL (children included) and sweeps by cmdline, since the servers are
grandchildren via `python.sh` and a PID-only sweep misses them.

### 6.17 GATE 6 ON THE FIRST HONEST CHECKPOINT — SR 0/20, and the failure is bimodal
`checkpoints/cnsv2.pth` (val l_dir 0.1993 @ iter 37800, scene-disjoint split, DAgger
ring healthy), evaluated with the row-1 protocol — `--deploy hybrid --max-steps 1500
--te-thresh 0.0005 --re-thresh 0.05`, 20 episodes, server seed left at the default so
the episodes are PAIRED with the oracle gate on identical scenes and initial poses:

| | Table I row 1 | measured |
|---|---|---|
| SR | 20/20 | **0/20** |
| final TE | 0.948 ± 0.606 mm | median **426 mm** |
| final RE | 0.075 ± 0.048 deg | median **42.2 deg** |
| median TE ratio | — | 0.667 |

The oracle solved these same scenes at SR 100% / ratio 0.008 the same day, so the
harness is not implicated.

**The failure splits cleanly in two**, which the median hides:
- **9/20 diverge** (ratio > 1), several to tens of metres. There is no velocity clamp
  and no workspace bound — `integrate()` runs 1500 steps x dt=1/50 = 30 s, so a
  sustained wrong-direction velocity of ~1.8 m/s ends 53 m away. For these episodes
  the final-TE *magnitude* is meaningless, only "diverged" is. Report medians and SR,
  never mean±std, or one runaway dominates the statistic.
- **11/20 make real progress and stall**: median ratio 0.345, best 0.077
  (495.6 -> 38.4 mm). Real closed-loop convergence, three orders of magnitude short
  of the 0.948 mm target.

**Counter-intuitive, and it kills the obvious far-field story:** the episodes that
diverge start *closer*, not further. Over the full 20: diverged median initial TE
**497 mm** vs **566 mm** for the ones that progress, initial RE **62.9** vs
**113.8 deg**, Spearman(initial TE, ratio) = **-0.33** (it was -0.47 at n=15, so treat
the magnitude as soft — the sign is the robust part). The row-7 run reproduces the
same inversion independently (-0.57), so this is a property of the POLICY, not of the
hybrid wrapper. Whatever is wrong is not simply "the far field is weak".

**Open lead, cheap to test.** The row-1 switch fires while
`||cXg - dXg|| > 0.1*sqrt(N16)`, so a run ending 37 m out should sit in hybrid mode
almost throughout. It does not: ep 7 ends at 37321 mm having spent only 410/1500
steps there, i.e. the criterion called it near-goal for **73%** of the episode. Same
for ep 8 (52%) and ep 11 (61%). Note this is the OPPOSITE direction from the failure
already recorded in `hybrid_control.py` (ep2 spending 1493/1500 steps in hybrid mode
because matching never recovered) — there the criterion correctly reported "far", here
it reports "near" while the pose error is tens of metres. Either the gravity centres
collapse once the objects leave the field of view, or they are mis-scaled and the
threshold is meaningless — which would connect to `scripts/probe_gravity.py`'s 3-5
patch offsets in every bin. Log `||cXg - dXg||` against true TE for one diverging
episode; no retraining needed.

**Next diagnostic, not yet run:** the same 20 paired scenes with `--deploy pbvs`
(row 7, raw policy every step). That separates "the policy's velocity is bad" from
"the hybrid wrapper amplifies a mediocre policy into a runaway". Until that is run,
attributing the divergences to either the model or the control law is guesswork.

### 6.18 ROW 7 ABLATION — the deployment law is NOT the culprit; the policy is
Same checkpoint, same 20 PAIRED scenes and initial poses, only `--deploy` changed
(`logs/eval_row7.log` vs `logs/eval_gate6.log`):

| | row 1 (hybrid) | row 7 (raw policy) | paper's row 7 |
|---|---|---|---|
| SR | 0/20 | **0/20** | 18/20 |
| diverged (ratio>1) | 9/20 | **10/20** | — |
| median TE ratio | 0.667 | 1.883 (see caveat) | — |
| median final TE | 426 mm | 950 mm | — |
| four best final TE | 38 / 58 / 60 / 88 mm | 72 / 154 / 200 / 207 mm | — |

**The divergence rate is unchanged (9/20 vs 10/20), so the hybrid wrapper neither
causes nor prevents the runaways.** It changes WHICH episodes fail, not whether:
ep 0 and ep 7 are catastrophic under row 1 (7.8x, 122x) and fine under row 7
(0.84, 0.70); ep 9 and ep 16 are the reverse (0.10 -> 5.9, 0.08 -> 35.4).
Head-to-head over 20: row 1 better on 6, row 7 better on 8, comparable on 6 — no
reliable advantage either way. Row 1 does reach better BEST cases, so when the
switch helps it helps, but it is close to a coin flip per episode.

The conclusion that matters: **the policy's velocity output is itself unreliable, and
the deployment law is not what is standing between this checkpoint and Table I.**
Row 7 removes the control law entirely and still scores 0/20 against the paper's
18/20. Stop suspecting Eq. 24 / the switch; the §6.17 `||cXg-dXg||` lead explains at
most which of the two failure paths a given episode takes.

**Methodological caveat — the median is unstable on a bimodal split.** Row 7's ratios
split 10 below 1.0 and 10 above, so its "median 1.883" is the mean of the 10th and
11th sorted values (0.835 and 2.931) and describes NO actual episode. Row 1's median
sits inside its partial band and is meaningful. §6.17 says report medians and SR
rather than mean±std; add to that: when the outcome is bimodal and the split is near
50/50, report the DIVERGENCE COUNT as the primary statistic, since the median can
swing by 2x on one episode crossing 1.0.

### 6.19 A RELATIVE `--usd` PATH SILENTLY STRIPPED EVERY OBJECT TEXTURE — root-caused and FIXED
Found by `scripts/view_scenes.py`, exactly the way §6.11 was found, and invisible to
every statistic that was being watched.

**Observation (confirmed, 8 random scenes / 24 images + native-resolution crops):**
objects rendered through `cns/render/isaac_eval_server.py` carry **no albedo texture**
— plain matte white/pale-blue/beige solids with shading only. The same objects through
the `isaac_scene.py` generator are fully textured (Reebok logos, legible "Connect 4
Launchers" box art, printed nutrition panels). **Backgrounds are textured in BOTH.**
That asymmetry is the clue: the ground material is built explicitly in code
(`_make_textured_material`), while object materials arrive via
`prim.GetReferences().AddReference(model)` (`isaac_scene.py:262`) from the USD.

**Ruled out:**
- *Texture streaming / warm-up.* `scene_0000.npz` — the very first scene the generator
  ever wrote — is fully textured, so it is not a cache-warming effect.
- *Exposure.* Both paths call `auto_expose`. DAgger is mildly brighter (mean luminance
  158 vs 136 / 255) with only 0.69% clipped pixels — nowhere near enough to erase
  texture across whole objects.
- *Different asset dir.* Both are passed `--usd data/gso_usd`, and the §6.11 guard
  hard-errors on the flat layout.
- *Saturation as a detector.* DAgger whole-image saturation is 0.247 vs 0.290 — the
  textured BACKGROUND carries the colour, so this statistic does not see the problem.
  Object-pixel statistics would, but the collector saves no `masks` key.

**ROOT CAUSE (confirmed by `scripts/probe_albedo.py`).** Every launcher passes
`--usd data/gso_usd` — a RELATIVE path. `IsaacSceneGen` globbed it as-is, so
`prim.GetReferences().AddReference(model)` authored a relative reference, leaving the
referenced layer without an absolute identifier. Each converted model USD points at
its texture with the relative path `./materials/textures/texture.png`, and Omniverse's
USD->MDL translation then has nothing to resolve it against:

```
[Error] [omni.rtx.materials] [UsdToMdl] Prim '/World/objects/obj_2/Looks/material_0/material_0'
parameter 'diffuse_texture': References an asset that can not be found: './materials/textures/texture.png'
```

It falls back to a blank material rather than failing. `data/isaac_train` escaped only
because that generator run happened to be invoked with an absolute path.

**FIX:** `IsaacSceneGen.__init__` now `abspath`s `usd_dir`/`hdri_dir`/`tex_dir`,
asserts every model path is absolute, and samples 20 models at startup to confirm
`materials/textures/*` is reachable — erroring with a pointer to this section rather
than rendering blanks. Normalizing inside the class covers all four construction
sites (`isaac_scene`, `isaac_eval_server`, `bench_isaac`, `probe_albedo`).

**Verified:** same scene, same models, same lighting, relative `--usd` both times —
texture-resolution errors 6 -> **0**, mean saturation 0.2574 -> **0.4770**, and the
render goes from three white blobs to a green felt basket with visible fibre, a
legible "SCIENCE" game box and a shoe sole with its label.

**Both earlier hypotheses were wrong**, and the probe is what killed them: +120 settle
frames changed the image by 2.15/255 and attaching `instance_segmentation` by 0.41/255.
Neither settle time nor `want_masks` had anything to do with it.

**WHAT THIS INVALIDATES.** The collectors and the closed-loop eval share this renderer,
so:
- **`checkpoints/cnsv2.pth` (val l_dir 0.1993) must be retrained.** 35% of every batch
  (the DAgger half) was untextured objects; the uniform 65% was textured.
- **§6.17 gate 6 (SR 0/20) and §6.18 (row 7, SR 0/20) are both void.** 100% of that
  evaluation ran on untextured objects — off-distribution from the majority of
  training. A method whose entire mechanism is *probabilistic correspondence* was
  asked to match matte blobs with no appearance signal to correspond WITH.
- §6.18's conclusion "the policy is the problem, not the deployment law" does not
  survive either; it compared two control laws over the same broken renders.

This is the third time an eval/train renderer mismatch has produced a confident wrong
conclusion here — §6.15 step 3 (`auto_expose`), §6.11 (shared texture), now this. The
pattern: pose, label and brightness statistics all pass, because none of them look at
whether the OBJECTS carry appearance. **Run `scripts/view_scenes.py` on the collector
output before trusting any closed-loop number**, and prefer object-pixel statistics
over whole-image ones — whole-image saturation was 0.247 vs 0.290 here, which looks
fine, because the textured BACKGROUND carries the colour.

### 6.20 THE BETA SCHEDULE WAS SILENTLY DISABLED TWICE, BY TWO DIFFERENT BUGS
DAgger's whole premise is that the expert share falls so the database ends up
holding states the POLICY visits. That schedule failed to run twice, each time
silently, each time leaving a plausible-looking log.

**Bug 1 -- the annealing horizon was the "unbounded" sentinel.**
`collect_dagger.py` annealed over `--episodes`, and every launcher passes
`--episodes 100000` to mean "run until training ends". At ~500 episodes actually
completed, `frac = 500/100000 = 0.005`, so beta went 1.000 -> 0.997 and the logs
printed `beta 1.00` for every episode of two entire 40k runs. **No DAgger data was
ever on-policy**; both runs were behavioural cloning on expert trajectories, which
is the phase-2 algorithm Sec. 6.12 exists to replace, reached by a third route.
Fixed: beta now anneals on `iters_done` read from the synced checkpoint
(`--beta-iters`, set to the trainer's `--iters`), so the schedule tracks TRAINING
progress and is independent of collector throughput. A guard now REFUSES to start
when `--beta-final` is given without an explicit horizon.

**Bug 2 -- `${VAR:-default}` treats empty as unset.** The fix for bug 1 added
`BETA_FINAL=${BETA_FINAL:-0.3}` to `run_concurrent_dagger.sh`. Passing
`BETA_FINAL=""` to mean "no annealing" therefore re-enabled it at 0.3, and an
intended constant-beta=0 round instead ran beta annealing 0 -> 0.30 with the expert
share RISING. Compounding it, the sync checkpoint was pre-seeded from a previous
run whose `iters_done` was 32200, so the collector's first beta read 0.24 until the
trainer's first publish. Fixed: `${BETA_FINAL-0.3}` (no colon), a `none` sentinel,
the seeded checkpoint's `iters_done` zeroed, and the launcher now ECHOES the
resolved rollout config:
`[run] rollout: drive=pbvs beta=0.0 beta_final=<none> iters=40000`.

**The lesson both share:** a schedule that silently does nothing looks exactly like
a schedule that ran. `beta 1.00` forever and `beta 0.00 -> 0.30` are both
well-formed log output. Verify the schedule MOVED before trusting a run --
`grep -oE 'beta [0-9.]+' logs/collect_0.log | sort -u` is the whole check, and it
now appears in the queue scripts.

### 6.21 A LOG-SPACE MAGNITUDE BLOW-UP CAN END AN EPISODE IN 3 STEPS
Found by the new divergence guard, which reports the step count it bailed at. Gate
6 on run 1, ep 15: **504.7 mm -> 7223.9 mm in 3 steps**. dt=1/50, so that is 6.7 m
in 0.06 s, about 112 m/s.

This is not a direction error -- the failure mode every earlier diagnosis assumed.
It is an UNBOUNDED magnitude head. `sigma(x) = exp(x-1) if x <= 1 else x`
(`controller.py:23`) is **linear above 1**, so the regressed magnitude passes
straight through with no saturation; 112 m/s at d*~0.5 m means the head emitted a
unit-world magnitude near 224. (An earlier version of this section said the blow-up
was exponential amplification of the log-norm -- wrong: sigma is exponential only
BELOW 1, which is the near-goal regime, and there it compresses rather than
amplifies.) One such step throws the camera clear of the workspace and ends an
otherwise recoverable episode.

FIXED: `postprocess(..., max_mag=3.0)` clamps the unit-world magnitude before the
scene scale is reapplied, preserving the DIRECTION (the part the policy is good at,
per-bin cos 0.77-0.92). 3.0 is far above anything legitimate -- the translation part
of the label is ~TE/d*, so 3.0 already means "traverse three scene-distances per
second". Verified: magnitudes up to 2.0 pass through untouched, 5.0 and 8.0 clamp to
2.16 m/s at d*=0.72. Note it is invisible to `l_dir` (a direction metric) and nearly invisible
to `l_norm` (a mean over the batch), which is why 40k iterations of offline
validation never surfaced it.

### 6.22 CORRECTED GATE-6 RESULTS — the runaways were a RENDERER bug, and beta is a null result
Four runs, all evaluated on the SAME 20 paired scenes and initial poses (server seed
left at default), row-1 protocol `--deploy hybrid --max-steps 1500 --te-thresh 0.0005
--re-thresh 0.05`. This supersedes Sec. 6.17 and Sec. 6.18, both of which were
measured on untextured renders (Sec. 6.19) and are void.

| run | val l_dir | median TE ratio | improved | runaway (>3x) | median final TE | best |
|---|---|---|---|---|---|---|
| void: untextured, beta pinned 1.0, hybrid drive | 0.1993 | 0.667 | 11/20 | **9** | 426.0 mm | 38.4 mm |
| run 1: textured, beta 1.0->0.31, hybrid drive | 0.2261 | 0.407 | 16/20 | 2 | 199.7 mm | 43.1 mm |
| run 2: beta 0->0.30, PBVS drive, guards, warm start | 0.2065 | **0.172** | **19/20** | **1** | 91.8 mm | 23.0 mm |
| run 3: beta 0 CONSTANT, PBVS drive, guards, warm start | 0.2108 | 0.200 | 18/20 | 2 | 91.5 mm | **17.4 mm** |

**The bimodal runaway failure was the untextured renderer, not the policy and not the
control law.** Runaways 9 -> 2 -> 1 -> 2, and 18-19 of 20 episodes now converge partway.
Sec. 6.18's conclusion ("the policy is unreliable, the deployment law is not the
culprit") was drawn from two control laws compared over the same broken images and
does not survive; Sec. 6.17's "closer starts diverge more" inversion goes with it.

**beta is a NULL RESULT.** Runs 2 and 3 differ ONLY in the expert share (same warm
start from run 1, same PBVS driving, same guards, same shared feature cache), and they
are indistinguishable: ratio 0.172 vs 0.200, median final TE 91.8 vs 91.5 mm, 19 vs 18
improved, 1 vs 2 runaways, val l_dir 0.2065 vs 0.2108. Each wins on some measures. At
n=20 this is noise. Once the renderer is fixed and driving is plain PBVS, the expert
share between 0% and ~15% average does not measurably matter -- do not spend more time
tuning beta.

**Attribution caveat.** Runs 2 and 3 were warm-started from run 1, so they carry ~65k
cumulative iterations against run 1's 32k. Nothing here separates the warm start from
the other changes, and the run1 -> run2 jump (0.407 -> 0.172) bundles four changes at
once. The run2/run3 comparison is the only clean single-variable measurement.

**Where the remaining gap is.** SR is still 0/20 because everything stalls in the
17-92 mm band against a 0.948 mm target -- a precision problem, roughly 20-100x, not a
stability problem. That is a far better-posed failure than half the episodes flying
away. The next suspects, in order:
1. Sec. 6.21's log-space magnitude blow-up (unfixed; a clamp is cheap).
2. `min_vel` admitting 19% of DAgger pairs on ROTATION alone while its justification
   is a translation/pixel argument -- it directly shapes near-goal supervision, which
   is exactly the regime the gate demands. See the discussion above Sec. 6.20.
3. Near-goal supervision below ~5 mm is now almost absent from the DAgger half:
   beta=0 rollouts bottom out at a median final TE of 65.9 mm (min 4.9 mm), because
   with no expert steps nothing drives the camera to 2 mm any more. The uniform half
   does not cover it either (it is far-heavy). Nothing in the database teaches the
   last two orders of magnitude.
4. Sec. 6.3 / the Fig. 2 fine CNN branch -- 16 px patches as the resolution floor.

Item 3 is new and follows directly from fixing beta: a correct DAgger schedule REMOVED
the near-goal coverage that the broken beta=1.0 was accidentally providing. That is the
Sec. 6.12 pile-up argument running in reverse, and it may want an explicit near-goal
sampler rather than relying on either half of the database.

### 6.23 FOUR FIXES FOR THE PRECISION GAP (2026-07-31) — the near-goal band was empty
Sec. 6.22 left SR 0/20 with everything stalling at 17-92 mm. These four address why
nothing teaches the last two orders of magnitude. Applied, verified, NOT yet trained.

**1. `min_vel` was gating a quantity with mixed units.** One threshold on
`||vel_si||` -- a 6-vector combining unit-world translation with radians -- and
rotation dominates it (median share 0.93). So it did not thin near-goal data
uniformly; it removed specifically the fine-TRANSLATION pairs:

| of pairs with TE < 20 mm | kept | median RE of kept | median RE of cut |
|---|---|---|---|
| old gate (combined >= 0.05) | 19% | 3.72 deg | 0.96 deg |

It kept "still needs rotating" and discarded "aligned but 8 mm off". The cut ended at
~16 mm; the stall band starts at 17 mm. Now gated PER DOF -- drop only if
`tv < 0.015 AND rw < 0.012`, each half a 16px patch in its own units. Measured effect:
admits 64.1% -> 88.2%, min admitted TE 5.0 -> 1.7 mm, sub-20mm pairs 316 -> 1222
(3.9x), and sub-20mm WELL-ALIGNED pairs **0 -> 8**. The old gate admitted literally
none of the examples the endgame needs. `--dagger-min-vel` is replaced by
`--dagger-min-tv` / `--dagger-min-rw`; `stats()` now reports the fine-translation
share, which is the number the combined gate hid.

**2. There was NO out-of-view check anywhere.** `_gravity_patch` returning None was
the only signal, and `--drive pbvs` (Sec. 6.23 item below) returns before calling it.
The TE guard added in Sec. 6.22 cannot substitute, and this is structural, not a
tuning question: `TE = ||t||` of `inv(tar) @ cur`, so rotating the camera IN PLACE
leaves TE at exactly 0 while the scene leaves the frame -- measured 0.0 mm TE with
**0.00 in-frame at 45 deg of yaw**. An episode can sit at TE 50 mm / RE 170 deg
staring at empty floor and spend its whole budget saving blank images whose pose
labels are perfectly valid. Added `cns.utils.perception.in_frame_fraction` and a
`--min-in-frame 0.05` guard to BOTH the collector and the eval, reported as
`LOST-VIEW`. Note it had not yet bitten: 0% of measured saved states had objects out
of frame -- but that sample is beta=1.0 expert-driven data, and beta=0 rollouts were
pruned before they could be checked.

**3. Fixing beta REMOVED the near-goal coverage the bug was supplying.** Expert-driven
rollouts converge to 2 mm, so beta=1.0 fed the pool a dense near-goal tail by
accident. With beta working, rollouts bottom out at a measured median 65.9 mm (min
4.9), only 0-4% of collection episodes ever reach the 2 mm threshold, and the uniform
half is far-heavy by construction. So NOTHING in the database teaches sub-20mm
convergence -- the regime the 0.948 mm gate is entirely about. This is Sec. 6.12's
pile-up argument running in reverse. Fixed by seeding the START pose instead of hoping
rollouts arrive: `isaac_eval_server.py --near-frac 0.25 --near-te 0.05 --near-re 5`,
wired into the collection launcher only. **run_isaac_eval.sh deliberately does not set
it** -- gate 6 must sample the paper's full initial distribution or SR is not
comparable to Table I. Verified the seeds span TE 5-50 mm (median 16.3), RE 0.5-5 deg.

**4. The magnitude head is unbounded.** See Sec. 6.21, now fixed with
`postprocess(..., max_mag=3.0)`.

**Also this session:** rollouts are driven by plain PBVS (`--drive pbvs`, default) so
the unvalidated hybrid Eq. 24 path is off the data-collection path entirely; gate 1
validates plain PBVS at 100% / 1.97 mm.

**Watch out for item 1 vs Sec. 6.12.** These pull in opposite directions and both are
right. Sec. 6.12's failure was near-goal DOMINATING the mix (1.8% -> 56.3%), which
made the closed loop worse. Item 1 admits ~4x more near-goal pairs, so the share must
be watched: the DAgger half is capped at 35% of each batch and its `near-goal(<0.5)`
fraction has been running 57-61%, giving an aggregate around 20-25%. If that climbs
toward 50% the Sec. 6.12 regression is back. `--dagger-frac` and the log-uniform
subsampling in `_subsample_states` are the two knobs that bound it.

### 6.24 THE CONTROLLER HAS NO POSITIONAL ENCODING — the precision path is architecturally disabled
Found by auditing against `/home/mowito/Downloads/cnsv2_implementation_spec.md` (2026-07-31).
Spec Sec. 3.6 specifies the controller's token stack explicitly:

    for 4 blocks: tok = self_attn(tok)     # RoPE

`cns/models/controller.py` has **no positional encoding of any kind** -- no RoPE, no
learned embedding, nothing. (`refine.py`, the matching transformer, does have RoPE; the
controller was simply never given one.) `_SelfBlock` is a bare
`nn.MultiheadAttention`, and the single learned action token then cross-attends and
collapses 1024 tokens to one vector.

**Consequence, measured on `run2_beta_up_to_03/cnsv2_b0_best.pth`** -- shuffle the
TOKEN ORDER while keeping feature content, cos(base, shuffled):

| shuffled | cos |
|---|---|
| coarse F_c only | +0.999471 |
| grid P only | +0.998494 |
| fine CNN only | +0.999285 |
| **all three, same permutation** | **+1.000000** |

Exactly 1.0 for the joint shuffle is the signature of exact permutation invariance.
The individual figures fall just short of 1.0 only because shuffling one stream
misaligns it against the others in the per-token concat before `fuse`.

**Why this matters most for the fine CNN branch.** Position can still reach the
controller through token CONTENT: RADIO's absolute position embeddings (CPE verified
True, see backbone.py) travel inside F_c, and Sec. 3.4 notes P's zero pattern encodes
absolute patch position. The fine CNN has NEITHER -- it is a plain 4-layer stride-2
conv stack whose only positional signal is where its features sit on the grid, and
that is exactly what gets discarded. So the branch the paper adds "to capture the
pixel-wise error to improve the servo precision" (Fig. 2 caption) can contribute only
a globally pooled "how different are these two images" scalar-ish signal.

That is consistent with its ablation profile: zeroing the branch changes the output by
55% (so it is NOT dead weight, unlike the P grid in Sec. 6.5) while rearranging it
changes nothing (cos 0.998-1.000).

**STRONGEST REMAINING HYPOTHESIS for the ~100 mm precision floor**, which survived
every data-side fix: the precision path is architecturally disabled. Sec. 6.22/6.23
established the gap is not data-bound -- adding the missing sub-20mm supervision
(0 -> 8 aligned near-goal pairs, 3.9x more sub-20mm pairs) did not improve closed loop
and may have hurt it. A position-blind aggregator explains why: no amount of
fine-grained supervision helps if the head cannot localize it.

**Test:** add RoPE to the controller's self-attention per spec Sec. 3.6, retrain,
re-run gate 6. Spec Sec. 7 gate 8 frames the with/without-fine-branch TE delta as the
paper's own missing ablation; do both arms at once and the result covers it.

**Also from the same audit, not yet acted on:**
- Spec Sec. 3.6 routes `cat([P, F_fine])` ONLY, noting "Eq. 10 writes
  NeuralController(F_c, S) but Fig. 2 is authoritative". We fuse a THIRD stream,
  `feat_proj(F_c)`. Since F_c is the only stream carrying real positional content, it
  may well be dominating -- consistent with the ablation above.
- Sec. 10.1: the dual-softmax confidence `C` is computed (`dual_softmax_conf`) and
  thrown away; the spec calls feeding it to the controller "the cheapest likely-
  positive change; do it in v1 of your build."
- Sec. 3.4: our Particle2Grid is the literal `scatter_add_` (optimized to 9 anchors
  from 16), not the separable-convolution form the spec derives. Correct but slow --
  the spec calls the scatter "unusable at 35 FPS". Fine for training; a deployment
  blocker.
- Sec. 11: `test_p2g.py`'s 12 property tests + literal Eq. 16 reference were never
  ported. There is no test of P at all.
- Minor: matching transformer runs at full 768 width (spec: project to 256 first);
  fine CNN is 128 wide (spec 64); controller has 3 self-attn blocks (spec 4); lr 5e-5
  (spec 1e-4, though 5e-5 is empirically justified here and 1e-4 was never tried).
