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
