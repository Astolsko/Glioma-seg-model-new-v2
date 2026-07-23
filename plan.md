# Technical Report: Pushing a Modified-UNETR Glioma Segmentation Model Past 0.90 Dice on ET/TC/WT While Improving HD95

---

## 0. Status Reconciliation (added 2026-07-24 — authoritative, read this first)

**This section supersedes the original report below wherever they disagree.** The original report (§1–§14) was written against an *earlier* version of the code. Cross-checking it against the actual codebase (`models/unetr.py`, `utils/losses.py`, `utils/transforms.py`, `utils/dataloader.py`, `utils/engine.py`) and the real results in `logs/v1-run3/` revealed that a large fraction of its "HIGH priority" recommendations are **already implemented**, and its core HD95 diagnosis is **inverted**. Use §0 as the working source of truth; keep §4 (literature) and §6–§9 (equations) as reference.

### 0.1 Actual current results (`logs/v1-run3`, 50 epochs)

| Metric | TC | WT | ET | Mean | Note |
|---|---|---|---|---|---|
| **Val Dice** (best, ep46) | 0.821 | 0.892 | **0.783** | **0.832** | ET is the floor |
| **Val HD95 (mm)** (ep50) | 4.26 | 4.12 | **21.0** | 9.80 | ET is catastrophic |
| **Test Dice** | 0.817 | 0.892 | 0.818 | 0.842 | small test split |
| **Test HD95 (mm)** | 9.10 | 3.97 | **39.7** | 17.6 | ET catastrophic |

- Baseline mean Dice is **~0.83, not ~0.88** as §1 claims.
- The binding constraint is **ET on both Dice AND HD95**. TC HD95 is *fine* (~4 mm val). The original report's repeated claim that "TC HD95 ~13 mm is the worst" (§1, §3.1, §9) is **wrong for this model** — ET HD95 is 5–10× worse than TC.
- ET HD95 of 21–40 mm with ET Dice already ~0.78–0.82 is the signature of **stray false-positive ET blobs + empty-mismatch cases**, not poor overlap. `utils/metrics.py:compute_hd95` returns **374.0** whenever exactly one of {pred, gt} is empty; a handful of near-empty-ET patients where the model emits a few FP voxels dominates the ET HD95 average. This makes **post-processing (small-component / small-volume ET suppression) the single highest-leverage ET-HD95 fix** — it was ranked "MED, do late" in §10 but should be near the top.

### 0.2 Original plan items that are ALREADY DONE (do not re-implement)

| §10 item | Status in code | Evidence |
|---|---|---|
| #1 CLAHE → per-channel z-score | **DONE** | `ApplyCLAHEAndZscored` calls only `zscore_normalize`; `apply_clahe_to_volume` is **dead code** (never invoked). CLAHE is already gone. |
| #2 Overlapping ET/TC/WT sigmoid multi-label | **DONE** | `ConvertToMultiChannelBasedOnBratsClassesd` builds TC/WT/ET channels; all losses use `sigmoid=True`; output_dim=3. |
| #4 Boundary/HD term | **DONE (unannealed)** | `HausdorffDTLoss` at constant weight 0.2 from epoch 1 — see 0.3, this is a *problem*, not a win. |
| #5 Deep supervision | **DONE** | `aux_head_z6` / `aux_head_z3` heads, weights 0.3 / 0.15 in `combine_main_and_aux`. |
| #8 Augmentation (gamma/noise/flip/rotate) | **MOSTLY DONE** | `RandAdjustContrastd` (=gamma), `RandGaussianNoised`, `RandGaussianSmoothd`, `RandScale/ShiftIntensityd`, flips, rot90. Missing only: bias-field, elastic. |
| #9 Bottleneck 3D ASPP / dilated context | **DONE** | `DilatedBottleneck` (dilations 1/2/4, residual) on the 512-ch bottleneck. |

The model **also already contains** modules §5/§6 treated as future work: `ConvNeXt3DBlock` (7³ depthwise large-kernel + GRN, ConvNeXt-V2) and `CoordAtt3D` at every decoder stage. So the "add a large-kernel / attention refinement block" idea (§5B, §6.1) substantially overlaps what's already there — a 7³ depthwise conv already gives a large receptive field. Any new block must be justified as a **replacement that beats ConvNeXt+CoordAtt**, per the user's "replace, don't just stack" directive.

### 0.3 Genuine remaining gaps (the real backlog, re-prioritized for THIS model)

1. ~~**[BUG] Focal loss double-sigmoids.**~~ **DONE (Session 2).** MONAI `FocalLoss` removed entirely; see #2.
2. ~~**No recall weighting for ET/TC false-negatives.**~~ **DONE (Session 2).** Plain Focal **replaced** by Focal-Tversky (α=0.7, β=0.3, γ=4/3) in `utils/losses.py:CombinedLoss`; loss stays at 3 components (Dice + Focal-Tversky + HD).
3. ~~**Hausdorff term is un-annealed.**~~ **DONE (Session 2).** Linear anneal 0→full over first `hd_anneal_frac`(=0.5) of epochs via `CombinedLoss.set_epoch()`, called in `run_training`.
4. **No post-processing.** No connected-component / min-volume ET suppression at inference. Given 0.1, this is likely the biggest ET-HD95 win and is near-zero Dice risk if tuned on val. **← next up**
5. **Plain `Adam`, no EMA.** `build_training_components` uses `torch.optim.Adam` (cosine schedule already present). Cheap upgrades: `AdamW` + model EMA (decay 0.999).
6. **(Optional, later) large-kernel/boundary refinement block** — only if 1–5 plateau, and only as a *replacement* for ConvNeXt+CoordAtt at the top stage, benchmarked head-to-head.

### 0.4 Recommended order for THIS model (replaces §12)

Loss-and-inference first (cheapest, highest ET leverage), architecture last:

- **Step 1 (correctness):** fix focal double-sigmoid (0.3 #1). Re-run, confirm no regression.
- **Step 2 (ET Dice):** swap plain Focal → **Focal-Tversky** (α=0.7, β=0.3, γ=4/3), weighted toward ET/TC.
- **Step 3 (ET HD95):** add **connected-component / min-volume ET suppression** at inference + anneal the Hausdorff weight.
- **Step 4 (free gains):** `Adam`→`AdamW`, add EMA.
- **Step 5 (only if still <0.90 ET):** benchmark a large-kernel boundary block as a *replacement* for the top-stage ConvNeXt+CoordAtt.

Validate one change at a time on the fixed val split; keep only what improves ET without regressing TC/WT (per §12 discipline). Every step logged in `journal.md`.

---

## 1. Executive Summary

**The fastest, lowest-risk path from ~0.88 mean Dice to >0.90 on all three regions (ET, TC, WT) simultaneously — with improved HD95 — is not a new backbone, but a coordinated redesign of the training objective, label formulation, preprocessing, and boundary supervision, layered on top of two targeted architectural additions (a boundary/large-kernel refinement module and deep supervision).** The binding constraint is the **enhancing tumor (ET) and tumor core (TC)**: these are small, low-volume, boundary-dominated regions where region-Dice is nearly blind to boundary error, so HD95 and ET/TC Dice fail together. On third-party BraTS reproductions the vanilla UNETR sits around ET ~0.82 / TC ~0.89 / WT ~0.85 Dice, and its HD95 on TC is notably poor (~13 mm), which matches the user's symptom that HD95 lags Dice.

The recommended program, in priority order:

1. **Fix data:** replace CLAHE with per-channel z-score normalization on the brain mask (standard BraTS); add bias-field, gamma, and elastic augmentation. CLAHE is documented to inject banding/over-sharpening artifacts that hurt segmentation.
2. **Fix the label formulation:** train the **overlapping ET/TC/WT regions directly with sigmoid** (multi-label), not the 3-class softmax — a consistently reported BraTS win.
3. **Redesign the loss:** a **scheduled composite** = Dice-CE (or Dice-Focal) + Focal-Tversky (recall-weighted for ET/TC) + an **annealed boundary/surface term** for HD95.
4. **Add deep supervision** at 2–3 decoder scales.
5. **Add a boundary-aware refinement block** (3D Large-Kernel Attention, which is SOTA-validated on BraTS 2020) at the highest-resolution decoder stage, and optionally upgrade the encoder→decoder skips with cross-attention.
6. **Regularize/optimize:** model EMA weights, DropPath, AdamW + cosine/poly schedule; consider SAM if a second GPU-pass is affordable.

Each item below is given with equations, published BraTS/medical evidence, compatibility notes for the UNETR transformer-encoder/conv-decoder design, MONAI hooks, and a ranked roadmap. Emphasis throughout is on the weakest classes (ET, TC) and on boundary metrics, because those are what gate the >0.90 target.

---

## 2. Current Architecture Analysis

The user's system is a **modified UNETR**: a 12-layer flat ViT encoder over 96×96×96 volumes, multi-scale features extracted at transformer layers {3, 6, 9, 12}, feeding a convolutional decoder (Feature Extraction Block: transposed conv → 3×3×3 conv → GroupNorm → ReLU, upsampling 6→12→24→48→96; Feature Refinement Block: 3×3×3 conv → GN → ReLU; final 1×1×1 conv). It already contains two custom modules:

- **Weighted Bidirectional Skip Connection (BSC):** learnable sigmoid-gated scalars α=σ(w_s), β=σ(w_d) weight shallow encoder feature F_s and deep decoder feature F_d, concatenate → 1×1×1 conv + GN + ReLU → concat with F_d → 3×3×3 conv refinement. This is a sound, lightweight adaptive-gating design.
- **3D Coordinate Attention (CA):** factorized attention along D/H/W via three orthogonal adaptive pools → shared 1×1×1 conv + GN + hard-swish → per-axis 1×1×1 conv + sigmoid gates → reweight. Uses InstanceNorm (GN with 1 group) and spatial dropout p=0.1.

Loss is Dice + CE; preprocessing is CLAHE + rotation/flip augmentation; data is BraTS 2019+2020 with a 70/15/15 split.

**Reference points for calibration (named, with caveats):**
- **UNETR (Hatamizadeh et al., WACV 2022)** headline results were on BTCV multi-organ CT and MSD, not the BraTS challenge server; third-party BraTS 2021 reproductions (e.g., SegTransVAE) report UNETR at roughly **ET 82.2 / WT 85.1 / TC 89.5 Dice**, with HD95 ET ~5.6, WT ~7.6, **TC ~13.2 mm** — i.e., a flat-ViT encoder underperforms hierarchical models on boundaries.
- **nnFormer on BraTS2021** (from the MMEF-nnFormer paper): **ET 0.839 / TC 0.878 / WT 0.915**, mean 0.877.
- **Swin UNETR (Hatamizadeh et al., arXiv:2201.01266, 2022):** 0.913 mean Dice on the BraTS 2021 validation set (ET/WT/TC = 0.858/0.926/0.885), ranked among the top methods across more than 2000 submissions.
- **SegMamba (Xing et al., MICCAI 2024, arXiv:2401.13560)** on BraTS2023 (1,251 volumes): **WT 93.61% / TC 92.65% / ET 87.71% Dice; HD95 3.37 / 3.85 / 3.48**, average Dice 91.32% (avg HD95 3.56), versus UX-Net 89.69% and SwinUNETR-V2 89.39%.
- **The cleanest single method that hits ET>0.90 AND TC>0.90 AND WT>0.92 simultaneously** in the literature is **UPMAD-Net** (arXiv:2505.03494, 2025) on BraTS2021 validation: **ET 90.00 / WT 94.72 / TC 92.42; HD95 2.45 / 2.88 / 2.38 mm** (note: preprint, not challenge-verified; ET drops to 89.18 on the held-out test split). Classic challenge winners — NVIDIA's Optimized U-Net (mean Dice ~0.8855), nnU-Net BraTS2020 (ET 82.0/TC 85.1/WT 89.0), SegResNet/Auto3DSeg (ET 0.86/TC 0.887/WT 0.926) — generally cap **ET at 0.86–0.88**, confirming that ET is the field-wide bottleneck.

**Interpretation:** reaching >0.90 on ET/TC is at the frontier of what published single models achieve on region-Dice; it is attainable but requires the loss/label/boundary changes below, not merely a bigger backbone. The user's mean of 0.88 is competitive; the gap is concentrated in ET/TC boundary quality.

---

## 3. Weaknesses Identified (why ET/TC and HD95 underperform)

1. **Region-Dice is boundary-blind on small structures.** ET occupies a tiny voxel fraction; a few misplaced boundary voxels barely move WT Dice but severely move ET Dice and HD95. WT is large and topologically simple, so it is "less sensitive to minor segmentation errors" (a point made explicitly in the MSD-KMamba analysis), which is exactly why WT is already near 0.90 while ET/TC lag.
2. **No boundary/distance term in the loss.** Dice+CE optimizes overlap only. Models "trained over such loss functions run the risk of achieving high accuracy for the Dice coefficient but exhibit poor accuracy regarding HD-based metrics" (Generalized Surface Loss, Celaya et al. 2023). This directly explains "HD95 is notably worse than Dice."
3. **CLAHE is a questionable choice for multi-modal MRI.** A 2025 multi-center nnU-Net study found CLAHE "introduced structured artifacts, such as banding and over-sharpening... These distortions negatively affect the segmentation performance and stability during training" and excluded it, retaining only z-score normalization. Standard BraTS practice is per-channel z-score.
4. **Flat ViT encoder loses fine spatial detail.** The single-resolution ViT (no hierarchical downsampling) is weaker at fine boundaries than Swin-style hierarchical encoders; this is the architectural reason UNETR trails Swin UNETR/SegMamba on TC HD95.
5. **No deep supervision.** Auxiliary losses at decoder scales are a repeatedly validated BraTS booster (nnU-Net, Optimized U-Net) that improves gradient flow to the ET/TC-relevant fine scales.
6. **Possibly training the disjoint 3-class softmax** rather than the overlapping ET⊂TC⊂WT sigmoid formulation, which the BraTS community (No New-Net, Auto3DSeg, NVIDIA) has shown "is beneficial for performance."
7. **Class imbalance not explicitly addressed** beyond Dice; ET/TC need recall-weighting (Tversky/Focal-Tversky).

---

## 4. Literature Review (organized by research area, 2018–2026)

### 4.1 Loss functions (VERY IMPORTANT — highest leverage for ET/TC and HD95)

| Loss | Key paper (venue, year) | Math essence | BraTS/medical evidence |
|---|---|---|---|
| Dice / Soft-Dice | Milletari, 3DV 2016 | 1 − 2⟨p,g⟩/(‖p‖+‖g‖) | Baseline; overlap-only, boundary-blind |
| Dice-CE (current) | — | L_Dice + L_CE | Solid baseline; no boundary term |
| **Tversky / Focal-Tversky** | Salehi MLMI 2017; **Abraham & Khan, ISBI 2019** | TI = TP/(TP+αFN+βFP); FTL=(1−TI)^(1/γ) | FTL + attention U-Net improved Dice by **25.7% (BUS2017) and 3.6% (ISIC2018)** over U-Net, optimal γ=4/3; recall-weighting (α>β, e.g. α=0.7) directly targets ET/TC false-negatives |
| **Boundary Loss** | **Kervadec et al., MIDL 2019 / MedIA 2021** | ∫_Ω φ_G(p)·s_θ(p) dp, φ_G = signed distance map of GT | Combined with GDL, "up to **8% improvement in Dice** and **10% improvement in Hausdorff**" on ISLES/WMH; trivially extends to 3D/multi-class; stabilizes training |
| Hausdorff-DT Loss | Karimi & Salcudean, IEEE TMI 2019 | Uses distance transforms to directly estimate HD | Directly reduces HD on 2D/3D U-Net |
| **Generalized Surface Loss** | Celaya et al., 2023 (arXiv:2302.03868) | Weighted, normalized surface integral w/ class weights | Lower HD95 and ASD than Dice/HL/BL on **BraTS** and LiTS with nnU-Net |
| Weighted Normalized Boundary Loss | Celaya et al. 2023 | Class-imbalance-weighted boundary term | Better HD/ASD, comparable Dice, on BraTS |
| Unified Focal Loss | Yeung et al., Comput Med Imaging Graph 2021 | Generalizes Dice+CE with focal parameter | Consistently outperforms 6 losses on 5 datasets incl. **BraTS20** (3D multiclass) |
| clDice (topology) | Shit et al., CVPR 2021 | Soft-skeleton overlap; preserves connectivity | Topology gains for tubular; less critical for blob-like glioma but can help TC connectivity |
| PolyLoss | Leng et al., ICLR 2022 | L_Poly-1 = −log(P_t) + ε₁(1−P_t) | One-line CE improvement; "outperforms CE and focal loss" on classification/detection; adopted as CE replacement in some BraTS pipelines |
| Lovász-Softmax | Berman et al., CVPR 2018 | Convex (Lovász) surrogate of Jaccard | Better IoU than CE; modest medical gains; recovers small objects |

### 4.2 Attention mechanisms for 3D medical segmentation

- **3D Large-Kernel Attention (LKA):** Li, Nan, Del Ser & Yang, *Cognitive Computation* 16(4):2063–2077, 2024 (Epub Feb 2023; DOI 10.1007/s12559-023-10126-7). Decomposes a large kernel into depth-wise + depth-wise-dilated + 1×1 convs; the best "Mid-type" 3D LK-attention U-Net achieved **state-of-the-art on CT-ORG and BraTS 2020**. Combines local context + long-range dependence + channel adaptation, with a receptive field like self-attention but far cheaper. **This is the single most BraTS-validated attention upgrade available and is the recommended addition.**
- **Deformable Large-Kernel Attention (D-LKA):** Azad et al., WACV 2024 (arXiv:2309.00121). Adds deformable sampling to LKA; a 3D variant exists (offset layer inserted after the depth-wise convs). Reported on Synapse (~84.9% 3D DSC), NIH Pancreas, ISIC — **not** on BraTS, so treat as promising but unvalidated for glioma.
- **EMA (Efficient Multi-scale Attention):** Ouyang et al., ICASSP 2023. Groups channels, cross-spatial learning; reported to reach "comparable accuracy to the CA attention mechanism with fewer parameters" and +0.52% top-1 over Coordinate Attention on ImageNet with ResNet. A candidate replacement for the user's CA if a leaner module is wanted.
- **SimAM:** Yang et al., ICML 2021. Parameter-free 3D attention via a neuroscience energy function; in DAUNet ablations it gave **+1.42% Dice and −1.50 HD95** on an ultrasound set with zero added parameters. Caveat: the "3D weight" is derived from global mean/variance only (limited local context).
- **Triplet / Coordinate / CBAM:** the user already has Coordinate Attention, which is competitive; EMA and LKA are the meaningful upgrades.

### 4.3 Transformer / architecture upgrades compatible with the encoder

- **Swin UNETR** (Hatamizadeh 2022): hierarchical shifted-window encoder, 0.913 mean Dice BraTS2021 — the natural encoder upgrade but a large change.
- **SwinUNETR-V2** (He et al., MICCAI 2023): adds stagewise residual convolutions to inject inductive bias, improving on six CT/MRI datasets.
- **SegMamba** (Xing MICCAI 2024): SSM encoder + CNN decoder, WT/TC/ET 93.61/92.65/87.71, HD95 ~3.5, high efficiency at 64³ resolution — a strong reference/alternative but a backbone swap.
- **MedNeXt** (Roy et al., MICCAI 2023): fully ConvNeXt 3D encoder-decoder with large kernels and the **UpKern** trick (initialize 5×5×5 kernels from upsampled trained 3×3×3 weights). BraTS2023 reproductions: WT 92.41 / TC 87.75 / ET 83.96, HD95 ~4.7. Its large-kernel idea motivates the LKA addition.
- **UNETR++** (Shaker et al.): Efficient Paired Attention (EPA) with shared query/key across spatial+channel attention — a template for cheaper attention in the decoder.

### 4.4 Skip connections beyond the user's BSC

- **Dual Cross-Attention** (Ates et al., 2023) and **U-Transformer** (Petit et al., 2021): cross-attention on skips "allows a fine spatial recovery in the U-Net decoder by filtering out non-semantic features," bridging the encoder-decoder semantic gap. Reported large gains over plain and Attention-U-Net skips.
- **UNet++/UNet3+**: nested/full-scale dense skips to narrow the semantic gap; heavier.
- The user's BSC is already a good adaptive gate; a **cross-attention skip at the highest-resolution level only** is the targeted upgrade (see §6).

### 4.5 Context modules

- **3D ASPP / dilated multi-scale context** at the bottleneck: parallel atrous convolutions (rates 6/12/18) aggregate multi-scale context "critical for segmenting brain tumors of diverse sizes." Cheng et al. (Comput & Graphics 2024) multi-scale context + attention block: ET/WT/TC 78.19/90.10/83.98 on BraTS2020 val.

### 4.6 Deep supervision

- **nnU-Net (Isensee, BraTS2020)** and **Optimized U-Net (Futrega et al., BraTS21, won validation phase, 3rd in test)** both use deep supervision with auxiliary heads at decoder scales. Optimized U-Net's ablation found deep supervision + a few modules jointly best. Attention-guided decoders with deep supervision "produce significant improvement in the segmentation performance of enhancing tumor and tumor core" (Prior Attention Network).

### 4.7 Regularization / optimization

- **SAM (Sharpness-Aware Minimization, Foret et al. 2021):** seeks flat minima; medical studies report "superior generalization... and more accurate boundary delineation," at ~2× the per-step cost.
- **DropPath/DropBlock, spatial dropout:** in Optimized U-Net's ablation, drop-block and residual connections contributed to the winning config.
- **Model EMA / SWA:** widely used in BraTS winners for stability and a free Dice/HD95 bump.

### 4.8 Data / preprocessing

- Standard BraTS: co-registration, 1 mm³ isotropic resampling, skull-stripping, then **per-channel z-score normalization** on the brain region. CLAHE excluded in careful pipelines due to artifacts.
- Augmentation with evidence: random flips (all axes), intensity shift/scale, **gamma correction, bias-field simulation, elastic deformation** (all in MONAI). MixUp/CutMix are less standard for 3D BraTS but region-mixing helps rare classes.

---

## 5. Proposed Architectural Improvements

**A. Deep supervision heads (high priority, low risk).** Attach 1×1×1 conv + sigmoid heads at decoder resolutions 24³ and 48³ (and optionally 12³), upsample to 96³, and add their losses with weights ~[1.0, 0.5, 0.25] (or nnU-Net's normalized descending weights). Improves gradient flow to fine scales that carry ET/TC boundary information.

**B. 3D Large-Kernel Attention (LKA) refinement block at the top decoder stage (high priority).** Insert one "Mid-type" 3D LKA block (validated on BraTS2020) at the 96³ (or 48³) decoder stage, replacing or augmenting the final Feature Refinement Block. See §6/§7 for math. Expected to help ET/TC by giving the highest-resolution features a self-attention-sized receptive field with channel adaptation, at modest FLOP cost (decomposed convolution).

**C. Bottleneck 3D ASPP context module (medium priority).** Add parallel dilated convs (rates 2/4/6 at 6³ bottleneck; small rates because the feature map is tiny) to aggregate multi-scale context for variable tumor sizes.

**D. Cross-attention skip at the highest-resolution level only (medium priority, medium risk).** Replace the top BSC with a lightweight multi-head cross-attention that queries decoder features against encoder features, filtering non-semantic activations. Keep BSC elsewhere to limit compute.

**E. Encoder upgrade to hierarchical/Swin (low priority, high effort/risk).** Only if A–D plateau below target; swapping to a Swin UNETR or SwinUNETR-V2 encoder is the largest structural change and should be validated last.

---

## 6. Proposed New Blocks (with derivations)

### 6.1 Boundary-Guided Coordinate-LKA (BG-CLKA) — recommended composite block

**Motivation.** The user already has Coordinate Attention (axis-factorized spatial gating) and lacks (i) a large receptive field and (ii) explicit boundary emphasis. LKA supplies the receptive field + channel adaptation; a boundary gate supplies HD95 pressure. Combining them reuses existing code and targets both failure modes.

**Definition.** Given decoder feature F ∈ ℝ^{C×D×H×W}:

1. **LKA branch (spatial+channel long-range):**
   - A(F) = Conv_{1×1}( DW-D-Conv( DW-Conv(F) ) )  — depth-wise K₁ conv, then depth-wise dilated K₂ (dilation d) conv, then 1×1 pointwise. This approximates a K×K×K kernel with K ≈ K₂·d.
   - F_LKA = A(F) ⊗ F  (element-wise; no softmax, preserving high-frequency detail).
2. **Coordinate-attention branch (kept from current model):** F_CA = CA(F).
3. **Boundary gate:** compute a soft edge map E = |F − AvgPool₃(F)| (local contrast), pass through 1×1 conv + sigmoid → g_b ∈ (0,1); F_edge = g_b ⊙ F.
4. **Fuse:** F_out = Conv_{1×1}( [F_LKA ; F_CA ; F_edge] ) + F  (residual).

**Why it should work.** LKA gives a self-attention-scale receptive field at conv cost (validated SOTA on BraTS2020); CA preserves the axis-aware localization the model already benefits from; the boundary gate up-weights high-gradient voxels, applying HD95 pressure at the feature level rather than only in the loss. Residual + no-normalization multiplication keeps gradients stable.

**Compute cost.** LKA decomposition makes a K=21-equivalent kernel cost ≈ two depth-wise convs + one 1×1 — roughly +5–10% FLOPs at the top stage. Boundary gate is negligible.

**Expected gain.** +0.5–1.5 Dice on ET/TC and a meaningful HD95 reduction (LKA alone is SOTA-validated on BraTS2020; the boundary gate adds HD95 pressure). **Disadvantage:** three-branch concat increases channel width transiently (mitigate with a 1×1 bottleneck); risk of over-smoothing if the edge gate saturates (initialize its bias negative).

### 6.2 Uncertainty-guided skip fusion (optional, medium)

Replace the top BSC scalar gate with a spatially-varying gate derived from predictive entropy: at the deepest supervised head, compute H(p) = −Σ p log p; upsample and use σ(w·H) as a per-voxel weight on the shallow encoder feature so that **uncertain (usually boundary) voxels pull in more high-resolution detail.** Mathematically this generalizes the BSC scalar α to a voxel field α(x)=σ(w_s + w_h·H(x)). Cheap, and directly targets ambiguous ET/TC boundaries. Disadvantage: needs an auxiliary head (already added via deep supervision) and can be noisy early in training (anneal w_h).

---

## 7. Mathematical Intuition (equations for key proposals)

**Soft Dice (per region c):**
L_Dice = 1 − (2 Σ_x p_c(x) g_c(x) + ε) / (Σ_x p_c(x) + Σ_x g_c(x) + ε).

**Tversky index and Focal-Tversky:**
TI_c = Σ p_c g_c / (Σ p_c g_c + α Σ(1−g_c)p_c + β Σ g_c(1−p_c)), with α+β=1.
Set **α=0.7, β=0.3** to penalize false negatives (missed ET/TC voxels) more than false positives.
L_FTL = Σ_c (1 − TI_c)^{1/γ}, γ=4/3 (Abraham & Khan). The exponent focuses learning on hard, poorly-segmented regions.

**Boundary (surface) loss (Kervadec):**
L_B(θ) = Σ_x φ_G(x)·s_θ(x), where φ_G is the **level-set / signed distance map** of the ground-truth boundary (negative inside, positive outside), precomputed once per volume. Minimizing it moves the predicted region toward the true contour; being a boundary integral, it is immune to the region-size imbalance that cripples ET.

**Hausdorff-DT loss (Karimi):**
L_HD ≈ Σ_x (p(x) − g(x))² · ( d_G(x)^a + d_p(x)^a ), where d_G, d_p are distance transforms of GT and prediction. Directly penalizes far-away errors → attacks HD95.

**LKA attention (§6.1):** A(F) = Conv_{1×1}(DW-D-Conv(DW-Conv(F))), F_out = A(F)⊗F. The composition of a (2d−1)³ depth-wise conv and a ⌈K/d⌉³ dilated depth-wise conv reconstructs a K³ receptive field with parameter count ~ C(⌈K/d⌉³ + (2d−1)³ + C) instead of C²K³.

**Deep supervision:** L_total = Σ_s w_s · L(ŷ_s, y↓_s), weights w_s descending with depth, ŷ_s upsampled or y downsampled to each scale.

---

## 8. Loss-Function Redesign — Final Recommended Composite

**Recommended objective (multi-label sigmoid over ET, TC, WT):**

**L = L_DiceCE + λ_T · L_FocalTversky + λ_B(t) · L_Boundary**

with:
- **L_DiceCE** = Dice + CE — the stable region+pixel anchor (MONAI `DiceCELoss(sigmoid=True)`).
- **L_FocalTversky** with α=0.7, β=0.3, γ=4/3, weighted more on ET/TC channels — attacks false-negatives on small classes (MONAI `TverskyLoss`, or `FocalLoss` combined; Focal-Tversky may need a short custom module).
- **L_Boundary** = Kervadec surface loss using precomputed signed distance maps — attacks HD95.

**Scheduling (critical).** Boundary losses destabilize training if applied full-strength from the start. Use the **rebalancing/annealing schedule** (Kervadec; Celaya uses a linear α schedule): start λ_B(0)=0 (or 0.01) and **linearly increase to ~0.5–1.0** over training while holding the region terms; equivalently use L = (1−α)L_region + α L_boundary with α: 0→~0.5. Set λ_T ≈ 0.5 constant.

**Why this composite over alternatives:**
- Pure Dice/DiceCE: boundary-blind → HD95 lags (the current problem).
- Adding **Focal-Tversky** provides the recall pressure that lifts ET/TC Dice (Abraham & Khan; ReHyDIL reports Tversky-based losses give +2.53% TC, +1.52% ET on BraTS2019).
- Adding an **annealed boundary/surface term** is the single best-evidenced HD95 reducer that does **not** hurt Dice (Kervadec: +8% Dice/+10% HD together; Celaya: lower HD95/ASD on BraTS with comparable Dice).
- **Unified Focal Loss** is a strong single-term alternative (outperformed 6 losses incl. on BraTS20) if you prefer fewer hyperparameters — a reasonable substitute for the DiceCE+FTL part.

**MONAI:** `monai.losses.DiceCELoss`, `DiceFocalLoss`, `TverskyLoss`, `GeneralizedDiceLoss`, `HausdorffDTLoss` (yes, MONAI ships a Hausdorff-DT loss), and `SoftclDiceLoss`/`SoftDiceclDiceLoss`. Boundary/surface loss with precomputed SDMs is a ~30-line custom module (or use `HausdorffDTLoss` as the distance-based term).

---

## 9. HD95 Improvement Strategy (dedicated section)

HD95 is the user's stated weak metric; here is the ranked, evidence-backed plan, flagged for Dice safety:

1. **Annealed boundary/surface loss (Kervadec) — improves HD95, does NOT hurt Dice** (published +8% Dice/+10% HD together). *Top priority.* Use MONAI `HausdorffDTLoss` or a signed-distance surface term, ramped in.
2. **Generalized/Weighted-Normalized Surface Loss (Celaya 2023) — improves HD95/ASD on BraTS, comparable Dice.** Alternative to #1 with better numerical properties for class imbalance.
3. **3D LKA boundary-refinement block (§6.1) — improves both**, by giving high-res features a large receptive field + explicit edge gate.
4. **Deep supervision — mild HD95 help via better fine-scale gradients; safe.**
5. **Train overlapping regions with sigmoid — cleaner TC/ET boundaries, safe.**
6. **Post-processing: connected-component removal of tiny false-positive ET blobs** — large HD95 win under lesion-wise metrics (BraTS penalizes stray FP lesions heavily); near-zero Dice cost if the threshold is tuned on validation. Model EMA also smooths boundaries.
7. **SAM optimizer — reported "more accurate boundary delineation"; safe but ~2× step cost.**

**Do NOT** rely on Dice-only tuning or add parameters without a boundary signal — that is precisely why HD95 currently lags.

---

## 10. Ranked Implementation Roadmap

| # | Modification | Δ Dice (esp. ET/TC) | Δ HD95 | Impl. complexity | Train cost | Infer cost | Risk | Compat. | Priority |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Replace CLAHE → per-channel z-score + brain-mask norm | + (removes artifacts) | + | Trivial | none | none | Very low | Full | **HIGH** |
| 2 | Train overlapping ET/TC/WT with sigmoid (multi-label) | ++ | + | Low | none | none | Low | Full | **HIGH** |
| 3 | Composite loss: DiceCE + Focal-Tversky (α=0.7) | ++ (ET/TC) | + | Low | none | none | Low | Full | **HIGH** |
| 4 | Add annealed Boundary/Surface (HausdorffDT) term | + | +++ | Low-Med | +precompute SDM | none | Med (anneal!) | Full | **HIGH** |
| 5 | Deep supervision heads (2–3 scales) | ++ | + | Low | +small | none | Low | Full | **HIGH** |
| 6 | 3D LKA refinement block at top decoder stage | ++ (ET/TC) | ++ | Med | +5–10% | +5–10% | Low-Med | Full | **HIGH** |
| 7 | Model EMA weights | + | + | Low | +small | none | Very low | Full | **HIGH** |
| 8 | Augmentation: gamma, bias-field, elastic (MONAI) | + | + | Low | +small | none | Low | Full | MED |
| 9 | Bottleneck 3D ASPP context | + | + | Med | +small | +small | Low | Full | MED |
| 10 | Cross-attention skip (top level only) | + | + | Med-High | +med | +med | Med | Full | MED |
| 11 | AdamW + cosine/poly + warmup; DropPath | + (stability) | + | Low | none | none | Low | Full | MED |
| 12 | Uncertainty-guided skip fusion (§6.2) | + (boundary) | + | Med | +small | +small | Med | Full | MED |
| 13 | SAM optimizer | + (generalization) | + | Med | +~2× step | none | Med | Full | LOW-MED |
| 14 | clDice / topology term for TC connectivity | ~ | + | Med | +med | none | Med | Full | LOW |
| 15 | Post-hoc small-FP-lesion removal | ~0 | +++ (lesion-wise) | Low | none | +small | Low | Full | MED (do late) |
| 16 | Swap encoder to Swin/SwinUNETR-V2 | ++ | ++ | High | +large | +med | High | Rewrite | LOW (last resort) |

---

## 11. Expected Impact of Every Change (per-class reasoning)

- **z-score (1):** removes CLAHE banding that most corrupts subtle T1ce enhancement → helps **ET** most, plus overall stability.
- **Sigmoid overlapping regions (2):** the network optimizes exactly the evaluated nested regions; documented BraTS boost, especially **TC/ET** boundaries.
- **Focal-Tversky (3):** recall weighting recovers missed small **ET/TC** voxels (false-negative-dominated) → the change most likely to lift the two weakest classes.
- **Boundary/Surface loss (4):** the primary **HD95** lever across all classes, most visible on **TC** (worst HD95 in UNETR baselines).
- **Deep supervision (5):** better fine-scale gradients → **ET/TC** Dice and convergence.
- **3D LKA block (6):** large receptive field + channel adaptation, SOTA-validated on BraTS2020 → **ET/TC** Dice and **HD95**.
- **EMA (7):** free ~+0.2–0.5 Dice and smoother boundaries.
- **Augmentation (8):** generalization; bias-field and gamma especially help cross-scanner **ET** enhancement variability.
- **ASPP (9):** multi-scale context helps **WT** (large, variable) and edema extent.
- **Cross-attention skip (10):** sharper boundaries where semantic gap is largest → **TC/ET**.
- **Post-processing (15):** the biggest **HD95** win under lesion-wise scoring (removes stray FP ET lesions that otherwise incur large distance penalties), near-zero Dice cost.

Cumulatively, items 1–7 are the realistic route from ~0.88 to >0.90 on all three regions with lower HD95; 8–15 are consolidation.

---

## 12. Suggested Implementation Order (incremental, validate each step)

Implement one change at a time, re-evaluating ET/TC/WT Dice and HD95 on a fixed validation split after each. **Stop-and-keep** any change that improves the weakest-class Dice or HD95 without regressing others; revert otherwise.

**Stage 0 — Data hygiene (no architecture change):**
1. Switch CLAHE → per-channel z-score on the brain mask. Verify no Dice regression.
2. Confirm/convert to the **overlapping sigmoid** ET/TC/WT formulation.
3. Add MONAI augmentation: random flips (all axes), intensity shift/scale, gamma, bias-field, light elastic. *Benchmark to change decision: if ET Dice ↑ ≥0.005, keep.*

**Stage 1 — Loss (biggest single lever):**
4. Move to `DiceCELoss(sigmoid=True)` if not already; then add **Focal-Tversky (α=0.7, β=0.3, γ=4/3)** with weight 0.5. Expect ET/TC ↑.
5. Add the **annealed boundary/surface term** (0→~0.5 linear over training) using `HausdorffDTLoss` or an SDM surface loss. Expect HD95 ↓. *Threshold: if HD95 ↓ without Dice loss, keep; if training destabilizes, slow the anneal.*

**Stage 2 — Cheap architecture + training:**
6. Add **deep supervision** heads at 24³/48³ (weights 0.25/0.5/1.0).
7. Add **model EMA** (decay 0.999) and AdamW + cosine schedule with warmup.

**Stage 3 — Targeted modules:**
8. Insert the **3D LKA refinement block** (or full BG-CLKA, §6.1) at the top decoder stage. Ablate LKA-only vs BG-CLKA.
9. Optionally add **bottleneck 3D ASPP** and/or **top-level cross-attention skip**.

**Stage 4 — Consolidation:**
10. Add **connected-component FP removal** (tune min-lesion-volume on validation) for HD95.
11. If still short on ET/TC, try **SAM** and/or **uncertainty-guided skip fusion**.
12. Only if plateaued below 0.90: consider the **Swin/SwinUNETR-V2 encoder** swap.

**Decision benchmarks that change the plan:** if after Stages 0–2 all three regions clear 0.90 and HD95 is acceptable, stop — do not add Stage-3 complexity. If ET remains <0.90, prioritize the LKA block + FP removal + stronger Tversky recall weighting (raise α to 0.75). If WT regresses at any point, the boundary weight or Tversky α is too aggressive — reduce.

---

## 13. Risks and Trade-offs

- **Boundary/surface losses can destabilize or over-shrink regions if introduced too early or weighted too high.** Mitigation: strict annealing (α: 0→0.5), keep DiceCE as anchor, monitor per-class Dice each epoch. This is the highest-risk-if-mishandled item, so it is scheduled after the loss anchor is stable.
- **Focal-Tversky with high α (recall) can inflate false positives**, hurting precision/HD95 on ET. Mitigation: pair with FP-lesion post-processing; tune α on validation.
- **LKA / extra modules add FLOPs and memory** at 96³; mitigate by placing LKA at 48³ or one stage only, use gradient checkpointing (MedNeXt-style) if memory-bound.
- **Cross-attention skips are memory-heavy at high resolution**; restrict to the top level or use EPA-style shared query/key.
- **Combined BraTS 2019+2020 with a random 70/15/15 split risks patient leakage** (2020 is a superset of 2019); ensure de-duplication by patient ID or metrics will be optimistic. This is a data-integrity caveat independent of architecture.
- **>0.98 Dice claims in some papers** (e.g., certain attention-U-Net variants) almost certainly reflect optimistic internal splits, not challenge-server evaluation — do not treat as targets.
- **Encoder swap (Swin/Mamba) is high-effort/high-risk** and may need more data or pretraining to beat the current model; reserved as last resort.
- **SAM roughly doubles per-step cost**; only worthwhile if generalization/boundary gains justify it on your hardware.

---

## 14. References (venue, year)

- Hatamizadeh et al. **UNETR: Transformers for 3D Medical Image Segmentation.** WACV 2022.
- Hatamizadeh et al. **Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI.** MICCAI BrainLes 2021 / arXiv:2201.01266, 2022.
- He, Nath, Yang, Tang, Myronenko, Xu. **SwinUNETR-V2: Stronger Swin Transformers with Stagewise Convolutions.** MICCAI 2023.
- Xing et al. **SegMamba: Long-range Sequential Modeling Mamba for 3D Medical Image Segmentation.** MICCAI 2024 / arXiv:2401.13560.
- Roy et al. **MedNeXt: Transformer-Driven Scaling of ConvNets for Medical Image Segmentation.** MICCAI 2023 / arXiv:2303.09975.
- Isensee et al. **nnU-Net for Brain Tumor Segmentation.** MICCAI BrainLes 2020 / arXiv:2011.00848.
- Futrega et al. **Optimized U-Net for Brain Tumor Segmentation.** MICCAI BrainLes 2021 / arXiv:2110.03352.
- Kervadec et al. **Boundary loss for highly unbalanced segmentation.** MIDL 2019; Medical Image Analysis 67, 2021.
- Karimi & Salcudean. **Reducing the Hausdorff Distance in Medical Image Segmentation with CNNs.** IEEE TMI 2019 / arXiv:1904.10030.
- Celaya et al. **A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation.** 2023 / arXiv:2302.03868 (and Weighted Normalized Boundary Loss).
- Abraham & Khan. **A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation.** IEEE ISBI 2019 / arXiv:1810.07842.
- Yeung et al. **Unified Focal loss.** Computerized Medical Imaging and Graphics, 2021.
- Shit et al. **clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.** CVPR 2021.
- Leng et al. **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions.** ICLR 2022 / arXiv:2204.12511.
- Berman, Rannen Triki, Blaschko. **The Lovász-Softmax loss.** CVPR 2018 / arXiv:1705.08790.
- Li, Nan, Del Ser, Yang. **Large-Kernel Attention for 3D Medical Image Segmentation.** Cognitive Computation 16(4):2063–2077, 2024 (DOI 10.1007/s12559-023-10126-7).
- Azad et al. **Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation.** WACV 2024 / arXiv:2309.00121.
- Ouyang et al. **Efficient Multi-Scale Attention Module with Cross-Spatial Learning (EMA).** ICASSP 2023.
- Yang et al. **SimAM: A Simple, Parameter-Free Attention Module for CNNs.** ICML 2021.
- Petit et al. **U-Net Transformer: Self and Cross Attention for Medical Image Segmentation.** 2021 / arXiv:2103.06104.
- Ates et al. **Dual Cross-Attention for Medical Image Segmentation.** Engineering Applications of AI, 2023.
- Shaker et al. **UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation.** arXiv:2212.04497.
- Foret et al. **Sharpness-Aware Minimization for Efficiently Improving Generalization.** ICLR 2021.
- Isensee et al. **No New-Net.** MICCAI BrainLes 2018 / arXiv:1809.10483.
- Myronenko et al. **Auto3DSeg / SegResNet for BraTS.** arXiv:2111.00742, 2510.25058.
- UPMAD-Net. arXiv:2505.03494, 2025 (preprint; ET 90.00/TC 92.42/WT 94.72 on BraTS2021 val — cited with caveat).

*Caveat on numbers:* several per-class BraTS figures for UNETR/MedNeXt above are third-party reproductions rather than the originating paper's headline metric, and are flagged as such in the text; BraTS 2023/2024 use lesion-wise Dice/HD95, which are stricter and not directly comparable to older region-wise numbers.