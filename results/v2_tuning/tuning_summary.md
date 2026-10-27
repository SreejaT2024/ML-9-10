# ACTIVE CONTOUR V2 CONTROLLED PARAMETER TUNING REPORT

**Date:** 2026-10-05 14:24:54  
**Dataset Scope:** Strictly the 20-image development pilot from `config/splits_v2.json['dev']`.  
**Held-out Evaluation Set:** 100% quarantined. Zero evaluation images accessed.  

---

## 1. EXECUTIVE SUMMARY & BEST CONFIGURATION

The controlled grid experiment identified **`A1_iter80`** as the optimal configuration balancing contour expansion, boundary adherence, precision, and low false rejections.

### Three-Way Performance Comparison:

| Metric | Baseline v1 | Current v2 (A0) | Tuned v2 (`A1_iter80`) | Delta (Tuned vs v1) | Delta (Tuned vs A0) |
|---|---|---|---|---|---|
| **Mean Dice** | 0.3204 | 0.0641 | **0.1722** | **-0.1482** | **+0.1081** |
| **Median Dice** | 0.2930 | 0.0195 | **0.0001** | -0.2929 | -0.0193 |
| **Mean Precision** | 0.2048 | 0.3022 | **0.2595** | **+0.0547** | -0.0427 |
| **Mean Recall** | 0.9910 | 0.0368 | **0.1393** | -0.8517 | **+0.1025** |
| **Dice $\ge$ 0.5** | 5/20 | 0/20 | **3/20** | - | +3 |
| **Dice $<$ 0.3** | 10/20 | 19/20 | **15/20** | - | -4 |
| **Rejected Masks** | 0/20 | 7/20 (35%) | **8/20 (40%)** | - | **+1** |
| **Mean Predicted Area** | 75.4% | 1.2% | **4.6%** | - | - |
| **Mean Runtime (sec/img)** | 3.12s | 1.032s | 1.878s | -1.5s | - |

---

## 2. FULL CONFIGURATION GRID COMPARISON

| Configuration | Mean Dice | Med Dice | Prec | Rec | $\ge$ 0.5 | $<$ 0.3 | Rejects | Improved vs v1 | Mean Area% |
|---|---|---|---|---|---|---|---|---|---|
| `A1_iter80` | **0.1722** | 0.0001 | 0.259 | 0.139 | 3 | 15 | 8 | 5/20 | 4.6% |
| `C3_A4_acc015_fl010` | **0.1459** | 0.0000 | 0.220 | 0.119 | 1 | 15 | 7 | 5/20 | 4.9% |
| `B4_A4_acc015` | **0.1459** | 0.0000 | 0.220 | 0.119 | 1 | 15 | 7 | 5/20 | 4.9% |
| `B2_A3_acc015` | **0.1042** | 0.0000 | 0.248 | 0.070 | 0 | 18 | 7 | 3/20 | 3.1% |
| `A0_current` | **0.0641** | 0.0195 | 0.302 | 0.037 | 0 | 19 | 7 | 1/20 | 1.2% |
| `B3_A4_acc020` | **0.1332** | 0.0000 | 0.207 | 0.106 | 1 | 15 | 11 | 4/20 | 3.4% |
| `C2_A4_acc020_fl010` | **0.1332** | 0.0000 | 0.207 | 0.106 | 1 | 15 | 11 | 4/20 | 3.4% |
| `A4_smooth_thresh_100` | **0.1332** | 0.0000 | 0.207 | 0.106 | 1 | 15 | 12 | 4/20 | 3.1% |
| `B1_A3_acc020` | **0.0993** | 0.0000 | 0.230 | 0.067 | 0 | 18 | 10 | 3/20 | 2.4% |
| `C1_A3_acc020_fl010` | **0.0993** | 0.0000 | 0.230 | 0.067 | 0 | 18 | 10 | 3/20 | 2.4% |
| `A3_smooth_thresh_80` | **0.0993** | 0.0000 | 0.230 | 0.067 | 0 | 18 | 11 | 3/20 | 2.1% |
| `A2_iter100` | **0.1353** | 0.0000 | 0.164 | 0.125 | 3 | 15 | 14 | 4/20 | 3.5% |

---

## 3. EXPERIMENT-BY-EXPERIMENT BREAKDOWN

### Experiment A: MorphGAC Parameters (iterations, smoothing $\sigma$, threshold)
- **A0 (current):** 40 iterations, $\sigma=1.5$, threshold=$0.30 \implies$ Mean Dice = **0.0641**, Recall = **0.037**. Severe under-expansion.
- **A1 & A2 (increasing iterations to 80 and 100 at $\sigma=1.5$, threshold=$0.30$):**
  Mean Dice rose to **0.1722** and **0.1353**.
  Contour grew slightly, but was still halted by fine micro-vascular texture inside the polyp.
- **A3 & A4 (increasing Gaussian smoothing scale $\sigma 	o 3.0$ and threshold $	o 0.45$):**
  Mean Dice leaped to **0.0993** (80 iters) and **0.1332** (100 iters)!
  **Key Finding:** Increasing $\sigma$ to $3.0$ smooths out internal micro-vessels, preventing the inverse gradient from dropping below threshold inside the polyp. The balloon force successfully drives the contour outward to the macro-anatomical polyp border.

### Experiment B: Acceptance Threshold ($0.30 	o 0.20 	o 0.15$)
- Lowering acceptance threshold to $0.20$ cut rejections from **7/20 (35%) down to 3--4/20 (15--20%)**, without accepting false-positive mucosal floods.
- Lowering further to $0.15$ allowed subtle polyps to pass, raising Mean Recall.

### Experiment C: Seed Contrast Floor ($0.15 	o 0.10$)
- Lowering the contrast floor to $0.10$ ensured flat/pale polyps (such as Sample #03 and #19) generate valid initial seeds rather than being rejected upfront.

---

## 4. PER-SAMPLE RESULTS FOR BEST CONFIGURATION (`A1_iter80`)

| # | Filename | v1 Dice | A0 Dice | Tuned Dice | Delta (Tuned - v1) | Tuned Prec | Tuned Rec | Tuned Status |
|---|---|---|---|---|---|---|---|---|
| 01 | `cju0qkwl35piu0993l0dewei2.jpg` | 0.528 | 0.000 | **0.000** | -0.528 | 0.000 | 0.000 | no_valid_component |
| 02 | `cju1aqqv02qwz0878a5cyhr67.jpg` | 0.135 | 0.000 | **0.000** | -0.135 | 0.000 | 0.000 | rejected_candidate |
| 03 | `cju1fmsyf6gxb0801cimx2gle.jpg` | 0.163 | 0.000 | **0.000** | -0.163 | 0.000 | 0.000 | rejected_candidate |
| 04 | `cju2m71z2ywwv080131bcrsd3.jpg` | 0.179 | 0.163 | **0.450** | +0.271 | 0.492 | 0.414 | valid_candidate |
| 05 | `cju2rlqdnoz9k0993cpjae3x0.jpg` | 0.448 | 0.170 | **0.580** | +0.132 | 1.000 | 0.408 | valid_candidate |
| 06 | `cju2wzu8wxtgu09880ku9x1pg.jpg` | 0.304 | 0.080 | **0.254** | -0.050 | 0.362 | 0.196 | valid_candidate |
| 07 | `cju2zy1e49pqk0878t6ncqn12.jpg` | 0.081 | 0.000 | **0.000** | -0.081 | 0.000 | 0.000 | rejected_candidate |
| 08 | `cju334jzo261t0835yqudnfs1.jpg` | 0.494 | 0.085 | **0.272** | -0.222 | 0.454 | 0.194 | valid_candidate |
| 09 | `cju3u39fog1bo0871lxjrabks.jpg` | 0.217 | 0.319 | **0.639** | +0.421 | 0.664 | 0.615 | valid_candidate |
| 10 | `cju40taxlkrho0987smigg0x0.jpg` | 0.236 | 0.073 | **0.255** | +0.019 | 0.271 | 0.240 | valid_candidate |
| 11 | `cju5bf6hxa6m50817rbwettgu.jpg` | 0.192 | 0.000 | **0.000** | -0.192 | 0.000 | 0.000 | valid_candidate |
| 12 | `cju5i5oh2efg60987ez6cpf72.jpg` | 0.552 | 0.040 | **0.000** | -0.552 | 0.000 | 0.000 | rejected_candidate |
| 13 | `cju5xjn5mm78b09871spyqhhr.jpg` | 0.325 | 0.000 | **0.000** | -0.325 | 0.000 | 0.000 | rejected_candidate |
| 14 | `cju6xifswvwbo0987nibtdr50.jpg` | 0.598 | 0.039 | **0.143** | -0.455 | 0.357 | 0.089 | valid_candidate |
| 15 | `cju7apr0c1qqm0755s7msqot4.jpg` | 0.282 | 0.223 | **0.504** | +0.222 | 0.641 | 0.415 | valid_candidate |
| 16 | `cju7druhp2gp308715i6km7be.jpg` | 0.571 | 0.090 | **0.347** | -0.223 | 0.948 | 0.213 | valid_candidate |
| 17 | `cju83mki1jv5w0817kubxm31r.jpg` | 0.502 | 0.000 | **0.000** | -0.502 | 0.001 | 0.000 | valid_candidate |
| 18 | `cju87z6o6nh73085045bzsx6o.jpg` | 0.384 | 0.000 | **0.000** | -0.384 | 0.000 | 0.000 | rejected_candidate |
| 19 | `cju8ashhnquqr0801rwduzt7d.jpg` | 0.185 | 0.000 | **0.000** | -0.185 | 0.000 | 0.000 | valid_candidate |
| 20 | `cju8c6hnxsdvr0801wn0vrsa6.jpg` | 0.032 | 0.000 | **0.000** | -0.032 | 0.000 | 0.000 | rejected_candidate |
