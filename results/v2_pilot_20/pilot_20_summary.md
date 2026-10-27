# KVASIR-SEG 20-IMAGE DEVELOPMENT-SET PILOT REPORT

**Execution Date:** 2026-10-05 13:56:15  
**Evaluation Scope:** Strictly 20 evenly-spaced samples from the 800-image development split (`splits_v2.json['dev']`).  
**Quarantine Confirmation:** The 200-image held-out evaluation set was 100% quarantined. Zero evaluation images were accessed.  

---

## 1. EXECUTIVE SUMMARY & METRIC COMPARISON

| Metric | Baseline v1 | Remediated v2 | Delta (v2 - v1) |
|---|---|---|---|
| **Mean Dice** | **0.3204** | **0.0641** | **-0.2563** |
| **Median Dice** | 0.2930 | 0.0195 | -0.2736 |
| **Std Dev Dice** | 0.1768 | 0.0903 | -0.0865 |
| **Min Dice** | 0.0318 | 0.0000 | -0.0318 |
| **Max Dice** | 0.5980 | 0.3190 | -0.2790 |
| **Mean IoU** | 0.2037 | 0.0354 | -0.1683 |
| **Median IoU** | 0.1717 | 0.0099 | -0.1618 |
| **Mean Precision** | **0.2048** | **0.3022** | **+0.0974** |
| **Mean Recall** | **0.9910** | **0.0368** | **-0.9542** |
| **Mean Runtime (sec/img)** | 3.123s | 1.594s | -1.529s |

### Distribution Benchmarks:
- **Dice $\ge$ 0.5:** v1 = `5/20` | v2 = `0/20`
- **Dice $\ge$ 0.7:** v1 = `0/20` | v2 = `0/20`
- **Dice $<$ 0.3:** v1 = `10/20` | v2 = `19/20`

### Pairwise Comparison:
- **Images with v2 Dice > v1 Dice:** **1 / 20** (5.0%)
- **Images with v1 Dice > v2 Dice:** **19 / 20** (95.0%)
- **Images Approximately Unchanged:** **0 / 20** (0.0%)
- **Mean Pairwise Improvement:** **-0.2563 Dice**

---

## 2. AUTOMATIC CASE CATEGORIZATION

- **Best v2 Result:** Sample #09 (`cju3u39fog1bo0871lxjrabks.jpg`) — **Dice = 0.3190** (v1 Dice: 0.2174)
- **Worst v2 Result:** Sample #01 (`cju0qkwl35piu0993l0dewei2.jpg`) — **Dice = 0.0000** (v1 Dice: 0.5283)
- **Largest Improvement:** Sample #09 (`cju3u39fog1bo0871lxjrabks.jpg`) — **Gain: +0.1015** (v1: 0.2174 $	o$ v2: 0.3190)
- **Largest Regression:** Sample #14 (`cju6xifswvwbo0987nibtdr50.jpg`) — **Loss: -0.5591** (v1: 0.5980 $	o$ v2: 0.0389)
- **Typical / Median v2 Sample:** Sample #01 (`cju0qkwl35piu0993l0dewei2.jpg`) — **v2 Dice: 0.0195**

---

## 3. REJECTION & LOW-CONFIDENCE ANALYSIS

Total Rejections in 20 images: **7**

| # | Filename | v2 Status | Confidence | GT Area % | v1 Dice | v2 Dice | Probable Reason |
|---|---|---|---|---|---|---|---|
| 01 | `cju0qkwl35piu0993l0dewei2.jpg` | no_valid_component | 0.00 | 28.3% | 0.528 | 0.000 | Rejection threshold (0.00 < 0.30) |
| 02 | `cju1aqqv02qwz0878a5cyhr67.jpg` | rejected_candidate | 0.16 | 5.6% | 0.135 | 0.000 | Rejection threshold (0.16 < 0.30) |
| 03 | `cju1fmsyf6gxb0801cimx2gle.jpg` | rejected_candidate | 0.09 | 7.9% | 0.163 | 0.000 | Rejection threshold (0.09 < 0.30) |
| 13 | `cju5xjn5mm78b09871spyqhhr.jpg` | rejected_candidate | 0.01 | 16.5% | 0.325 | 0.000 | Rejection threshold (0.01 < 0.30) |
| 18 | `cju87z6o6nh73085045bzsx6o.jpg` | rejected_candidate | 0.10 | 11.7% | 0.384 | 0.000 | Rejection threshold (0.10 < 0.30) |
| 19 | `cju8ashhnquqr0801rwduzt7d.jpg` | rejected_candidate | 0.28 | 7.2% | 0.185 | 0.000 | Rejection threshold (0.28 < 0.30) |
| 20 | `cju8c6hnxsdvr0801wn0vrsa6.jpg` | rejected_candidate | 0.08 | 1.1% | 0.032 | 0.000 | Rejection threshold (0.08 < 0.30) |

---

## 4. SANITY CHECKS & FAILURE MODE FREQUENCY

- **Mask Covers Most of Image (> 55% frame):** v1 = 18/20 | v2 = 0/20
- **Tiny Meaningless Mask (< 0.5% frame):** v1 = 0/20 | v2 = 7/20
- **Severe Mucosal Leakage (Precision < 20%):** v1 = 12/20 | v2 = 10/20
- **Excessive Empty-Mask Rejections:** 7 / 20 (35.0%)

---

## 5. COMPLETE PER-SAMPLE RESULTS

| # | Filename | GT Area% | v1 Dice | v2 Dice | Delta | v1 Prec | v2 Prec | v1 Rec | v2 Rec | v2 Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | `cju0qkwl35piu0993l0dewei2.jpg` | 28.3% | 0.528 | 0.000 | -0.528 | 0.360 | 0.000 | 0.990 | 0.000 | no_valid_component |
| 02 | `cju1aqqv02qwz0878a5cyhr67.jpg` | 5.6% | 0.135 | 0.000 | -0.135 | 0.073 | 0.000 | 1.000 | 0.000 | rejected_candidate |
| 03 | `cju1fmsyf6gxb0801cimx2gle.jpg` | 7.9% | 0.163 | 0.000 | -0.163 | 0.089 | 0.000 | 1.000 | 0.000 | rejected_candidate |
| 04 | `cju2m71z2ywwv080131bcrsd3.jpg` | 8.9% | 0.179 | 0.163 | -0.016 | 0.098 | 0.491 | 1.000 | 0.098 | valid_candidate |
| 05 | `cju2rlqdnoz9k0993cpjae3x0.jpg` | 17.5% | 0.448 | 0.170 | -0.278 | 0.290 | 1.000 | 0.987 | 0.093 | valid_candidate |
| 06 | `cju2wzu8wxtgu09880ku9x1pg.jpg` | 15.9% | 0.304 | 0.080 | -0.225 | 0.180 | 0.364 | 1.000 | 0.045 | valid_candidate |
| 07 | `cju2zy1e49pqk0878t6ncqn12.jpg` | 3.2% | 0.081 | 0.000 | -0.081 | 0.042 | 0.000 | 1.000 | 0.000 | valid_candidate |
| 08 | `cju334jzo261t0835yqudnfs1.jpg` | 20.0% | 0.494 | 0.085 | -0.409 | 0.342 | 0.460 | 0.893 | 0.047 | valid_candidate |
| 09 | `cju3u39fog1bo0871lxjrabks.jpg` | 8.7% | 0.217 | 0.319 | +0.102 | 0.122 | 0.938 | 1.000 | 0.192 | valid_candidate |
| 10 | `cju40taxlkrho0987smigg0x0.jpg` | 10.2% | 0.236 | 0.073 | -0.163 | 0.134 | 0.233 | 0.990 | 0.043 | valid_candidate |
| 11 | `cju5bf6hxa6m50817rbwettgu.jpg` | 8.7% | 0.192 | 0.000 | -0.192 | 0.106 | 0.000 | 1.000 | 0.000 | valid_candidate |
| 12 | `cju5i5oh2efg60987ez6cpf72.jpg` | 29.4% | 0.552 | 0.040 | -0.512 | 0.383 | 0.352 | 0.989 | 0.021 | valid_candidate |
| 13 | `cju5xjn5mm78b09871spyqhhr.jpg` | 16.5% | 0.325 | 0.000 | -0.325 | 0.194 | 0.000 | 1.000 | 0.000 | rejected_candidate |
| 14 | `cju6xifswvwbo0987nibtdr50.jpg` | 33.8% | 0.598 | 0.039 | -0.559 | 0.427 | 0.360 | 0.997 | 0.021 | valid_candidate |
| 15 | `cju7apr0c1qqm0755s7msqot4.jpg` | 13.2% | 0.282 | 0.223 | -0.059 | 0.164 | 0.844 | 1.000 | 0.129 | valid_candidate |
| 16 | `cju7druhp2gp308715i6km7be.jpg` | 32.2% | 0.571 | 0.090 | -0.481 | 0.401 | 1.000 | 0.994 | 0.047 | valid_candidate |
| 17 | `cju83mki1jv5w0817kubxm31r.jpg` | 26.6% | 0.502 | 0.000 | -0.502 | 0.335 | 0.000 | 0.997 | 0.000 | valid_candidate |
| 18 | `cju87z6o6nh73085045bzsx6o.jpg` | 11.7% | 0.384 | 0.000 | -0.384 | 0.238 | 0.000 | 0.983 | 0.000 | rejected_candidate |
| 19 | `cju8ashhnquqr0801rwduzt7d.jpg` | 7.2% | 0.185 | 0.000 | -0.185 | 0.102 | 0.000 | 1.000 | 0.000 | rejected_candidate |
| 20 | `cju8c6hnxsdvr0801wn0vrsa6.jpg` | 1.1% | 0.032 | 0.000 | -0.032 | 0.016 | 0.000 | 1.000 | 0.000 | rejected_candidate |
