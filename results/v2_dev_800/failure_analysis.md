# Active Contour V2 Failure-Case Analysis (800 Development Images)

Systematic categorization of segmentation failures and boundary limitations across the 800 development frames.

## 1. False Positives (Accepted Predictions with Dice < 0.05): 101 cases (12.6%)

Occurs when non-neoplastic structures produce prominent vascular or textural contrast above the acceptance threshold (0.30).

| Index | Filename | Confidence | Pred Pixels | GT Pixels | Info Status | Primary Mechanism |
|---|---|---|---|---|---|---|
| 479 | `cju5x28nzm7t907558ocq4bt7.jpg` | 0.558 | 14,841 | 45,160 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 527 | `cju6z2616wqbk07555bvnuyr1.jpg` | 0.449 | 18,637 | 27,616 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 310 | `cju358pwtdby20878cg7nm0np.jpg` | 0.425 | 19,010 | 13,707 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 694 | `cju88msmoo3470817m441j4sg.jpg` | 0.420 | 29,220 | 49,828 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 477 | `cju5wuhm1lwm40987vugqn3vv.jpg` | 0.415 | 20,294 | 28,308 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 135 | `cju2otvvv0l7z0855x7we8cb0.jpg` | 0.405 | 25,334 | 22,156 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 274 | `cju32l161bi1v07990vm376in.jpg` | 0.404 | 31,463 | 10,582 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 633 | `cju7fnfv02tt90801djnix9m8.jpg` | 0.404 | 29,708 | 6,249 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 668 | `cju85je7vlht70817c9jcjwi4.jpg` | 0.397 | 91,523 | 441,850 | valid_candidate | Normal mucosal fold vascularity / specular residual |
| 790 | `cjyzuio1qgh040763k56deohv.jpg` | 0.392 | 42,128 | 5,183 | valid_candidate | Normal mucosal fold vascularity / specular residual |

## 2. Under-Segmentation (Accepted Predictions with Recall < 0.25): 208 cases (26.0%)

Occurs predominantly on large, flat, or sessile lesions where the single localized seed expands into only one prominent nodule rather than engulfing the full lesion footprint.

| Index | Filename | Dice | Precision | Recall | GT Area % | Pred Area % |
|---|---|---|---|---|---|---|
| 779 | `cju8dn0c3u2v50801k8rvq02f.jpg` | 0.188 | 0.351 | 0.128 | 31.4% | 11.5% |
| 358 | `cju40poe4kt9s0755f9cnm3h5.jpg` | 0.186 | 0.364 | 0.125 | 27.8% | 9.5% |
| 196 | `cju2uwz9f5yf1085506cfamfx.jpg` | 0.190 | 0.364 | 0.128 | 26.3% | 9.2% |
| 659 | `cju84jdl9kv0i0871eog9b3i9.jpg` | 0.295 | 0.450 | 0.220 | 21.6% | 10.5% |
| 693 | `cju88l66no10s0850rsda7ej1.jpg` | 0.178 | 1.000 | 0.098 | 81.2% | 7.9% |
| 780 | `cju8doa16u5gh0818w1ywda3q.jpg` | 0.243 | 1.000 | 0.138 | 71.9% | 10.0% |
| 631 | `cju7flevb2wii08188otgs9p2.jpg` | 0.268 | 1.000 | 0.155 | 66.0% | 10.2% |
| 505 | `cju6vifjlv55z0987un6y4zdo.jpg` | 0.252 | 1.000 | 0.144 | 61.5% | 8.9% |
| 259 | `cju30ywtc0oar0835bp2en7ec.jpg` | 0.072 | 0.087 | 0.062 | 13.9% | 9.9% |
| 680 | `cju87vqa0ndwg0850onjdz7ol.jpg` | 0.186 | 1.000 | 0.102 | 67.3% | 6.9% |

## 3. Rejected Valid Lesions (False Negatives with GT Area > 5,000 px): 283 cases (35.4%)

Frames where legitimate polyps were suppressed due to border penalties or low saliency contrast.

| Index | Filename | Rejection Reason | Border Penalty | GT Pixels | GT Area % |
|---|---|---|---|---|---|
| 624 | `cju7f5ghb2r5s0801chwkxxh9.jpg` | `rejected_candidate` | 0.0 | 367,077 | 29.8% |
| 327 | `cju3uz4o6gr9z0850lhxyxvsj.jpg` | `rejected_candidate` | 0.0 | 298,240 | 20.7% |
| 202 | `cju2x7vw87mu30878hye2ca0m.jpg` | `rejected_candidate` | 0.0 | 291,968 | 20.2% |
| 707 | `cju8a1jtvpt9m081712iwkca7.jpg` | `rejected_candidate` | 0.0 | 269,591 | 18.7% |
| 471 | `cju5woy82m07m08505dmjg7g1.jpg` | `rejected_candidate` | 0.0 | 239,416 | 16.6% |
| 174 | `cju2sszfq3uye0878sucelzk2.jpg` | `rejected_candidate` | 0.0 | 203,729 | 61.7% |
| 59 | `cju1cyjb5qtie0993njqne9m3.jpg` | `rejected_candidate` | 0.0 | 173,688 | 50.7% |
| 6 | `cju0sr5ghl0nd08789uzf1raf.jpg` | `rejected_candidate` | 0.0 | 165,185 | 13.4% |
| 789 | `cjyzufihqquiw0a46jatrbwln.jpg` | `rejected_candidate` | 0.0 | 164,631 | 39.7% |
| 513 | `cju6wi3akvn8r0801px8eligc.jpg` | `rejected_candidate` | 0.0 | 160,689 | 61.4% |

## 4. Summary of Root Limitations and Downstream Mitigation

1. **Single-Seed Localized Topology:** MorphGAC enforces smooth, localized level sets. For polyps spanning >30% of the field, a single seed disk guarantees high precision but bounded recall.
2. **Normal Vascular Folds:** The green-channel CLAHE strongly detects hemoglobin. Normal peristaltic mucosal ridges occasionally present elevated local vascularity, leading to false-positive component scores.
3. **Downstream Feature Pipeline Role:** The Stage A feature extractor extracts color, GLCM texture, and shape descriptors directly from the segmented ROI. The Stage B classifier will utilize multi-parametric feature signatures to readily separate normal mucosal folds from genuine adenomatous polyps.
