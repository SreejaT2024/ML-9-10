# EndoVision Research Dashboard

Interactive Streamlit research interface for the binary polyp classification project.

## Quick Launch Command

Run from the project root (`d:\ML-PROJECT`):

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Or, if `streamlit` is directly on your PATH:

```bash
streamlit run dashboard/app.py
```

Local URL: `http://localhost:8501`

---

## Features & Sections

1. **Project Overview:** High-level executive summary, key test metrics (93.75% accuracy, 93.50% sensitivity, 94.00% specificity), and dataset split structure (1,280 train / 320 val / 400 test).
2. **Live Image Prediction:** Upload any endoscopy still (JPG/PNG) or select preset samples. The app applies the deterministic ScopeGuide corner mask, resizes, normalizes, and runs inference through the locked ScopeGuide-mitigated EfficientNet-B0 backbone.
3. **Test Model Evaluation:** Interactive Plotly heatmap of the final test confusion matrix (TN=188, FP=12, FN=13, TP=187) and clinical error-mode explanations.
4. **Multi-Model Benchmark:** Side-by-side validation comparison between Handcrafted HistGradientBoosting, Original EfficientNet-B0 Baseline, and ScopeGuide-Mitigated EfficientNet-B0.
5. **Methodology & Biases:** End-to-end pipeline visualization, disclosure of parallel active-contour/handcrafted feature extraction branches, and clinical/dataset source bias disclosures.
