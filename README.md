# Retail Data Science: Strategic Plan and Feasibility Analysis

Week 1 deliverable: a strategic plan and feasibility study for a Python-based retail data science programme, plus a small, runnable code base that demonstrates the methods described in the report.

**Report:** [`docs/Retail_Data_Science_Strategic_Plan.docx`](docs/Retail_Data_Science_Strategic_Plan.docx)

## What the report covers
- Research on 2026 retail and AI trends, with a source register and rationale
- Business objectives and a scored portfolio of eight data science initiatives (I1 to I8)
- Python technology stack and reference architecture
- Methodology, validation principles and challenges with solutions
- Technical, financial, organisational and legal feasibility (3-year business case)
- Risk register with mitigations
- 27 KPIs in five tiers, each justified
- Roadmap with stage gates

## What the code covers
| Module | Report section | Purpose |
|---|---|---|
| `src/retail_ds/data.py` | 6.2 | Synthetic transactions and daily sales (no private data needed) |
| `src/retail_ds/segmentation.py` | I2, 6.3 | RFM features and K-Means customer segments |
| `src/retail_ds/forecasting.py` | I1, 6.3 | LightGBM demand forecast, rolling-origin validation, WAPE and bias vs seasonal-naive baseline |
| `src/retail_ds/experiments.py` | 6.3 | A/B test sample-size and significance helpers |
| `src/retail_ds/business_case.py` | 7.2 | 3-year NPV, ROI, payback, break-even and sensitivity |
| `src/retail_ds/kpis.py` | 9 | Formulas for core KPIs |

## Quick start
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q            # run tests
python scripts/run_demo.py     # run the end-to-end demo
```
The demo prints results and saves `segment_profile.csv`, `forecast_eval.csv` and `npv_sensitivity.png` to `outputs/`.

## Example results (synthetic data)
- Forecast WAPE about 0.20 to 0.21 vs about 0.30 to 0.32 for the seasonal-naive baseline
- Business case: run-rate benefit $5.65M, NPV $4.49M at 10%, payback in Year 2, break-even at 44% of modelled benefit

## Project structure
```
docs/       Word report
src/        retail_ds package
scripts/    run_demo.py
tests/      pytest tests
data/       place real data here (git-ignored)
outputs/    generated results (git-ignored)
```

## Important notes
- All data in this repository is **synthetic**, and all financial inputs are **illustrative assumptions** for a reference retailer. Replace them with real figures before drawing business conclusions.
- Code is a starter kit, not a production system. See Section 5 and 6 of the report for the production architecture.

## License
MIT (see `LICENSE`).
