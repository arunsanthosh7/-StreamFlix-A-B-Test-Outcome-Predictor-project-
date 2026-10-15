# 🎬 A/B Test Outcome Predictor — StreamFlix

> **Predict experiment winners 43% faster using Bayesian inference.**  
> Built by **Arunsanthosh** — Data Analyst  
> Python · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Scikit-learn

---

## Project Summary

Traditional A/B testing waits until p < 0.05 — which can take **4–6 weeks**. Half your users experience the worse variant the entire time. Engineers wait. Product is blocked.

This project builds a **Bayesian prediction system** that tracks confidence daily and stops the moment it reaches **95% certainty** — no arbitrary sample size required.

Tested across **12 streaming-service experiments** with **~10,000 users**, the system delivers decisions **43% faster** and saves an average of **~3 weeks per test**.

---

## Results

| Metric | Value |
|--------|-------|
| Speed improvement | **43.2% faster** than frequentist |
| Average days saved | **20.8 days (~3 weeks)** per test |
| Tests with early stopping | **9 of 12** |
| Decision accuracy | **12 / 12** correct |
| Estimated annual saving (20-test portfolio) | **416 experiment-days freed** |

---

## The Analyst

**Arunsanthosh** — Data Analyst

- 🔗 [LinkedIn](https://linkedin.com/in/arunsanthosh07)
- 💻 [GitHub](https://github.com/arunsanthosh7)

---

## How It Works

### Traditional Method (Frequentist)
Runs until a chi-squared test returns **p < 0.05**. Binary outcome at the end. No intermediate signal.

### This Project (Bayesian)
Updates a **Beta distribution** every day as conversions come in. Computes **P(B > A)** via Monte Carlo sampling. Stops when confidence ≥ 95%.

```
Prior:      Beta(1, 1)   → no opinion yet
After data: Beta(1 + conversions, 1 + non-conversions)
```

**P(B > A)** = draw 60,000 hypothetical rates from each distribution, count how often B's rate beats A's. Simple, fast, interpretable.

| | Bayesian | Frequentist |
|--|---------|------------|
| **Decision rule** | P(B > A) ≥ 95% | p-value < 0.05 |
| **Statistical test** | Beta-Binomial | Chi-squared |
| **Updates** | Every day | End of experiment |
| **Readable as** | "95% confident B wins" | "5% chance this is noise" |
| **Early stopping** | ✅ Built-in | ❌ Requires correction |

---

## Dataset

**File:** `ab_test_data.csv` — 9,984 rows × 9 columns

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `test_id` | Experiment ID (TEST_01 – TEST_12) |
| `test_name` | Descriptive test name |
| `variant` | A (control) or B (treatment) |
| `day` | Day the user entered the experiment |
| `converted` | 1 = converted, 0 = did not |
| `time_on_page` | Seconds spent on page |
| `device` | mobile / desktop / tablet / tv |
| `country` | US, UK, CA, AU, DE, FR, BR, IN |

**12 experiments simulated:**

| Test | Variant A Conv. | Variant B Conv. | Lift |
|------|----------------|----------------|------|
| Thumbnail Style | 8.0% | 12.5% | +56% |
| Autoplay Preview | 12.0% | 15.5% | +29% |
| Email Subject Line | 5.5% | 9.0% | +64% |
| Homepage Layout | 15.0% | 17.5% | +17% |
| Recommendation Algo | 18.0% | 21.5% | +19% |
| Signup Flow | 22.0% | 28.0% | +27% |
| Push Notification | 9.0% | 10.8% | +20% |
| Pricing Page Copy | 11.0% | 15.5% | +41% |
| Watchlist CTA | 14.0% | 17.2% | +23% |
| Continue Watching UI | 16.0% | 19.0% | +19% |
| Search Results Layout | 10.0% | 14.0% | +40% |
| Trailer Autoplay | 13.0% | 19.5% | +50% |

---

## Visualizations

**Chart 1 — Executive Dashboard**  
KPI cards (speed, days saved, weeks freed, accuracy) · Decision day bar chart · Lift vs. savings scatter · % faster distribution · Final P(B > A) per test.

**Chart 2 — Real-Time Probability Tracking**  
P(B > A) rising day by day for 4 featured tests. Green line = Bayesian decision. Red = Frequentist. The gap between them is the time saved.

**Chart 3 — Beta Distribution Belief Updates**  
Side-by-side Beta PDFs for Variant A and B at multiple snapshots. Wide = uncertain. Narrow and separated = confident. Bayesian stops the moment separation is clear.

**Chart 4 — Business Impact Timeline**  
Horizontal timeline for all 12 tests + cumulative days saved curve showing total portfolio efficiency gain.

---

## Repository Structure

```
streamflix-ab-predictor/
│
├── generate_data.py                       # Dataset simulation
├── bayesian_model.py                      # Bayesian engine + charts
├── StreamFlix_AB_Test_Predictor.ipynb     # Full analysis notebook (22 cells)
│
├── ab_test_data.csv                       # 9,984 rows × 9 cols
├── summary_results.csv                    # Per-test comparison
│
├── charts/
│   ├── chart01_dashboard.png
│   ├── chart02_probability_evolution.png
│   ├── chart03_beta_distributions.png
│   └── chart04_business_impact.png
│
├── requirements.txt
└── README.md
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Generate the dataset:**
```bash
python generate_data.py
```

**Run the full analysis and generate charts:**
```bash
python bayesian_model.py
```

**Open the notebook:**
```bash
jupyter notebook StreamFlix_AB_Test_Predictor.ipynb
```

---

## Requirements

```
pandas>=2.1.0
numpy>=1.26.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

Python 3.10+ recommended.

---

## Resume Bullets

- **Engineered** a Bayesian A/B test prediction system using Beta-Binomial conjugate priors on **~10,000 users** across **12 experiments**, achieving **43% faster** decisions vs. traditional p<0.05 frequentist methods with zero accuracy loss.

- **Reduced** average experiment runtime by **20.8 days (~3 weeks)** per test via Bayesian early stopping at **95% confidence**, enabling **9/12 tests** to conclude early — equivalent to **12 additional experiments/year** for a 20-test portfolio.

- **Built** an end-to-end A/B analytics pipeline in Python (Pandas, SciPy, Matplotlib) including daily probability dashboards, Beta distribution belief-update visualizations, and a frequentist chi-squared baseline across 12 streaming-service experiments.

---

## Limitations

**Simulated data** — Users were generated synthetically with known ground truth. In production, true conversion rates are unknown and lift estimates carry more uncertainty.

**Single metric per test** — The model optimises for one conversion event. Multi-metric experiments require more sophisticated approaches (multi-armed bandits, hierarchical models).

**No multiple-testing correction** — Running 12 tests simultaneously inflates family-wise error rate. In production, apply Bonferroni correction or a hierarchical Bayesian model.

---

## References

- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall.
- Kohavi, Tang & Xu (2020). *Trustworthy Online Controlled Experiments*. Cambridge.
- Stucchio, C. (2015). *Bayesian A/B Testing at VWO*. VWO Whitepaper.
- VanderPlas, J. (2014). *Frequentism and Bayesianism: A Python-driven Primer*. arXiv.

---

## License

MIT — use freely, attribution appreciated.

---

*Python · SciPy · Pandas · Matplotlib · Seaborn · Beta-Binomial Bayesian Inference*
