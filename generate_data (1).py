"""
StreamFlix A/B Test Dataset Generator
Generates 10,000 users across 12 calibrated A/B tests.
Each test has a realistic positive lift for Variant B (2–8% higher conversion).
"""

import pandas as pd
import numpy as np

np.random.seed(2024)

# ── 12 tests: (name, base_conv_A, absolute_lift_B, duration_days)
# Lift is the ABSOLUTE increase in conversion rate for B
TESTS = [
    ("thumbnail_style",       0.080, 0.040, 28),
    ("autoplay_preview",      0.120, 0.030, 21),
    ("email_subject_line",    0.055, 0.025, 35),
    ("homepage_layout",       0.150, 0.020, 14),
    ("recommendation_algo",   0.180, 0.035, 28),
    ("signup_flow",           0.220, 0.050, 21),
    ("push_notification",     0.090, 0.018, 42),
    ("pricing_page_copy",     0.110, 0.040, 28),
    ("watchlist_cta",         0.140, 0.028, 21),
    ("continue_watching_ui",  0.160, 0.022, 14),
    ("search_results_layout", 0.100, 0.035, 35),
    ("trailer_autoplay",      0.130, 0.060, 28),
]

DEVICES   = ["mobile", "desktop", "tablet", "tv"]
D_WEIGHTS = [0.45, 0.35, 0.12, 0.08]
COUNTRIES = ["US", "UK", "CA", "AU", "DE", "FR", "BR", "IN"]
C_WEIGHTS = [0.35, 0.15, 0.10, 0.08, 0.10, 0.08, 0.07, 0.07]

records = []
user_id = 1

users_per_test    = 10000 // 12   # ~833
users_per_variant = users_per_test // 2  # ~416

for test_idx, (test_name, conv_a, lift_b, duration) in enumerate(TESTS):
    test_id = f"TEST_{test_idx+1:02d}"
    conv_b  = conv_a + lift_b   # B is strictly better

    for variant, true_rate in [("A", conv_a), ("B", conv_b)]:
        for _ in range(users_per_variant):
            device  = np.random.choice(DEVICES,   p=D_WEIGHTS)
            country = np.random.choice(COUNTRIES, p=C_WEIGHTS)

            # Day follows a Beta distribution — most conversions happen early
            day = int(np.random.beta(1.8, 4.0) * duration) + 1
            day = max(1, min(day, duration))

            # Small per-device noise (±1%)
            dev_mod = {"mobile":-0.010, "desktop":+0.010,
                       "tablet": 0.000, "tv":      0.000}[device]

            p = np.clip(true_rate + dev_mod + np.random.normal(0, 0.004), 0.01, 0.95)
            converted = int(np.random.random() < p)

            # Converters spend longer on page
            if converted:
                time_on_page = int(np.clip(np.random.lognormal(5.5, 0.55), 60, 3600))
            else:
                time_on_page = int(np.clip(np.random.lognormal(4.1, 0.75), 5, 1800))

            records.append({
                "user_id":      user_id,
                "test_id":      test_id,
                "test_name":    test_name,
                "variant":      variant,
                "day":          day,
                "converted":    converted,
                "time_on_page": time_on_page,
                "device":       device,
                "country":      country,
            })
            user_id += 1

df = pd.DataFrame(records)
df.to_csv("/home/claude/ab_test_project/ab_test_data.csv", index=False)

print(f"Dataset: {len(df):,} rows × {len(df.columns)} cols")
print(f"Tests: {df['test_id'].nunique()} | Users: {df['user_id'].nunique():,}")
print(f"\nConversion rates per test:")
for tid in sorted(df['test_id'].unique()):
    sub = df[df['test_id']==tid]
    ca = sub[sub['variant']=='A']['converted'].mean()
    cb = sub[sub['variant']=='B']['converted'].mean()
    print(f"  {tid}: A={ca:.3f}  B={cb:.3f}  lift={cb-ca:.3f} (+{(cb-ca)/ca*100:.1f}%)")
