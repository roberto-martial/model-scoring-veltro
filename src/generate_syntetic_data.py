"""
Synthetic dataset generator: Veltro deal-scoring prototype.

Generates fictional lower-middle-market company profiles ($2M-$50M revenue)
with a binary label (is_attractive_target) built from a weighted, documented
combination of features + noise. Weights are explicit assumptions, not facts -
see README section on methodology for how to talk about this honestly.

Run: python generate_synthetic_data.py
Output: veltro_synthetic_deals.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

INDUSTRIES = {
    "Business services": 0.18,
    "Healthcare services": 0.15,
    "Specialty manufacturing": 0.15,
    "Software": 0.12,
    "Consumer products": 0.12,
    "Industrials": 0.13,
    "Distribution/logistics": 0.10,
    "Other": 0.05,
}

# Rough per-industry EBITDA margin center (mean, std)
MARGIN_BY_INDUSTRY = {
    "Business services": (0.17, 0.05),
    "Healthcare services": (0.14, 0.05),
    "Specialty manufacturing": (0.12, 0.04),
    "Software": (0.22, 0.08),
    "Consumer products": (0.10, 0.05),
    "Industrials": (0.11, 0.04),
    "Distribution/logistics": (0.08, 0.03),
    "Other": (0.12, 0.05),
}

# Rough per-industry recurring revenue % center
RECURRING_BY_INDUSTRY = {
    "Business services": (0.45, 0.15),
    "Healthcare services": (0.35, 0.15),
    "Specialty manufacturing": (0.15, 0.10),
    "Software": (0.75, 0.15),
    "Consumer products": (0.10, 0.08),
    "Industrials": (0.15, 0.10),
    "Distribution/logistics": (0.20, 0.10),
    "Other": (0.25, 0.15),
}

GEOGRAPHIES = ["Quebec", "Ontario", "Northeast US", "Midwest US", "Southeast US", "West Coast US", "Other"]
OWNERSHIP = ["Founder-owned", "Family-owned", "Already PE-backed", "Other"]

def sample_industry():
    names = list(INDUSTRIES.keys())
    weights = list(INDUSTRIES.values())
    return np.random.choice(names, p=weights)

rows = []
for i in range(N):
    industry = sample_industry()

    # Revenue: log-normal, bounded to $2M-$50M
    revenue = np.clip(np.random.lognormal(mean=15.5, sigma=0.7), 2_000_000, 50_000_000)

    # EBITDA margin: per-industry normal, clipped to sane range
    m_mean, m_std = MARGIN_BY_INDUSTRY[industry]
    ebitda_margin = np.clip(np.random.normal(m_mean, m_std), 0.01, 0.45)

    # Revenue growth: normal, slight right skew, this is a key signal feature
    revenue_growth_3yr = np.clip(np.random.normal(0.08, 0.15), -0.30, 0.80)

    # Customer concentration: beta skewed low, occasional high-concentration outliers
    customer_concentration = np.random.beta(2, 6)

    # Recurring revenue %: per-industry
    r_mean, r_std = RECURRING_BY_INDUSTRY[industry]
    recurring_revenue_pct = np.clip(np.random.normal(r_mean, r_std), 0.0, 0.95)

    # Years in operation: log-normal, floor at 3
    years_in_operation = max(3, np.random.lognormal(mean=2.5, sigma=0.5))

    geography = np.random.choice(GEOGRAPHIES)
    ownership_type = np.random.choice(OWNERSHIP, p=[0.45, 0.25, 0.15, 0.15])

    # Management depth: 1-5 proxy score, correlated loosely with size and years
    management_depth_score = np.clip(
        np.random.normal(1.5 + (np.log(revenue) - 14) * 0.6 + years_in_operation * 0.03, 0.8),
        1, 5
    )

    rows.append({
        "company_id": f"C{i+1:05d}",
        "industry": industry,
        "geography": geography,
        "ownership_type": ownership_type,
        "annual_revenue": round(revenue, -3),
        "ebitda_margin": round(ebitda_margin, 4),
        "revenue_growth_3yr": round(revenue_growth_3yr, 4),
        "customer_concentration": round(customer_concentration, 4),
        "recurring_revenue_pct": round(recurring_revenue_pct, 4),
        "years_in_operation": round(years_in_operation, 1),
        "management_depth_score": round(management_depth_score, 2),
    })

df = pd.DataFrame(rows)


def normalize(s):
    return (s - s.min()) / (s.max() - s.min())


# Documented label weights - these are assumptions, not ground truth.
# See README for how to talk about this honestly.
score = (
    0.25 * normalize(df["revenue_growth_3yr"])
    + 0.20 * normalize(df["ebitda_margin"])
    + 0.15 * normalize(df["recurring_revenue_pct"])
    + 0.15 * normalize(df["management_depth_score"])
    - 0.15 * normalize(df["customer_concentration"])
    + 0.10 * normalize(df["years_in_operation"])
    + np.random.normal(0, 0.08, size=N)  # noise so it's not perfectly separable
)

# Threshold set so roughly 30% of companies are "attractive targets" -
# adjust if you want a different class balance
threshold = np.quantile(score, 0.70)
df["is_attractive_target"] = (score > threshold).astype(int)

df.to_csv("veltro_synthetic_deals.csv", index=False)
print(f"Generated {len(df)} rows. Positive class rate: {df['is_attractive_target'].mean():.2%}")
print(df.head())