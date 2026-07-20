# Buyer's vs. Seller's Market Classifier

Classifying 900+ U.S. metros monthly as Buyer's, Balanced, or Seller's Market using public Zillow data and machine learning — built to give a real estate investment team a scalable way to prioritize which markets to look at.

## Problem

Real estate investment firms need a fast and data-driven way to identify which U.S. metros favor buyers vs. sellers, in order to prioritize where to look for deals. Existing tools like Zillow's Market Heat Index report a single score per metro without translating it into a buy/hold/avoid signal, and manually tracking dozens of indicators across hundreds of metros doesn't scale.

## Data

Five public datasets from [Zillow Research](https://www.zillow.com/research/data/), covering ~900 U.S. metros and 8 years of monthly history:

- Market Heat Index
- Mean Days to Pending
- Share of Listings with a Price Cut
- Sale-to-List Ratio
- Zillow Home Value Index (ZHVI)

Each dataset was reshaped from wide to long format and integrated via SQL joins on `RegionID` + `Date`. Raw source files aren't included in this repo — download them directly from the link above if you want to reproduce the pipeline.

## Approach

1. **Data integration** — unpivoted five wide-format datasets and joined them in SQLite across a composite `RegionID` + `Date` key.
2. **Data quality diagnosis** — traced non-random missingness in two features to Zillow's minimum sales-volume reporting threshold, and identified a data-leakage risk in a third feature by cross-referencing Zillow's own published methodology. Dropped the compromised columns rather than imputing over structural gaps.
3. **Target engineering** — built a 3-class target (Buyer's / Balanced / Seller's Market) using within-month percentile binning, so labels reflect a metro's relative standing rather than the broader housing boom-bust cycle (notably 2020–2022).
4. **Feature engineering** — corrected a feature-target mismatch by converting raw predictors into percentile-rank versions, matching the target's relative framing and removing a systematic classification bias.
5. **Modeling** — trained a multinomial logistic regression classifier on a time-based train/test split, checked multicollinearity via VIF, and benchmarked against unsupervised K-means clustering.
6. **Validation** — interpreted coefficients as odds ratios, then validated performance on a genuinely out-of-sample, most-recent month held out from all training and testing.

## Results

| Metric | Model v1 (raw features) | Model v2 (percentile-rank features) |
|---|---|---|
| Overall Accuracy | 60% | 62% |
| Seller's Market Recall | 47% | 75% |
| Buyer's Market Precision | 56% | 64% |
| Macro Avg F1-Score | 0.58 | 0.61 |

- **62% overall accuracy** on held-out test data vs. a 33% random-guess baseline for three classes.
- **Sale-to-List Ratio** was the strongest predictor — a one-standard-deviation increase in a metro's relative sale-to-list ratio roughly tripled the odds of a Seller's Market classification.
- Accuracy on a genuinely out-of-sample month ranged from **79.9%** for the largest metro quartile down to **44.5%** for the smallest (a limitation traced to Zillow's own data-reporting thresholds for low-volume markets, not a modeling flaw).

## Repo Structure

```
real-estate-market-classifier/
├── README.md
├── notebooks/
│   ├── 01_data_merge_clean.ipynb      # data integration, cleaning, SQL joins
│   └── 02_analysis_modeling.ipynb     # EDA, feature/target engineering, modeling, validation
├── data/
│   └── clean_real_estate.csv          # final cleaned dataset used for modeling
├── visuals/
│   ├── accuracy_by_size_chart.png     # held-out accuracy by metro size
│   └── tableau_map_final.jpg          # predicted market conditions map
└── docs/
    └── market_classifier_portfolio_final.pdf   # 2-page project write-up
```

## Tools & Technologies

Python (pandas, scikit-learn, statsmodels), SQL (SQLite), Tableau, Google Colab

## Limitations & Future Work

Model reliability decreases for smaller, less-liquid metros, consistent across three independent signals: higher missingness rates, greater index volatility, and lower prediction accuracy. Future iterations could explore a size-tiered modeling approach, test tree-based models for non-linear feature interactions, or bring in external features (e.g., inventory levels, mortgage rates) to improve small-metro performance.

## Full Write-Up

See [`docs/market_classifier_portfolio_final.pdf`](docs/market_classifier_portfolio_final.pdf) for the full Problem / Goals / Action / Result summary with visuals.

## Author

Huy Nguyen
