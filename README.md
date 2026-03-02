# Customer Churn Prediction System

**BIS405 Graduation Project – Data Analytics and Decision Intelligence**  
**Imam Abdulrahman Bin Faisal University | College of Business Administration**

## Project Overview

AI-powered customer churn prediction system that identifies at-risk customers before they leave, enabling proactive retention strategies.

**Dataset:** [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) by Muhammad Shahid Azeem  
**Records:** 440,832 training + 64,374 testing | **Features:** 12

## Team

| Name | ID | Role |
|------|----|------|
| Gader| XXXXX | Data Scientist |
| hadeel|2230004733| Business Analyst |
| maha| XXXXX | Data Engineer / Scrum Master |

**Supervisor:** Dr. [Name]

## Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | int | Unique customer identifier |
| Age | int | Customer age (18-65) |
| Gender | cat | Male / Female |
| Tenure | int | Months with company (1-60) |
| Usage Frequency | int | Service usage frequency (1-30) |
| Support Calls | int | Number of support calls (0-10) |
| Payment Delay | int | Payment delay in days (0-30) |
| Subscription Type | cat | Basic / Standard / Premium |
| Contract Length | cat | Monthly / Quarterly / Annual |
| Total Spend | int | Total amount spent ($) |
| Last Interaction | int | Days since last interaction (1-30) |
| **Churn** | **binary** | **0 = Retained, 1 = Churned** |

## Project Structure

```
customer-churn-analytics/
├── data/
│   ├── customer_churn_dataset-training-master.csv   # 440K training records
│   └── customer_churn_dataset-testing-master.csv    # 64K test records
├── src/
│   ├── data_loader.py             # Data loading & validation (Student 1)
│   ├── eda_analysis.py            # EDA visualizations (Student 1)
│   ├── business_insights.py       # Business insights report (Student 2)
│   └── data_preprocessing.py      # Preprocessing pipeline (Student 3)
├── models/                        # Saved model artifacts
├── outputs/
│   └── eda_plots/                 # 11 EDA visualizations
├── docs/
│   └── sprint_reports/            # Bi-weekly sprint reports
├── tests/                         # Unit tests
├── run_sprint1.py                 # Sprint 1 main runner
├── requirements.txt               # Python dependencies
└── README.md
```

## Quick Start

```bash
git clone https://github.com/[team]/customer-churn-analytics.git
cd customer-churn-analytics
pip install -r requirements.txt
python run_sprint1.py
```

## Sprint Progress

| Sprint | Weeks | Goal | Status |
|--------|-------|------|--------|
| Sprint 1 | 4-5 | Data Foundation & EDA | ✅ Complete |
| Sprint 2 | 6-7 | Baseline Modeling | 🔜 Next |
| Sprint 3 | 8-9 | Advanced Models | ⬜ Planned |
| Sprint 4 | 10-11 | Dashboard Development | ⬜ Planned |
| Sprint 5 | 12-13 | Final Integration | ⬜ Planned |

## Sprint 1 Key Findings

- **440,832** records after cleaning (1 null row removed)
- **56.7%** overall churn rate
- **Support Calls** is the strongest churn predictor
- **Usage Frequency** strongly differentiates churners vs retained
- Contract Length and Subscription Type show similar churn rates across categories
- No significant gender-based churn difference

## License

Academic project – Imam Abdulrahman Bin Faisal University, BIS Department.  
Dataset: [CC0 Public Domain](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
