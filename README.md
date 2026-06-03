 Credit Risk Probability Model for Alternative Data

Credit Scoring Business Understanding

Basel II and Model Interpretability

The Basel II Accord emphasizes risk measurement, transparency, regulatory compliance, and proper documentation of credit risk models. Because lending decisions directly affect financial exposure, models should be interpretable, reproducible, and well documented. This allows both business stakeholders and regulators to understand how risk scores are generated and ensures that model decisions can be audited when necessary.

Why a Proxy Variable is Necessary

The dataset does not contain a direct loan default label. Therefore, a proxy target variable must be created to estimate customer credit risk.

To address this limitation, customer behavior was analyzed using Recency, Frequency, and Monetary (RFM) metrics. Customers were segmented using K-Means clustering, and the least engaged customer segment was identified as the high-risk group. A new binary target variable named "is_high_risk" was created, where:

- 1 = High Risk Customer
- 0 = Low Risk Customer

The primary limitation of this approach is that behavioral risk does not perfectly represent actual loan default behavior. Therefore, predictions should be interpreted as estimated risk indicators rather than confirmed default outcomes.

Trade-Off Between Interpretable and High-Performance Models

Interpretable models such as Logistic Regression provide transparency and can be easily explained to regulators and business stakeholders. However, they may sacrifice some predictive performance.

More complex models such as Random Forest and Gradient Boosting often achieve higher predictive performance but reduce explainability. In regulated financial environments, an appropriate balance between predictive power and transparency is required.

---

Exploratory Data Analysis (EDA)

The dataset contains transaction-level records from the Xente eCommerce platform.

Key EDA activities included:

- Dataset inspection and validation
- Summary statistics analysis
- Missing value assessment
- Distribution analysis
- Correlation analysis
- Outlier detection
- Fraud distribution analysis
- Customer transaction behavior analysis

Key Findings

1. The majority of transactions were concentrated within a small number of transaction channels.

2. Fraudulent transactions represented only a very small proportion of all observations, indicating significant class imbalance.

3. Transaction amounts were highly right-skewed and contained several extreme outliers.

4. Product category usage was uneven, with a small number of categories dominating transaction volume.

5. Missing values were limited and could be handled through preprocessing and imputation techniques.


Feature Engineering

A reusable preprocessing pipeline was implemented using Scikit-Learn Pipeline and ColumnTransformer.

Feature engineering included:

Aggregate Customer Features

Customer-level behavioral features were created including:

- Total Transaction Amount
- Average Transaction Amount
- Transaction Count
- Standard Deviation of Transaction Amounts

Datetime Features

Transaction timestamps were transformed into additional predictive features:

- Transaction Hour
- Transaction Day
- Transaction Month
- Transaction Year

Data Preprocessing

The pipeline performs:

- Missing value imputation
- Numerical feature scaling using StandardScaler
- One-Hot Encoding for categorical variables
- Automated transformation through ColumnTransformer

These improvements were added after project review feedback to strengthen the predictive feature set.


Proxy Target Variable Engineering

Because no direct default label exists in the dataset, an RFM-based proxy target was developed.

RFM Metrics

For each CustomerId:

- Recency
- Frequency
- Monetary Value

were calculated.

Customer Segmentation

Customers were segmented using:

- StandardScaler
- K-Means Clustering
- 3 Customer Clusters
- random_state=42
The cluster with the lowest average Frequency and Monetary value was identified as the high-risk customer segment.

The resulting binary target variable:

"is_high_risk"

was merged back into the processed dataset for model training.


Model Training

Two machine learning models were trained and compared:

Logistic Regression

Used as an interpretable baseline model.

Random Forest Classifier

Used as a higher-performance ensemble model.

Hyperparameter Tuning

To improve model performance, GridSearchCV was implemented for Random Forest tuning.

Parameters explored included:

- n_estimators
- max_depth

The best-performing model was selected using ROC-AUC evaluation.

Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC


Experiment Tracking

MLflow was used to track:

- Model runs
- Evaluation metrics
- Hyperparameters
- Model artifacts

This ensured reproducibility and comparison across experiments.


API Deployment

A FastAPI application was implemented to expose the model through a REST API.

Endpoint:

POST /predict

Response:

{
  "risk_probability": 0.73
}

The API enables real-time credit risk scoring for new customer applications.


CI/CD

GitHub Actions was configured to automate:

- Dependency installation
- Code quality checks using flake8
- Unit testing using pytest

This helps ensure code reliability and maintainability.


Project Structure

credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── api/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md


Challenges Encountered

Several implementation challenges were encountered and resolved:

- Initial loading of the incorrect metadata file instead of the dataset.
- Missing package installations including pandas, pytest, and scikit-learn.
- Virtual environment activation issues in PowerShell.
- Missing pytest installation causing test failures.
- Docker installation issues preventing container execution.
- Dataset path and file-loading errors.
- Debugging training script dependency errors.

These challenges were systematically addressed throughout development.


Conclusion

This project successfully implemented an end-to-end credit risk modeling workflow for Bati Bank using alternative transaction data.

The solution includes:

- Business understanding aligned with Basel II principles
- Exploratory Data Analysis
- Advanced feature engineering
- RFM-based proxy target creation
- Hyperparameter-tuned machine learning models
- MLflow experiment tracking
- FastAPI deployment
- CI/CD automation

The resulting system provides a practical framework for estimating customer credit risk in situations where traditional loan repayment history is unavailable.

Author: Bisrat Tamrat Bekele
