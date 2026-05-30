# Credit Scoring Business Understanding

## Basel II and Model Interpretability

The Basel II Accord emphasizes risk measurement, transparency, and regulatory compliance. Because lending decisions directly affect financial risk, credit scoring models must be interpretable and well documented. An interpretable model allows the bank to explain why a customer was approved or rejected and helps regulators audit the decision-making process.

## Why a Proxy Variable is Necessary

The dataset does not contain a direct default label. Therefore, a proxy variable must be created to estimate customer credit risk. Customer behavior patterns can be analyzed using Recency, Frequency, and Monetary (RFM) metrics to identify customers who appear disengaged or risky.

The main risk of proxy-based prediction is that the proxy may not perfectly represent true default behavior. This can introduce bias and reduce model accuracy if the assumptions behind the proxy are incorrect.

## Trade-Off Between Interpretable and High-Performance Models

Interpretable models such as Logistic Regression with Weight of Evidence (WoE) provide transparency and are easier to justify to regulators. However, they may achieve lower predictive performance.

More advanced models such as Gradient Boosting or XGBoost often provide higher predictive accuracy but are harder to explain. In regulated financial environments, a balance must be found between predictive power and explainability.
