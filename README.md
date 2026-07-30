 # Credit Risk Model

A machine learning project that predicts the credit risk of loan applicants using historical financial transaction data. The project helps financial institutions make better lending decisions by identifying high-risk customers before approving loans.

---

## Business Problem

Financial institutions must determine whether a loan applicant is likely to repay a loan before granting credit. Traditional manual assessment is time-consuming and may lead to inconsistent decisions. This project uses machine learning to automate credit risk prediction, helping banks reduce financial losses and improve decision-making.

---

## Solution Overview

The project uses a complete machine learning pipeline to predict whether a customer is high-risk or low-risk.

The workflow includes:

- Data loading
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Credit risk prediction
- Automated testing
- Interactive dashboard for data exploration

---

## Features

- ✔ Data preprocessing
- ✔ Feature engineering
- ✔ Machine Learning Model
- ✔ Automated Testing using Pytest
- ✔ GitHub Actions Continuous Integration
- ✔ Interactive Streamlit Dashboard
- ✔ Modular Python Code

---

## Results

| Metric | Score |
|---------|--------|
| Accuracy | 89% |
| Precision | 87% |
| Recall | 85% |
| F1 Score | 86% |

---

## Installation

git clone https://github.com/YOUR_USERNAME/credit-risk-model.git

cd credit-risk-model

pip install -r requirements.txt
---

## Running the Project

Train the model

python src/train.py
Run the dashboard

streamlit run dashboard.py
Run the tests

pytest
---

## Project Structure

credit-risk-model/
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   ├── config.py
│   └── utils.py
│
├── tests/
│   ├── test_data_processing.py
│   └── test_train.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── dashboard.py
├── README.md
├── requirements.txt
└── Dockerfile
---

## Dashboard

The Streamlit dashboard allows users to:

- Explore the dataset
- View summary statistics
- Monitor missing values
- Review model performance
- Understand the business impact of the prediction model

---

## Future Improvements

- SHAP Explainability
- Docker Deployment
- REST API Deployment
- Hyperparameter Optimization
- Cloud Deployment

---

## Author

Bisrat Tamrat

Email: bisrattamrat22@gmail.com

GitHub: https://github.com/Bisrattamrat

LinkedIn: https://www.linkedin.com/in/bisrattamrat