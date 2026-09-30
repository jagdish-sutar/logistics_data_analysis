# Week 4 — Predictive Modeling and Optimization in Logistics

**YuvaIntern | Logistics Data Analyst Intern**

This project demonstrates predictive modeling and a simple optimization workflow for a hypothetical logistics system.

## Objective

Predict shipment delivery time and use the prediction in an illustrative transport-mode decision scenario.

## Models

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

## Evaluation

Models are evaluated using:

- MAE
- RMSE
- R²
- 5-fold cross-validation

Random Forest hyperparameters are tuned using `GridSearchCV`.

## Optimization

A representative shipment is evaluated across Road, Rail, and Air. The illustrative objective combines transport cost with a penalty when predicted delivery time exceeds a 5-day target.

## Important limitation

All data and results are synthetic. They are intended to demonstrate the methodology and do not represent real logistics performance or real operational recommendations.

## Files

- `week4_predictive_modeling.ipynb`
- `src/week4_predictive_logistics.py`
- `data/hypothetical_logistics_week4_dataset.csv`
- `results/`
- `charts/`
- `requirements.txt`

## Run

```bash
pip install -r requirements.txt
python src/week4_predictive_logistics.py
```

Or open the notebook:

```bash
jupyter notebook week4_predictive_modeling.ipynb
```
