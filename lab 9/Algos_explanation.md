# House Price Prediction: Six Regression Algorithms Compared

**Muhammad Majid Ali | FA23-BAI-045**

## Overview

This notebook predicts median house value from the California Housing dataset using six regression algorithms: Linear Regression, Decision Tree, Random Forest, Neural Network, XGBoost and KNN. All six models use the same data, features, target and 80:20 split, so the comparison is fair.

## 1. Data

- 20,640 districts, 9 input features plus the target `median_house_value`.
- `total_bedrooms` has 207 missing values.
- `ocean_proximity` is categorical.
- Target prices are capped at 500,001 in the source data.
- A histogram and a correlation heatmap show the price distribution and which features relate to price.

## 2. Preprocessing

1. One-hot encode `ocean_proximity` (special characters removed from column names, because XGBoost rejects `<`).
2. Split 80:20 with `random_state=42`: 16,512 training rows and 4,128 test rows.
3. Fill the missing `total_bedrooms` values with the training median.
4. Standardise the numeric features using training statistics only, for the models that need scaling (Linear Regression, Neural Network, KNN). The Neural Network also gets a standardised target, and its predictions are converted back to dollars.

## 3. Models

| Model | Setting | Scaled |
| --- | --- | --- |
| Linear Regression | Ordinary least squares | Yes |
| Decision Tree | max depth 8 | No |
| Random Forest | 200 trees | No |
| Neural Network | Hidden layers 64 and 32, early stopping | Yes |
| XGBoost | 300 trees, learning rate 0.1, depth 6 | No |
| KNN | 5 neighbours | Yes |

No hyperparameter tuning was done.

## 4. Evaluation

Each model is trained and tested by one reusable function. It reports MAE, MSE, RMSE and R², and the training time.

| Rank | Model | MAE | RMSE | R² |
| --- | --- | --- | --- | --- |
| 1 | XGBoost | 30,508 | 46,129 | 0.8376 |
| 2 | Random Forest | 31,487 | 48,823 | 0.8181 |
| 3 | Neural Network | 35,958 | 53,444 | 0.7820 |
| 4 | KNN | 40,763 | 61,310 | 0.7132 |
| 5 | Decision Tree | 42,777 | 62,485 | 0.7020 |
| 6 | Linear Regression | 50,670 | 70,059 | 0.6254 |

Training time ranges from 0.04 seconds (KNN) to 42.8 seconds (Random Forest).

## 5. Plots

- **Actual vs predicted:** points close to the diagonal line mean accurate predictions.
- **Residual histograms:** errors centred on zero show an unbiased model.
- **RMSE and R² bars:** a quick visual ranking of the models.
- **Feature importance:** shows which features Random Forest and XGBoost rely on most.

## 6. Conclusion

XGBoost is the best model: it has the lowest error and the highest R². Its RMSE is 5.5% lower than Random Forest and 34.2% lower than Linear Regression. Boosting corrects the remaining errors step by step and captures non-linear effects, which a straight line cannot. Linear Regression is the weakest because the relationship between features and price is not linear.

**Notes:** the models were not tuned, the results come from a single split, and prices are capped at 500,001 in the source data.
