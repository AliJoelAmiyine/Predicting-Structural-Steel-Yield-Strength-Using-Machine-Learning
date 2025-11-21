# Steel Yield Strength Prediction with Machine Learning
This project demonstrates **predicting steel yield strength** using machine learning. I came across this dataset by chance and wanted to explore how data-driven models can support **material performance analysis** in the **construction industry**.
The repository includes **data exploration**, **visualization**, **modeling**, and **predictions**.

---

## Dataset Overview
* **Number of Samples:** 312
* **Number of Features:** 17 (chemical composition + mechanical properties)
* **Target Variable:** `Yield Strength` (MPa)
> The dataset captures variations in steel composition and their corresponding yield strengths—ideal for predictive modeling.

---

## Exploratory Data Analysis (EDA)

### 1. Yield Strength Distribution

Visualizing the range and frequency of yield strength values:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['Yield_strength'], bins=30, kde=True)
plt.title("Distribution of Steel Yield Strength")
plt.xlabel("Yield Strength (MPa)")
plt.ylabel("Frequency")
plt.show()
```

*Example Output:*

![Yield Strength Histogram](path_to_your_histogram.png)

---

### 2. Carbon Content vs Yield Strength

Scatter plot with regression line to explore correlation:

```python
sns.regplot(x='C', y='Yield_strength', data=data)
plt.title("Carbon Content vs Yield Strength")
plt.xlabel("Carbon Content (%)")
plt.ylabel("Yield Strength (MPa)")
plt.show()
```

*Example Output:*

![Carbon vs Yield Strength](path_to_your_scatterplot.png)

---

## Machine Learning Models Tested

Three regression models were trained and evaluated:

| Model                             | MAE    | R² Score |
| --------------------------------- | ------ | -------- |
| RandomForest Regressor         | 78.99  | 0.822    |
| XGBoost Regressor               | 79.98  | 0.816    |
| Support Vector Regressor (SVR) | 182.29 | –0.027   |

> RandomForest and XGBoost performed well. SVR struggled, likely due to hyperparameter sensitivity.

### Example: RandomForest Training

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = data.drop(columns='Yield_strength')
y = data['Yield_strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

---

## Prediction Example

Using **RandomForest**, the predicted yield strength for a new steel sample:

```python
new_sample = [[0.12, 0.35, 0.02, 0.04, ...]]  # Replace with actual feature values
predicted_strength = rf.predict(new_sample)
print("Predicted Yield Strength:", predicted_strength[0], "MPa")
```

> **Predicted Yield Strength:** 1293.24 MPa

---

## Key Takeaways

* RandomForest and XGBoost can **accurately predict steel yield strength**.
* Exploratory Data Analysis (EDA) is essential for understanding feature-target relationships.
* Machine learning can **support material selection, quality control, and structural engineering decisions**.
