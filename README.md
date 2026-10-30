# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, NumPy and Scikit-learn.
2.Initialize the dataset with house features such as area and number of bedrooms, along with house price and number of occupants.
3.Train the SGD Regressor model using the input features and target values.
4.Predict the house price and number of occupants for new input data and display the results. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: THARUN R
RegisterNumber:  212224240172
*/
python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Input features: Area (sq.ft) and number of bedrooms
X = np.array([
    [1000, 2],
    [1500, 3],
    [1800, 3],
    [2000, 4],
    [2500, 4]
], dtype=float)

# Target values: House price
y_price = np.array([20, 30, 36, 40, 50], dtype=float)

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SGD Regressor
model = SGDRegressor(max_iter=10000, tol=1e-6, random_state=42)
model.fit(X_scaled, y_price)

# Predict house price
new_house = np.array([[1600, 3]])
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)

print("House Area:", new_house[0][0], "sq.ft")
print("Number of Bedrooms:", int(new_house[0][1]))
print("Predicted House Price:", round(predicted_price[0], 2), "lakhs")
```


## Output:

<img width="967" height="85" alt="image" src="https://github.com/user-attachments/assets/3188277d-5a4b-4818-9bac-fc34adef6072" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
