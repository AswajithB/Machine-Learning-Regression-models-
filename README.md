# 🤖 Machine Learning Regression Projects

This repository contains two complete **Machine Learning Regression Projects**:
1. 🚗 **Car Price Prediction using Gradient Boosting Regressor**
2. 🏠 **House Price Prediction using Random Forest Regressor**

Each project explores multiple regression techniques, applies feature scaling and encoding, and includes a web interface built using **Streamlit** or **Flask**.

---

# 🚗 Car Price Prediction using Gradient Boosting Regressor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Model-Gradient%20Boosting-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📘 Overview
This project predicts the **selling price of used cars** based on parameters such as year, mileage, fuel type, and transmission.  
The data was preprocessed with encoding and scaling, and several regression algorithms were evaluated to find the best-performing model.

---

## 📊 Dataset
Dataset file: [`car data (2).csv`](car_price/car%20data%20(2).csv)

| Feature | Description |
|----------|--------------|
| Car_Name | Brand/Model of the car |
| Year | Manufacturing year |
| Present_Price | Current ex-showroom price |
| Kms_Driven | Distance driven |
| Fuel_Type | Type of fuel (Petrol/Diesel/CNG) |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |
| Selling_Price | Target variable (predicted price) |

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy – Data preprocessing  
- Matplotlib, Seaborn – Visualization  
- Scikit-Learn – Model development and evaluation  
- Streamlit – Interactive user interface  
- Pickle – Model persistence  

---

## 🧠 Model Development
The following regression models were implemented and tested:
- Linear Regression  
- Polynomial Regression  
- Support Vector Regressor (SVR)  
- Decision Tree Regressor  
- Random Forest Regressor  
- AdaBoost Regressor  
- **Gradient Boosting Regressor** ✅ *(Selected Model)*

### Data Processing Steps
1. Encoded categorical features (`Fuel_Type`, `Transmission`)  
2. Scaled numerical attributes using `MinMaxScaler`  
3. Split data into train and test sets  
4. Trained and compared all models  
5. Saved the final model as `model.pkl`  

---

## 📈 Results
After evaluating all models, **Gradient Boosting Regressor** produced the most accurate results with an **R² score of approximately 0.95** on the test data.  
The model effectively reduced error variance and generalized well across unseen samples.

---

## 🖥️ Web App Interface
<img src="car_price/static/images/car_ui.png" width="700">

The Streamlit app allows users to enter car details (fuel type, transmission, year, etc.) and view real-time price predictions.

---

# 🏠 House Price Prediction using Random Forest Regressor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📘 Overview
This project predicts **house prices** using features like the average number of rooms, location, and tax rate from the **Boston Housing dataset**.  
Multiple regression algorithms were tested, and the **Random Forest Regressor** was selected for its excellent predictive power and robustness.

---

## 📊 Dataset
Dataset file: [`BostonHousing.csv`](House_price/BostonHousing.csv)

| Feature | Description |
|----------|--------------|
| CRIM | Per capita crime rate by town |
| ZN | Residential land zoned for large lots |
| INDUS | Non-retail business acres per town |
| CHAS | River adjacency indicator |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Age of owner-occupied units |
| DIS | Distance to employment centers |
| RAD | Highway accessibility index |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Student-teacher ratio |
| B | Proportion of Black residents |
| LSTAT | % of lower-income population |
| MEDV | Target variable – Median value of homes |

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy – Data handling  
- Matplotlib, Seaborn – Visualization  
- Scikit-Learn – Modeling and evaluation  
- Flask – Web interface deployment  
- Pickle – Model serialization  

---

## 🧠 Model Development
Models evaluated:
- Linear Regression  
- Polynomial Regression  
- Support Vector Regressor (SVR)  
- Decision Tree Regressor  
- **Random Forest Regressor** ✅ *(Selected Model)*  
- AdaBoost Regressor  
- Gradient Boosting Regressor  

### Workflow
1. Standardized numerical variables using `MinMaxScaler`  
2. Split dataset into training and test sets  
3. Trained all regression models  
4. Compared R², MAE, and RMSE scores  
5. Deployed the best model using Flask  

---

## 📈 Results
The **Random Forest Regressor** achieved an **R² score of approximately 0.84** on the test data, outperforming all other models.  
It showed excellent prediction consistency and minimized overfitting compared to single-tree models.

---

## 🖥️ Web App Interface
<img src="House_price/static/images/house_ui.png" width="700">


The Flask web application accepts input parameters (e.g., average rooms, tax rate, and pollution level) and outputs predicted house prices.

---
