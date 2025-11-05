# 🤖 Machine Learning Regression Projects

This repository contains two end-to-end **Machine Learning Regression Projects**:
1. 🚗 **Car Price Prediction using Gradient Descent Regression**
2. 🏠 **House Price Prediction using Random Forest Regressor**

Both projects include complete workflows — from data preprocessing to model deployment — built with **Python**, **Scikit-Learn**, and **Flask/Streamlit**.

---

## 🚗 Car Price Prediction using Gradient Descent Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Regression-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

### 📘 Overview
This project predicts the **selling price of a used car** based on multiple features like fuel type, transmission, mileage, and manufacturing year.  
A **Linear Regression model trained using Gradient Descent** was implemented to minimize prediction error.  
Feature scaling and categorical encoding were applied to enhance model performance.

---

### 📊 Dataset
Dataset file: [`car data (2).csv`](car_price/car%20data%20(2).csv)

| Feature | Description |
|----------|--------------|
| Car_Name | Brand/Model of the car |
| Year | Manufacturing year |
| Present_Price | Current ex-showroom price |
| Kms_Driven | Distance covered by the car |
| Fuel_Type | Type of fuel (Petrol/Diesel/CNG) |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |
| Selling_Price | Target variable (Predicted price) |

---

### ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy – Data manipulation  
- Matplotlib, Seaborn – Visualization  
- Scikit-Learn – Regression modeling  
- Streamlit – Web app interface  
- Pickle – Model serialization  

---

### 🧠 Model Building
**Algorithm Used:**  
➡️ **Linear Regression (trained using Gradient Descent Optimizer)**

**Steps Applied:**
1. Handled missing data and outliers  
2. Label-encoded categorical columns (`Fuel_Type`, `Transmission`)  
3. Scaled numerical features using `StandardScaler`  
4. Trained Gradient Descent-based Linear Regression model  
5. Saved model and encoders as `.pkl` files for deployment  

---

### 📈 Results

✅ **Final Model:** Gradient Descent Regression  
⚙️ Balanced between interpretability and accuracy  

---

### 🖥️ Web App Interface
<img src="car_price/static/images/car_ui.png" width="700">

Built with **Streamlit**, this interactive app allows users to input car details and view the estimated resale value instantly.

---

## 🏠 House Price Prediction using Random Forest Regressor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Regression-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

### 📘 Overview
This project predicts the **price of a house** based on various socio-economic and environmental factors.  
A **Random Forest Regressor** model was used to capture non-linear relationships between features and the target variable.  
The model was trained using the **Boston Housing Dataset**.

---

### 📊 Dataset
Dataset file: [`BostonHousing.csv`](House_price/BostonHousing.csv)

| Feature | Description |
|----------|--------------|
| CRIM | Per capita crime rate by town |
| ZN | Residential land zoned for large lots |
| INDUS | Non-retail business acres per town |
| CHAS | Charles River dummy variable (1 = bounds river) |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of old owner-occupied units |
| DIS | Distance to employment centers |
| RAD | Accessibility to radial highways |
| TAX | Property-tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion of Black residents |
| LSTAT | % lower status of the population |
| MEDV | Target variable – Median home value |

---

### ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy – Data preprocessing  
- Matplotlib, Seaborn – Visualization  
- Scikit-Learn – Regression & Evaluation  
- Flask – Web application deployment  
- Pickle – Model persistence  

---

### 🧠 Model Building
**Algorithm Used:**  
➡️ **Random Forest Regressor (Ensemble Learning)**

**Workflow Steps:**
1. Loaded and explored Boston Housing data  
2. Scaled features using `StandardScaler`  
3. Split dataset into training and test sets  
4. Trained multiple models (Linear Regression, Decision Tree, Random Forest)  
5. Selected Random Forest as the final model  
6. Deployed using Flask web framework  

---

### 📈 Results
 **Random Forest Regressor**  ✅ Final selected model 

---

### 🖥️ Web App Interface
<img src="House_price/static/images/house_ui.png" width="700">
<img src="House_price/static/images/house_result.png" width="700">

Flask web interface allows users to input feature values and view predicted house prices instantly.


