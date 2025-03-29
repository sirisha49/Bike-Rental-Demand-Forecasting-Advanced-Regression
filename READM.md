# 🚲 Bike Rental Demand Forecasting – Advanced Regression Project

This project presents a comprehensive machine learning pipeline to predict **daily bike rental demand** based on temporal, seasonal, and weather-related data. The analysis covers data preprocessing, extensive assumption validation, implementation of diverse regression models, and diagnostic evaluation to arrive at the most statistically sound and interpretable forecasting model.

---

## Dataset Description

The dataset includes **daily aggregated bike rental data** from a bike-sharing system, structured with the following attributes:

- **Date & Calendar Features**:
  - `Year`, `Month`, `Day`, `Weekday`
  - `Holiday`, `Working Day`
  - `Season` (Winter, Spring, Summer, Fall)

- **Environmental Attributes**:
  - `Temperature`, `Humidity`, `Windspeed`
  - Normalized variables for temperature and windspeed are included

- **Target Variable**:
  - `Count`: Total number of rentals per day (continuous count variable)

This diverse feature set allows us to analyze time-driven patterns, weather sensitivity, and weekday/holiday effects on demand.

---

## Project Goals

1. **Understand patterns in bike rental behavior** through visualizations and summary statistics
2. **Test classical regression assumptions** to ensure appropriate model choice
3. **Apply and compare regression models** suitable for count, nonlinear, and multicollinear data
4. **Perform residual diagnostics and statistical testing**
5. **Select the most appropriate model** for reliable, interpretable predictions

---

## Exploratory Data Analysis

Initial analysis included:

- Time series plots showing rental growth over years and seasonal dips
- Boxplots to compare rental counts across seasons, months, and weekdays
- Distribution plots of numerical variables with transformation tests
- Correlation heatmaps indicating multicollinearity between predictors
- Scatterplots highlighting relationships with temperature and humidity

Findings:
- Rentals peaked during **summer** and **working weekdays**
- **Humidity negatively** impacted demand, while **temperature positively** influenced it
- Slight skew in count data necessitated transformation and count-based modeling

---

## Assumption Diagnostics

All models were subjected to rigorous statistical checks:

- **Linearity**: Improved via log transformation of `count`
- **Homoscedasticity**: Tested using the **Breusch-Pagan test**
- **Normality of residuals**: Evaluated via histogram and **QQ plots**
- **Multicollinearity**: Addressed with **VIF** analysis and Ridge/Lasso
- **Overdispersion in count data**: Led to testing Poisson vs. Negative Binomial

---

## Models Implemented

A total of **eight models** were trained, tuned, and compared:

1. **Multiple Linear Regression (MLR)** – Benchmark model
2. **Ridge Regression** – Mitigates multicollinearity
3. **Lasso Regression** – Shrinks coefficients; ideal for feature selection
4. **Polynomial Regression** – Captures non-linearity (with caution to avoid overfitting)
5. **Poisson Regression** – Appropriate for count data assuming equal mean and variance
6. **Negative Binomial Regression** – Better suited for **overdispersed count data**
7. **Principal Component Regression (PCR)** – Handles dimensionality reduction
8. **Support Vector Regression (SVR)** – Robust to outliers; captures nonlinear relationships

---

## Model Comparison

Each model was evaluated using:

- **RMSE**: Root Mean Squared Error  
- **R² / Adjusted R²**  
- **AIC/BIC**: For model parsimony (where applicable)  
- **Residuals**: Distribution and structure post-modeling  
- **Interpretability**: Key for practical implementation

**Key Results**:

- **Lasso Regression** offered strong performance with sparse coefficients
- **Negative Binomial Regression** captured variance in count data effectively
- **Polynomial Regression** fit training data well but introduced risk of overfitting
- **PCR** retained variance with reduced noise, improving generalizability
- **SVR** delivered competitive RMSE but lacked transparency in feature contribution

---

## Learning Outcomes

- How to validate and interpret statistical assumptions before modeling
- Selection of appropriate models based on **data type and distribution**
- Application of **regularization** for improved generalization
- Diagnostic techniques for identifying violations and improving models
- Communication of insights through visual and statistical summaries
- Balancing **prediction performance** with **model interpretability**

---

## Technologies Used

- Python 3.x  
- Jupyter Notebook  
- Libraries:
  - `pandas`, `numpy`, `seaborn`, `matplotlib`
  - `statsmodels`, `scikit-learn`, `scipy`
  - `patsy` for regression syntax handling

---

## Final Report Summary

The accompanying PDF report includes:

- Step-by-step EDA visualizations and summaries
- Statistical tests and regression assumptions
- Full model specifications and comparative results
- In-depth analysis of residuals and diagnostics
- Final recommendation with interpretability vs. performance discussion

