# Gold Price Forecasting

## Overview

This project focuses on predicting gold prices using various machine learning models, including linear regression, LSTM neural networks, and ensemble methods like Random Forest and XGBoost. The goal is to leverage historical data to forecast future prices, aiding investors and financial analysts in making informed decisions. Additionally, a trading strategy based on the forecasted prices has been implemented and evaluated over the entire period to assess its effectiveness and profitability.


## Modeling Techniques


### Linear Models

+ **Ordinary Least Squares** (**OLS**): A baseline linear regression technique that minimizes the sum of squared residuals to model the relationship between gold prices and independent variables.
+ **Ridge Regression**: A linear regression method that applies $\mathcal L^2$ regularization to prevent overfitting, particularly useful in the presence of multicollinearity.
+ **Lasso Regression**: Similar to Ridge but uses $\mathcal L^1$ regularization, which can shrink some coefficients to zero, effectively performing feature selection.
+ **Elastic Net**: Combines $\mathcal L^1$ and $\mathcal L^2$ penalties, balancing the strengths of Ridge and Lasso, making it effective for high-dimensional datasets.
+ **Principal Component Regression** (**PCR**): Applies PCA to reduce dimensionality and capture variance, followed by regression on the principal components to improve model performance.

### LSTM Neural Network

A type of RNN designed for sequential data, capturing temporal dynamics in gold prices. The model is optimized through hyperparameter tuning using Keras Tuner.

### Ensemble Methods

+ **Stacking Models**: Combines predictions from multiple ensemble methods, including AdaBoost, Random Forest, Gradient Boosting, & Extra Trees. The base model predictions serve as inputs for the final stacking model (XGBoost), enhancing overall predictive accuracy.


## Results

### Model Performance Metrics :

- **OLS**: $R^2$ Score: 0.990847.
- **Ridge**: $R^2$ Score: 0.989789.
- **Lasso**: $R^2$ Score: 0.985848.
- **Elastic Net**: $R^2$ Score: 0.985380.
- **Forward Variable Selection**: $R^2$ Score: 0.990410.
- **PCR**: $R^2$ Score: 0.990841.
- **LSTM**: $R^2$ Score: 0.888253.
- **Stacking Model (XGBoost)**: $R^2$ Score: 0.445287.

### Trading Strategy Evaluation :

- **Linear Models**: Sharpe Ratio: 0.68.
- **LSTM**: Sharpe Ratio: 0.89.
- **Stacking Model (XGBoost)**: Sharpe Ratio: 0.6.

  
### Best Strategy Performance (Yielded by LSTM Model) :

![image](https://github.com/user-attachments/assets/35b0802e-d0ce-40d8-aedd-88c1ab628bbb)

### Future Work
Future enhancements could include integrating additional features, refining the trading strategy, and exploring more advanced models for improved accuracy.

> **⚠️ Note:** For an enhanced experience when viewing Jupyter notebooks, we recommend using **github.dev** rather than **github.com**. By navigating to `https://github.dev/ayoub-mg/Gold-Price-Forecasting/`, you can access the repository in a more interactive environment that better supports the rendering of Jupyter notebooks and Plotly visualizations. Alternatively, you can also utilize [nbviewer.org](https://nbviewer.org/) to load the notebooks, which offers improved rendering and a cleaner, more user-friendly viewing experience.
