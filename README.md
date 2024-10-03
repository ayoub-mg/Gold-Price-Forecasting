# Gold Price Forecasting

## Overview

This project focuses on predicting gold prices using various machine learning models, including linear regression, LSTM neural networks, and ensemble methods like Random Forest and XGBoost. The goal is to leverage historical data to forecast future prices, aiding investors and financial analysts in making informed decisions. Additionally, a trading strategy based on the forecasted prices has been implemented and evaluated over the entire period to assess its effectiveness and profitability.

## Data Preprocessing & Feature Engineering

The script (`dataprep.py`) was used to prepare and engineer features from historical gold price data (GLD ETF) sourced from Yahoo Finance. After cleaning the data by removing missing values, various technical indicators were computed, such as Simple and Exponential Moving Averages (SMA, EMA), Average True Range (ATR), and Relative Strength Index (RSI), to capture key market trends and volatility. Additional data from related financial instruments and indices, such as the US Dollar Index (DXY) and VanEck Vectors Gold Miners ETF (GDX), were incorporated to provide broader market context. Feature selection techniques, including correlation analysis, F-regression, RandomForest, and Lasso regression, were applied to optimize the dataset for predictive modeling by selecting the most relevant features.

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


## Results & Assessments

### Model Performance Metrics :

- **OLS**: $R^2$ Score: 0.990847.
- **Ridge**: $R^2$ Score: 0.989789.
- **Lasso**: $R^2$ Score: 0.985848.
- **Elastic Net**: $R^2$ Score: 0.985380.
- **Forward Variable Selection**: $R^2$ Score: 0.990410.
- **PCR**: $R^2$ Score: 0.990841.
- **LSTM**: $R^2$ Score: 0.978041.
- **Stacking Model (XGBoost)**: $R^2$ Score: 0.445287.

### Trading Strategy Evaluation :

- **Linear Models**: Sharpe Ratio: 0.68.
- **LSTM**: Sharpe Ratio: 0.81.
- **Stacking Model (XGBoost)**: Sharpe Ratio: 0.6.
  
### Conclusion

The LSTM model has effectively predicted the next day's closing price, providing valuable insights for future investment strategies. The accompanying figure demonstrates the model's ability to deliver accurate predictionsAdditionally, this model has achieved superior performance in strategy returns, as demonstrated by the accompanying figure. For a interactive comparison of actual vs. predicted prices, please refer to the [Actual vs. Predicted Prices by LSTM](./LSTM.html).

![image](https://github.com/user-attachments/assets/b9b8ddb5-f1fd-4b26-95e2-f888f693bf32)


> **⚠️ Note:** For an enhanced experience when viewing Jupyter notebooks, we recommend using **github.dev** rather than **github.com**. By navigating to `https://github.dev/ayoub-mg/Gold-Price-Forecasting/`, you can access the repository in a more interactive environment that better supports the rendering of Jupyter notebooks and Plotly visualizations. Alternatively, you can also utilize [nbviewer.org](https://nbviewer.org/) to load the notebooks, which offers improved rendering and a cleaner, more user-friendly viewing experience.
