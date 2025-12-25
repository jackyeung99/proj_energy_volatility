### Main Idea/ Hypothesis 
Does anomolies in weather help explain volatility of energy companies 


### 1. Weather Anomalies 
Using weather time series data develop anomaly detection models. 


- Gain weather data on multiple sources from openmeteo 
- Aggregate data from multiple sources 
- use different anomaly detection models on these data 


### 2. Volatility Modeling 


- Pre-Processing
    - gather data on energy stocks and macroeconomic featyres
    - merge data with weather time series 
    - shift all predictors to avoid data leakage 
- Volatility Measure
    - get idiosynractic componet of XLE 
    - develop a measure fo realized volatility to compare to 
- split data into train, validation, test
- Develop evaluation criteteria
    - use expandinng window for out of sample prediction on next period 
    - compute QLIKE of predicted volatility vs actual volatility 
- Hyperparameter tuning 
- test the impact of each exogenous feature on marginal imrpovement of QLIKE


### 3. Trading Strategy