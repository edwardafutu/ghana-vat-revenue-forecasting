# Predictive Modelling of Domestic Value Added Tax Revenue in Ghana Using Machine Learning Techniques

## Project Overview

This research investigates the application of machine learning techniques to forecasting monthly Domestic Value Added Tax (VAT) revenue in Ghana.

The study evaluates four machine learning models and compares their forecasting performance with conventional time-series benchmarks. It also examines the macroeconomic and calendar factors associated with changes in Domestic VAT revenue.

The research was conducted as a joint MSc Big Data Analytics dissertation at the University of Ghana, Department of Statistics and Actuarial Science.

## Research Contributors

**Edward Afutu Amoni**  
**Victoria Naa Adzeley Koney-Mensah**

Both contributors are co-authors of the dissertation.

**Supervisor:** Dr. Benedict Mbeah-Baiden  
**Institution:** University of Ghana  
**Programme:** MSc Big Data Analytics  
**Submission:** November 2026

## Research Objectives

The study pursued three main objectives:

1. Apply machine learning models to forecast Domestic VAT revenue in Ghana using historical VAT and relevant economic indicator data.
2. Compare the forecasting performance of machine learning models with conventional forecasting benchmarks using appropriate accuracy measures.
3. Identify and analyse the key macroeconomic and calendar factors that influence Domestic VAT revenue.

## Dataset

The study used 192 monthly observations covering January 2010 to December 2025.

The data consisted of:

- Domestic VAT revenue (GHS million)
- Composite Index of Economic Activity (CIEA)
- Exchange rate (GHS per US dollar)
- Treasury bill rate
- Consumer Price Index (CPI)
- Monthly inflation

Domestic VAT data were obtained from the Ghana Revenue Authority (GRA), while the macroeconomic indicators were sourced from the relevant public institutions, including the Bank of Ghana and Ghana Statistical Service.

The study focused specifically on Domestic VAT collected on goods and services consumed within Ghana. Import VAT was excluded.

## Methodology

### 1. Data Preparation and Feature Engineering

The analysis involved:

- Data checks and exploratory data analysis.
- Examination of trends, seasonality and unusual observations.
- Stationarity testing using the Augmented Dickey-Fuller (ADF) and KPSS tests.
- Transformation of VAT revenue into monthly changes in log VAT.
- Construction of lagged predictors using historical information.
- Creation of calendar variables to account for monthly and annual patterns.

The models used 15 predictors constructed from earlier months. This ensured that the forecasting process used information available before the month being predicted.

### 2. Forecasting Models

**Machine Learning Models**

- Least Absolute Shrinkage and Selection Operator (LASSO)
- Support Vector Regression (SVR)
- Extreme Gradient Boosting (XGBoost)
- One-Dimensional Convolutional Neural Network (1D-CNN)

**Conventional Benchmark Models**

- Naive Forecast
- Seasonal Naive Forecast
- Autoregressive Integrated Moving Average with Exogenous Variables (ARIMAX)

### 3. Model Training and Validation

The modelling dataset was divided chronologically:

- **Training period:** January 2010 – December 2023 (155 modelling observations).
- **Test period:** January 2024 – December 2025 (24 observations).

Hyperparameter tuning was conducted using rolling validation. Models were evaluated using one-month-ahead forecasts, ensuring that future information was not used to predict earlier observations.

### 4. Model Evaluation

Forecasting performance was assessed using:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

Lower values indicate better forecasting performance.

## Results and Key Findings

SVR achieved the best overall forecasting performance across all three evaluation measures. However, its improvement over ARIMAX was small.

### Model Performance on the Test Data

| Model | RMSE (GHS million) | MAE (GHS million) | MAPE (%) |
|---|---:|---:|---:|
| SVR | 201.40 | 145.59 | 9.36 |
| ARIMAX | 201.94 | 152.16 | 9.52 |
| LASSO | 204.69 | 158.09 | 10.02 |
| XGBoost | 210.25 | 155.78 | 10.07 |
| 1D-CNN | 239.63 | 151.21 | 9.65 |
| Naive | 239.65 | 156.44 | 9.82 |
| Seasonal Naive | 371.59 | 329.89 | 20.43 |

### Main Findings

1. **SVR achieved the highest forecasting accuracy.** It recorded the lowest RMSE, MAE and MAPE among the seven models evaluated.

2. **ARIMAX remained highly competitive.** The difference between SVR and ARIMAX was small, suggesting that a well-specified conventional time-series model can perform almost as well as a machine learning model.

3. **More complex models did not necessarily perform better.** XGBoost and the 1D-CNN did not outperform ARIMAX on RMSE. The 1D-CNN also showed signs of overfitting during training.

4. **Historical VAT movements and economic indicators mattered.** Past VAT changes, previous-month changes in economic activity (CIEA), exchange rate movements and the December calendar effect were important forecasting factors.

5. **Sudden revenue changes remained difficult to predict.** All models struggled to forecast the sharp increase in Domestic VAT revenue observed in August 2024.

## Conclusions and Recommendations

The study demonstrated that machine learning models can be applied to forecasting Domestic VAT revenue in Ghana. However, their advantage over conventional forecasting methods was limited in this relatively small dataset.

The study recommended that the Ghana Revenue Authority consider using SVR alongside ARIMAX for one-month-ahead forecasts, while continuing to compare predictions with actual monthly collections.

It also recommended recording policy and administrative events, such as VAT rate changes and the rollout of electronic VAT systems, to help explain unusual movements in revenue.

## Limitations

The study had several limitations:

- The relatively small sample size limited the ability of more complex models to learn patterns effectively.
- Only aggregated monthly macroeconomic data were used; taxpayer-level and administrative variables were unavailable.
- The analysis focused on one-month-ahead forecasting.
- The test period contained unusual revenue movements that affected model performance.
- No formal statistical test of differences between forecasting models was conducted.
- The findings were specific to Ghana's Domestic VAT environment.

## Tools and Technologies

**Programming language**
- R (version 4.5.1)

**Statistical and Machine Learning Packages**
- `forecast` – ARIMAX modelling
- `tseries` – ADF and KPSS stationarity tests
- `glmnet` – LASSO regression
- `e1071` – Support Vector Regression
- `xgboost` – Extreme Gradient Boosting
- `keras3` and TensorFlow – 1D-CNN

**Data Management and Visualisation**
- `readxl`
- `dplyr`
- `tidyr`
- `zoo`
- `ggplot2`

**Additional Environment**
- Python 3.11
- TensorFlow 2.21.0
- Keras 3.15.1

## Research Significance

This research contributes empirical evidence on the application of machine learning to Domestic VAT revenue forecasting in Ghana.

It also demonstrates the importance of comparing machine learning algorithms against conventional statistical benchmarks, particularly when working with relatively small time-series datasets.

## Repository Contents

| File | Description |
|---|---|
| `VAT_forecasting_pipeline (2).R` | Main R script containing data preparation, exploratory analysis, forecasting model implementation, model evaluation and visualisation. |
| `dissertation_public.Rmd` | R Markdown source file containing the MSc dissertation, including the research background, methodology, analysis, results, discussion and conclusions. |

**Note:** The dataset (`full_dataset.xlsx`) required to run the R script is not included in this repository.

## Authors

**Edward Afutu Amoni**  
[GitHub](https://github.com/edwardafutu) | [LinkedIn – Add profile link]

**Victoria Naa Adzeley Koney-Mensah**

---

*Joint MSc dissertation submitted to the University of Ghana in November 2026.*
