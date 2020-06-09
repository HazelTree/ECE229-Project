# **Bank Telecaller Decision Support System** 

![dashboard_preview](/dashboard_shot.png)

## Introduction

This project aims to create a Bank Telecaller Decision Support System using interactive plots and machine learning. The main purpose is to help bank managers and telecallers to target customers who are more likely to purchase their product. 

This Decision Support System maximizes the output of the marketing campaigns by creating accurate success predictions for the future customers. Using this product, telecallers can start calling the customers with high probability of success and maximize their campaign success. In this project, we achieved a 90% prediction accuracy using a XGBoost model.

This project was pursued for fulfilment of the coursework ECE 229- Compuation data analysis and product development, offered by UC San Diego in Spring 2020. 

---

## Application website

https://ece229.herokuapp.com/

---

## Documentation

Full documentation can be seen at https://hazeltree.github.io/ECE229-Project/

---

## Table of Content
- [User Stories](#userstories)
- [Methodology](#methodology)
- [Data source](#datasource)
- [Code](#code)
- [File Structure](#filestructure)
- [Require Packages](#requirepackages)
- [Contributors](#contributors)

---

## User Stories

1- Upper management:
   
   - Understand potential success rate of campaign to optimize campaign investment
   
2- Telecaller:

   - Able to call the customers who are more likely to purchase the product

---

## Methodology

1. Analyzed important features, created insightful plots to be used in macro-level decision processes.

2. Developed statistical models to predict probability of success.

   Models used: Logistic Regression, XGBoost, Random Forest

3. Created live prediction functionality to be used in both strategic and daily decisions.


----

## Data source

The UCI dataset can be found in the `./data/` directory. the`.txt` file is an introduction to the dataset.

The original data can be found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing `./data/`

---

## Code

### Data Analysis
- [pre_processing.py](../master/src/pre_processing.py) : Data loading, cleaning and pre-processing
- [analysis.py](../master/visualization/analysis.py) : Analysis tools, plots and visualizations
### Feature Extraction
- [feature_extraction.py](src/feature_extraction.py) : Feature extraction and preparing the data for ML models
### Prediction
- [prediction.py](src/prediction.py) : Prediction on the test data using Logistic Reg., XGBoost, and Random Forest models
### Dashboard
- [dashboard.py](dashboard.py) : Dash app of the product, includes all plots and capabilities of the dashboard


---

## File Structure

```
├── dataset
|   ├── bank-additional-full.csv
|   ├── bank-additional-full.csv
|   └── bank-additional.csv
├── src
|    ├── pre_processing.py
|    ├── feature_extraction.py
|    └── prediction.py
├── doc
|    ├── build
|    |   ├── doctrees
|    |   |    ├── environment.pickle
|    |   |    └── index.doctree
|    |   └── html
|    |   |    ├── _sources
|    |   |    ├── _static
|    |   |    ├── .buildinfo
|    |   |    ├── genindex.html
|    |   |    ├── index.html
|    |   |    ├── objects.inv
|    |   |    ├── py-modindex.html
|    |   |    ├── search.html
|    |   |    └── searchindex.js
|    ├── source
|    |   |    ├── conf.py
|    |   |    └── index.rst
|    ├── Makefile
|    └── make.bat
├── test
|    ├── generate_coverage_report.py
|    ├── test_analysis.py
|    ├── test_dashboard.py
|    ├── test_feature_extraction.py
|    ├── test_pre_processing.py
|    ├── test_prediction.py
|    └── test_util.py
├── visualization
|    ├── analysis.py
|    └── 229_plots.ipynb
├── Dockerfile
├── LR_prediction.joblib
├── README.md
├── dashboard.py
├── docker-compose.yml
├── requirements.txt
├── temp-plot.html
└──  util.py
```
---

## Required Packages
- Python >=3.7
- dash
- dash-renderer
- dash-core-components
- dash-html-components
- dash-table
- plotly
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- xgboost
- scikit-learn
- coverage
- pytest
- joblib

Run pip install -r requirements.txt to set up your computer 

---

## Contributors
| <a href="https://github.com/HazelTree" target="_blank">**Chenhao Zhou**</a> | <a href="https://github.com/harshita1804" target="_blank">**Harshita Krishna**</a> | <a href="https://github.com/ShikaZzz" target="_blank">**Xintong Zhou**</a> | <a href="https://github.com/AmolSakhale" target="_blank">**Amol Sakhale**</a> | <a href="https://github.com/iocak28" target="_blank">**Ismail Ocak**</a> |
| :---: |:---:| :---:| :---:| :---:|
| [![Chenhao Zhou](https://avatars1.githubusercontent.com/u/42596038?s=400)](https://github.com/HazelTree) | [![Harshita Krishna](https://avatars1.githubusercontent.com/u/42465912?s=400)](https://github.com/harshita1804) | [![Xintong Zhou](https://avatars1.githubusercontent.com/u/56456781?s=400)](https://github.com/ShikaZzz) | [![Amol Sakhale](https://avatars1.githubusercontent.com/u/10459987?s=400)](https://github.com/AmolSakhale) | [![Ismail Ocak](https://avatars3.githubusercontent.com/u/14804342?s=400)](https://github.com/iocak28) |
| <a href="https://github.com/HazelTree" target="_blank">`github.com/HazelTree`</a> | <a href="https://github.com/harshita1804" target="_blank">`github.com/harshita1804`</a> | <a href="https://github.com/ShikaZzz" target="_blank">`github.com/ShikaZzz`</a> | <a href="https://github.com/AmolSakhale" target="_blank">`github.com/AmolSakhale`</a> | <a href="https://github.com/iocak28" target="_blank">`github.com/iocak28`</a> |
