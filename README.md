# **Bank Telecaller Decision Support System** 

![dashboard_preview](/dashboard_shot.png)

## Introduction

This project is a Bank Telecaller Decision Support System. 
The main purpose is to help bank managers and telecallers to target customers who are more likely to subscribe to their product.

This project was pursued for fulfilment of the coursework ECE 229- Compuation data analysis and product development, offered by UC San Diego in Spring 2020. 

---

## Application website

https://ece229.herokuapp.com/

---

## Documentation

Full documentation can be seen at https://hazeltree.github.io/ECE229-Project/

---

## Table of Content

- [Methodology](#methodology)
- [Data source](#datasource)
- [Code](#code)
- [File Structure](#filestructure)
- [Require Packages](#requirepackages)
- [Contributors](#contributors)

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

## Directory Structure

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
|    ├── TODO
|    └── TODO
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
Chenhao Zhou,
Ismail Oack,
Xintong Zhou,
Amol Sakhale,
Harshita Krishna (h1krishn@ucsd.edu)
