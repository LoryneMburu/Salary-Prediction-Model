# Salary Prediction Model

## Project Overview
This project develops a machine learning model to predict salaries based on demographic and professional attributes such as age, gender, education level, job title, and years of experience. Using a comprehensive dataset from Kaggle, the model achieves strong predictive performance with an R² score of approximately 0.90.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

## Introduction
Salary prediction models can be valuable for various stakeholders:
- Job seekers determining fair compensation expectations
- HR professionals establishing equitable pay scales
- Organizations benchmarking their compensation structures
- Researchers studying wage trends and potential demographic disparities

This project aims to create an accurate and reliable model for predicting salaries using various personal and professional attributes.

## Dataset
The dataset used in this project contains salary information along with several features:
- Age
- Gender
- Education Level (Bachelor's, Master's, PhD)
- Job Title
- Years of Experience
- Salary (target variable)

The dataset contains approximately 300+ entries and was selected from Kaggle with a rating above 7.

### Data Preprocessing
- Removed missing values
- Handled outliers, particularly extremely low salaries that appeared to be data entry errors
- Converted categorical variables using one-hot encoding
- Standardized numerical features

## Methodology
The project follows a comprehensive data science workflow:

### 1. Exploratory Data Analysis
- Examined distributions of features and target variable
- Analyzed correlations between variables
- Identified patterns and relationships in the data
- Visualized salary distributions by education level, gender, etc.

### 2. Feature Engineering
- Created preprocessing pipelines for categorical and numerical features
- Applied one-hot encoding for categorical variables (Gender, Education Level, Job Title)
- Used standard scaling for numerical features (Age, Years of Experience)

### 3. Model Selection
Evaluated several regression algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 4. Hyperparameter Tuning
- Used GridSearchCV with 5-fold cross-validation to find optimal hyperparameters
- Optimized the best-performing model (Ridge Regression)

### 5. Model Evaluation
- Utilized metrics including R² score, RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error)
- Analyzed residuals to ensure model assumptions were met
- Verified model generalizability through learning curves

## Results
Ridge Regression was identified as the best-performing model with:
- R² Score: ~0.90
- Strong performance across different education levels and job titles
- Minimal overfitting as shown by learning curves
- Key predictors: Years of Experience, Education Level, and Job Title

The model demonstrates good balance between bias and variance as evidenced by the learning curve analysis.

### Key Insights
- Years of Experience is the strongest predictor of salary
- Advanced degrees (PhD, Master's) significantly impact salary levels
- Executive and senior positions show the highest salary variability
- Age contributes to salary prediction but with less impact than experience

## Usage
### Installation
```bash
git clone https://github.com/yourusername/salary-prediction.git
cd salary-prediction
pip install -r requirements.txt
```

### Making Predictions
```python
import pandas as pd
import joblib

def predict_salary(age, gender, education, job_title, experience):
    # Create a dataframe with the input data
    new_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })
    
    # Load the model
    model = joblib.load('salary_prediction_model.pkl')
    
    # Make prediction
    predicted_salary = model.predict(new_data)[0]
    
    return predicted_salary

# Example usage
salary = predict_salary(35, 'Female', "Master's", 'Data Scientist', 7)
print(f"Predicted Salary: ${salary:.2f}")
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Future Work
- Expand dataset with more diverse entries
- Incorporate regional and industry-specific factors
- Develop an interactive web application for real-time predictions
- Include time-series analysis to track salary trends over time
- Add confidence intervals to predictions
- Explore more advanced models like neural networks or ensemble methods

## Author
Loryne Muthoni Mburu

## Acknowledgments
- Data source: Kaggle
- This project was completed as part of a Data Science Capstone Project