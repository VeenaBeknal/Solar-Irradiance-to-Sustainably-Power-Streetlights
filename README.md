# Predicting Solar Irradiance to Sustainably Power Streetlights

# Overview of Project

The world is moving towards a new era of harnessing various renewable energy sources. One such consideration is solar energy, which holds substantial promise in the face of global energy challenges, especially in California. Powering streetlights can significantly benefit from solar energy, particularly in California, where insufficient lighting creates safety hazards for pedestrians and drivers. Our study focuses on the possibility of making streetlights on highways solar-powered by predicting irradiance. We can enhance reliability and reduce uncertainties in generating sustainable energy by employing machine learning models to predict solar irradiance. This can also help design a smart power grid that uses energy efficiently. Our research aims to predict solar irradiance by quantifying its relationship with other environmental factors through supervised regression models to aid in the shift toward sustainable energy solutions.The goal of this project is to predict the solar irradiance that can be utilized to convert the irradiance into electricity. This electricity can be used to light up the streetlights which can be a sustainable solution for clean energy.

# Data Overview

Solar irradiance prediction relies on diverse data sources, including ground-based solar monitoring stations, satellite data, and advanced modeling techniques
We utilized the National Solar Radiation Database (NSRDB), a comprehensive resource created by integrating data from various sources.

The NSRDB combines ground-based solar monitoring, NASA satellite data, and advanced modeling to create a comprehensive dataset of solar radiation for specific locations and time periods.

Data collected from NSRDB contained detailed solar irradiance data for selected zip codes across San Jose, California.Data spanned from 2018 to 2022, enabling a comprehensive analysis of solar irradiance. patterns and trends over time

# Data Preprocessing

This phase aimed to enhance data quality, reduce noise, and extract relevant features, ultimately improving the performance and interpretability of the subsequent modeling techniques. The data preprocessing steps we included are data filtering, handling missing and irrelevant data, feature selection, and derived feature creation. The original dataset acquired from NREL contained solar irradiance data for multiple zip codes in San Jose, CA. To focus on the areas with a higher population concentration, the data was filtered to include only the top 10 zip codes, accounting for over 50% of San Jose's population. Rows with missing or zero values for irradiance during night/dark hours were to be removed. Still, since there was no uniform rule that could be applied, we had to split a day into eight parts, i.e., “late_night_start_of_day” (12 AM - 5 AM), “early_morning” (5 AM - 8 AM), “morning” (8 AM - 12 PM), “afternoon” (12 PM - 4 PM), “evening” (4 PM - 6 PM), “late_evening” (6 PM - 8 PM), “early_night” (8 PM - 10 PM) and “late_night” if otherwise. Based on looking at the distribution of data across day parts, we observed that 'early_night', 'late_evening', 'late_night' and 'late_night_start_of_day' day part values showed >50% as 0 and were hence removed.

# EDA

We performed univariate and bivariate analysis of the data. Visualization can be found in the python notebook.

# Feature Engineering
After analysis done in the preprocessing, we plotted the correlation heatmap to check how the features are correlated as shown in the following slide
We found that features like "Clearsky GHI", "Clearsky DHI", "Clearsky DNI", "DHI", ”DNI”, "Solar Zenith Angle," and “Relative Humidity” were highly correlated
We decided to have a cut of 0.85 for positively correlation (>0.85) and negative correlation (<-0.85). We excluded the correlated features mentioned above from the dataset to address potential multicollinearity issues.  The “minute” column, which was constant across the data, was also dropped. Categorical variables such as “Cloud Type”, “zip code,” and “hour_day_part” were one-hot encoded to represent them as binary variables, facilitating their inclusion in the modeling process.

# Machine Learning 

Six supervised ML algorithms were attempted to predict GHI (target Variable) 

Linear Regression: Linear Regression is an appropriate machine learning algorithm, and it maps a simple correlation between the dependent variable and the independent categorical features like temperature, wind speed, dew point, etc.

Random Forest Regression: Random Forest employs the method of ensemble that joins various decision trees to come up with a prediction.It can best recognize the non-linear behavior within the data and probably overcompensate the prediction accuracy when compared with the linear regression.

XGBoost Regression: XGBoost is an ensemble learning technique based on gradient boosting. It trains multiple decision trees sequentially; each tree learns from the mistakes of the one before it, getting better at predicting solar irradiance with each step. XGBoost is known for its ability to achieve high prediction accuracy on various regression tasks; it can effectively learn complex non-linear relationships between solar irradiance and meteorological data.

Support Vector Regression: Support Vector Machines find a hyperplane in the feature space that best separates the data points belonging to different classes. This hyperplane is then used for prediction. SVMs can handle datasets with many features (various meteorological parameters).

L1 (Lasso) Regularization Regression: LASSO (Least Absolute Shrinkage and Selection Operator) regression is a linear regression technique incorporating L1 regularization. This regularization penalizes the absolute values of the coefficients in the linear regression model, encouraging sparsity. In simpler terms, LASSO forces some coefficients to become exactly zero, effectively removing those features from the model.

L2 (Ridge) Regularization Regression: Ridge Regression uses a linear regression technique that incorporates L2 regularization. This regularization penalizes the squared values of the coefficients in the linear regression model. While LASSO encourages sparsity by driving some coefficients to zero, Ridge Regression shrinks all coefficients towards zero, but typically not all the way to zero.

The predictive performance of various machine learning regression models was evaluated using diverse evaluation metrics like MSE, RMSE, MAE, MAPE, MEDAE, R-Squared, and MSLE. This section details the performance of each of the machine learning models.

### Outcomes from modeling
Random Forest Regression was the most versatile of the lot with 0.97 r-squared and we were able to balance the bias-variance trade-off with it. XGBoost Regression is over-fitting.Linear regression showed stable performance but it’s performance was limited given the most important variable was the intercept. Python file contains the electricity calculation as well.

# Conclusion 

Among the models evaluated, the Random Forest Regression model emerged as the best performer, with a 0.97 R-squared value.

The analysis involved extensive data preprocessing, EDA, modeling with various techniques, feature importance analysis, model validation, and simulations for practical applications.

The Random Forest Regression model emerged as the top-performing model, providing insights into the most important features and demonstrating the potential for estimating energy generation and consumption in real-world scenarios.

A simulation was performed to calculate daily energy generation per solar panel based on the predicted irradiance values from the models, considering factors like panel area, panel efficiency, hours of sunlight, and system losses.

# Recommendations
An example calculation is provided for estimating the total daily energy consumption for street lights based on the number of street lights, their wattage, and operating hours.

Experiment with ensemble techniques, such as bagging, boosting, or stacking, to combine the strengths of different models and potentially improve overall predictive performance.

Explore additional use cases and applications for solar irradiance prediction models, such as solar farm planning, energy load forecasting, or grid optimization.

Collaborate with domain experts and stakeholders to identify specific requirements and tailor the models to meet their needs.

Experiment with sophisticated techniques for feature engineering to ensure that the models generalize well for regions outside of the areas used for training.


## File Structure

`Pickle_LR_Model.pkl`, `Pickle_best_RF_Model.pkl`, `Pickle_best_xgb_Model.pkl`, `Pickle_lasso_Model.pkl`,`Pickle_ridge_Model.pkl`,`Pickle_xgb_Model.pkl`: These the pickle files for our project

`Data Collection and Merging.ipynb`: Python file containing the data collect and merging for solar data for specific zip codes of San Jose city

`Data Collection and Merging.py`: Python file containing the data collect and merging for solar data for specific zip codes of San Jose city

`SolarIrradiance_EDA_Modeling_Final_Results.ipynb`: Python file containing the EDA, Feature Engineering and Modeling for solar data on San Jose city zip code

`SolarIrradiance_EDA_Modeling_Final_Results.py`: : Python file containing the EDA, Feature Engineering and Modeling for solar data on San Jose city zip code

`Github Copilot - Screenshots.pdf`: File containg the screenshots of the pair programming

`Grammarly - Sceenshots.pdf`: File containg the screenshots of the Grammarly content validation

`Agile-Trello Sceenshots.pdf`: File containg the screenshots of the Agile Methodology 


## Contact Information

For inquiries or feedback, contact:

- Neha Sharma: neha.sharma01@sjsu.edu

- Nikitha Goturi: Nikitha.goturi@sjsu.edu

- Sandeep Reddy Potula: sandeepreddy.potula@sjsu.edu

- Veenaramesh Beknal: veenaramesh.beknal@sjsu.edu

Explore the project and leverage insights from the Solar Irradiance data!

"Keep your face to the sun and you will never see the shadows" - HELLEN KELLER
