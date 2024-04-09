# Solar-Irradiance-to-Sustainably-Power-Streetlights
Predicting Solar Irradiance to Sustainably Power Streetlights

The world is moving towards a new era of harnessing various renewable energy sources. One such consideration is solar energy, which holds substantial promise in the face of global energy challenges, especially in California. Solar energy provided 27% of California's total electricity net generation, including small-scale solar generation [1]. The state has been a leader in solar energy production, with more than 17,500 megawatts of utility-scale solar power capacity and almost 32,000 megawatts of total solar capacity, making it the nation's top producer of electricity from solar energy [1].

Powering streetlights can significantly benefit from solar energy, particularly in California, where insufficient lighting creates safety hazards for pedestrians and drivers. Our study focuses on the possibility of making streetlights on highways solar-powered by predicting irradiance. Solar irradiance is the quantity of solar energy that reaches a particular surface of the Earth at a specific angle. This solar energy, specifically solar irradiance, can be converted into electricity using photovoltaic cells. We can enhance reliability and reduce uncertainties in generating sustainable energy by employing machine learning models to predict solar irradiance. Our research aims to predict solar irradiance by quantifying its relationship with other environmental factors through supervised regression models to aid in the shift toward sustainable energy solutions.

# Data Overview
Data Collection: Extracted data from the National Solar Radiation Database [2] portal by providing specific California zip codes for the years 2018-2022

Data Merging
The datasets extracted from the NREL data explorer portal contain only latitude and longitude data for a given input zip code - latitude and longitude are not intuitive features
For a particular zip code, there are five datasets that we get - 1 for each year between 2018 and 2022; these are stored manually in folders labeled with the input zip code
By following this structure, we wrote a Python script to read multiple files from each folder, affix zip code as a column extracted from the folder name, and then append all these datasets together into a master data frame - this ensures that we can differentiate between the multiple datasets combined

# Data preprocessing
Consolidated the hourly data across locations and years into a master dataframe using Python. Additionally, we've decided to limit zero values in the target variable, i.e., Clearsky DHI column, since DHI is 0 during night-time. This step is crucial for refining the dataset and ensuring accurate insights

# Feature exploration
We have excluded features that are unintuitive while also transforming hours into meaningful day-part representations for ease of understanding.

Numerical Features: The dataset comprises several numerical features, including temporal data, weather attributes, solar irradiance measurements, and location information. These features show diverse ranges and distributions, some normally distributed and others skewed. The correlation heatmap shows strong positive correlations among solar irradiance measurements (Clearsky DHI, Clearsky DNI, Clearsky GHI, DHI, DNI, GHI), indicating that we need to cut out some highly correlated features to avoid multicollinearity issues.

Categorical Feature: The dataset features some categorical variables already present in the data, along with some variables that will likely be encoded.

Temporal Features: Temporal features (year, month, day) help show the change in various metrics over time we have considered data from 2018-2022 only.

Feature Engineering Opportunities: There are opportunities to combine temporal features to capture seasonal patterns and encode some categorical features.

# Exploratory Data Analysis (EDA)

Explored the dataset's characteristics. With over 1.3 million rows and multiple columns, the dataset contains many features, including temperature, various solar irradiance metrics, weather attributes, and location data.

Exploratory Visualizations:  
The target variable, Clearsky DHI, exhibits a right-skewed distribution likely due to the presence of night-time values that are inflating the volume of 0 value.

# Predictive Modeling 

Employing regression models, including linear regression, support vector machine (SVM) regression, random forest regression, and polynomial regression, to predict solar irradiance based on the selected features. Splitting the dataset into training and testing sets, and train each regression model using the training data. Optimize the hyperparameters of each regression model using techniques such as grid search or randomized search to improve performance. Evaluate the performance of each regression model using appropriate metrics, including mean squared error (MSE), coefficient of determination (Rsquared), Root Mean Square Error (RMSE) and Mean Average Percent Error (MAPE).Compare the regression models based on their performance metrics to identify the most effective model for predicting solar irradiance.

Linear Regression: Utilize linear regression to model the relationship between the selected features and solar irradiance, modeling for a linear relationship between the dependent and independent variables

SVM Regressor: Apply support vector machine regression to capture complex relationships between the independent features and solar irradiance, leveraging the kernel trick to map the input features into a higher-dimensional space

Random Forest Regression: Employ random forest regression, an ensemble learning technique, to predict solar irradiance by aggregating the predictions of multiple decision trees trained on random subsets of the data

Polynomial Regression: Employ polynomial regression to capture nonlinear relationships between the features and solar irradiance by fitting a polynomial function to the data

# File Structure
`Data Collection and Merging.ipynb` : This python file contains the data collection for zip codes. Data is collected for five years and merged into single csv file for data preprocessing. Data is collected for certain zip codes in california.

# References

[1] https://www.eia.gov/state/analysis.php?sid=CA
[2] https://nsrdb.nrel.gov/data-viewer

https://www.ncei.noaa.gov/thredds/catalog/avhrr-patmos-x-cloudprops-noaa-des-fc/catalog.html?dataset=avhrr-patmos-x-cloudprops-noaa-des-fc/PATMOS-X_Cloud_Properties:_Aggregation,_NOAA_descending_best.ncd

https://nsrdb.nrel.gov/data-sets/us-data 



# Contact Information 
Neha Sharma : neha.sharma01@sjsu.edu
Nikitha Goturi : nikitha.goturi@sjsu.edu
Sandeep Reddy Potula : sandeepreddy.potula@sjsu.edu
Veena Ramesh Beknal : veenaramesh.beknal@sjsu.edu
