# Diabetes Prediction

Our goal is to explore data related to the incidence of diabetes among Pima Indian women and predict whether women from this population are likely to have type-2 diabetes or not.

# Problem statement

Pima is typically referred to as “River People” a group of native North American Indians living in Arizona, USA. More than half of all Pima Indians over age 35 have diabetes, a condition arising from a body's decreased ability to metabolize glucose (Knowler et al. 1990). Adult-onset, non-insulin-dependent diabetes, now called type 2 Diabetes, is the most common form of diabetes among Pima Indians due to several historical and sociological factors. In order to understand diabetes better, we are studying the Pima Indian Diabetes dataset which includes females from age 21 – 81 from the Pima Indian heritage. 

The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. We are using supervised machine learning models to predict the ‘Outcome’ (target variable in our dataset) along with exploratory data analyses to understand further the nuances of relationships between the biological parameters and the prevalence of diabetes. The real-world implications among the Pima Indian population are crucial for healthcare professionals, aiding in identifying high-risk individuals.

# Data Overview

There are 8 features in our dataset: pregnancies, glucose, blood pressure, insulin, skin thickness, BMI, diabetes pedigree function, age, and the target variable is outcome. These variables in our dataset have varying degrees of outlier and missing value presence. Each independent variable has been studied in detail to arrive at the appropriate treatment strategy. Due to the dataset being small, we have decided to retain all records post treatment and preserve the integrity of the original dataset as much as possible.

# Data exploration summary

Based on our exploratory analysis, there is evidence to suggest that older Pima women with higher Insulin, Glucose levels and considerably overweight are at at greater risk of being diabetic. Below is the summary for all the features of the dataset:

> Pima women who are 30 years or older are 2.3x more likely to be diabetic than otherwise*
> Insulin post imputation became a multimodal distribution for both diabetic and non-diabetic Pima women, with the difference in Insulin >= 150 being 2.7x more likely being diabetic than lower readings* 
> Pima women with >= 120 glucose levels were 3x more likely to be diabetic than otherwise*
> Women with Skin Thickness >= 30 are 2.5x more likely to be diabetic than otherwise*
> Pima women with BMI >= 30 are 2.7x more likely to be diabetic than otherwise*
> Pima women with >= 7 pregnancies are 2x more likely to be diabetic than those with lower*
> Pima women with Blood Pressure >= 82 are 1.5x more likely to diabetic than otherwise*
> Pima women with Diabetes Pedigree Function >= 0.5 are 1.6 times more likely to be diabetic than otherwise*

# Predictive Modeling

Our goal is to predict if a Pima woman is diabetic or not based on the provided features. Hence, we experimented with 4 classification algorithms to get a range of models that we can choose from based on the criteria of model performance like accuracy, precision, recall, explainability, etc. We chose to experiment with 4 tree-based models since these are intuitive to understand, perform well with different types of data and certain algorithms are less prone to data issues like missing values, outliers, etc. For our project, we used the following 4 algorithms: 

● Decision trees
● Random Forest 
● Gradient Boosted Machine 
● Extreme Gradient Boosting or XGBoost

# Solution summary and key takeaways

Three imputation strategies were explored based on univariate and bivariate analyses: 
● Winsorization for pregnancies
● Class mean imputation for Skin Thickness & Insulin
● Mean/median imputation for the rest features

By using the variable treatment approach and the model in our project, we can predict with a fairly high degree of accuracy whether any new Pima Indian woman is at risk of diabetes or not. Shapley values are useful in determining the model explainability - they provide the importance of the variables along with the direction of influence.
    ○ This might aid medical professionals in early medical intervention to manage type 
        2 diabetes.
    ○ In particular, we can recommend that older Pima women with high Insulin and 
        Glucose readings are at higher risk of type 2 diabetes and hence, should be 
        prioritized for preventive care

From our findings, Gradient Boosted Machine performed the best with accuracy of 73.59%, precision of 64.29% and recall of 55.56%. We observe AUC (area under curve) of 0.82

We see that the accuracy of the baseline model is slightly higher and this is likely due to the 
influence of the 0s in Insulin and Skin Thickness
● Not much change is observed in the Recall and AUC though
● A considerable drop in accuracy is observed as compared to our original approach
● In both cases, Glucose and BMI are the top 2 important variables
Studying the model results, we can infer that older Pima women who have higher Insulin and Glucose readings are at higher risk of diabetes 


# File Structure

`DataSet\PimaIndianDiabetesData`: Detailed report on the approach,analysis and outcome from the dataset

`PythonFile\PIMA_Diabetes_EDA_Modelling`: This python file contains the EDA(Univariate and bivariate analysis), imputation methods and predictive modeling for the dataset

`Report\Report`: Detailed report on the approach,analysis and outcome from the dataset


# References

● High-Risk Populations: The Pimas of Arizona and Mexico:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4418458/#:~:text=By%201970%2C%20
the%20prevalence%20of,similarly%20aged%20Caucasians%20%5B5%5D
● Reducing Diabetes in Indian Country: Lessons from the Three Domains Influencing Pima 
Diabetes:
https://www.smu.edu/~/media/Site/Dedman/Departments/Anthropology/pdf/Smith-Morris/HO%20Virchow.ashx
● PIMA Indian Diabetes dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
● Effect of pregnancies on diabetes:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5124395/
● Relationship between Insulin and Glucose: https://pubmed.ncbi.nlm.nih.gov/8422798/
● Relationship between Insulin and Glucose: https://pubmed.ncbi.nlm.nih.gov/8898771/
● Women have a higher baseline TSF and hence upper limit of 30 can be considered [study: 
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9127233/#:~:text=In%20the%20study
%20population%2C%20the,vs%2014.3%20%C2%B1%206.8%20mm).] 
● Insulin and age study for Pima women: 
https://www.foodandnutritionjournal.org/volume7number2/significance-of-health-related-predictors-of-diabetes-in-pima-indians-women/
● BMI vs diabetes: https://pubmed.ncbi.nlm.nih.gov/2031485/