# Financial Inclusion in Africa
## Project Overview
The goal of this project is to build a machine learning model that can predict which individuals in East Africa are most likely to have or use a bank account. This is part of the broader effort to achieve financial inclusion, which means ensuring that individuals and businesses have access to useful and affordable financial products and services.

## Dataset
The dataset used in this project contains demographic information and financial service usage data for approximately 33,600 individuals across East Africa. The dataset is provided as part of the Financial Inclusion in Africa challenge hosted on the Zindi platform.

## Approach
### Data Exploration and Preprocessing:
Installed the necessary packages
Imported the data and performed basic data exploration
Generated a pandas profiling report to gain insights into the dataset
Handled missing and corrupted values
Removed any duplicates
Used the boxplot method to identify outliers, and then applied the winsorization method to handle them
Encoded categorical features using LabelEncoder

## Model Training and Evaluation:
Trained and tested a machine learning classifier based on the preprocessed data, using the Random Forest Classifier algorithm
The features used to train the model were: education level, age of respondent, cellphone access, job type, and country
The perfromance of the model was as follows:

              precision    recall  f1-score   support

           0       0.94      0.84      0.88      6073
           1       0.40      0.67      0.50       985

    accuracy                           0.81      7058


Using the f1 score metric, 88% of 0's were predicted correctly and 50% of 1's were predicted correctly

This concludes that the model was not performing well in predicting 1's which were the folks with bank accounts as compares to predicting zeros, which were folks with no bank account. Thus this suggests an instance of imbalanced data on the target variable

## Streamlit Application:
Created a Streamlit application that allows users to input feature values and receive a prediction of whether the individual is likely to have a bank account
Deployed the Streamlit application to Streamlit Share

## Usage
To use the Financial Inclusion in Africa application, follow these steps:

Visit the Streamlit Share deployment: Financial Inclusion in Africa
Enter the required feature values in the input fields
Click the "Validate" button to receive the prediction

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, with emphasis in improving the models performance, please feel free to create a new issue or submit a pull request.
