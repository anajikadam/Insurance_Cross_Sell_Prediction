# Vehicle Insurance Cross Sell Prediction
Predicting whether a customer would be interested in buying Vehicle Insurance so that the company can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.


---

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](#data-wrangling)
    - 2 [Normalization](#normalization)
    - 3 [EDA](#eda)
    - 4 [ Encoding categorical values](#encoding-categorical-values)
    - 5 [Feature Selection](#feature-selection)
    - 6 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
  * [Conclusion](#conclusion)
  * [Reference](#reference)

---



# Abstract
An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. There are multiple factors that play a major role in capturing customers for any insurance policy. 

Here we have information about demographics such as age, gender, region code, and vehicle damage, vehicle age, annual premium, policy sourcing channel.

Based on the previous trend, this data analysis and prediction with machine learning models can help us understand what are the pattern for customers, who show interest for buying Vehicle Insurance and obtain the best classification model.

# Problem Statement
Our client is an Insurance company that has provided Health Insurance to its customers. Now they need the help in building a model to predict whether the policyholders (customers) from the past year will also be interested in Vehicle Insurance provided by the company.

An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.

**Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue.**


# Data Description
We have a dataset which contains information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc. related to a person who is interested in vehicle insurance.
We have 381109 data points available.

- `id` : Unique ID for the customer

- `Gender` : Gender of the customer

- `Age` : Age of the customer

- `Driving_License` : 0 : Customer does not have DL, 1 : Customer already has DL

- `Region_Code` : Unique code for the region of the customer

- `Previously_Insured` : 1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance

- `Vehicle_Age` : Age of the Vehicle

- `Vehicle_Damage` :1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.

- `Annual_Premium` : The amount customer needs to pay as premium in the year

- `PolicySalesChannel` : Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.

- `Vintage` : Number of Days, Customer has been associated with the company

- `Response` (Dependent Feature): 1 : Customer is interested, 0 : Customer is not interested


----

# Project Outline

## 1. Data Wrangling
After loading our dataset, we observed that our dataset has 381109 rows and 12 columns. We applied a null check and found that our data set has no null values. Further, we treated the outliers in our dataset using a quantile method.


### Outlier Treatment

--


## 2. Normalization
After outlier treatment, we observed that the values in the numeric columns were of different scales, so we applied the min-max scaler technique for feature scaling and normalization of data.

## 3. EDA
In Exploratory Data Analysis, firstly we explored the 4 numerical features: `Age, Policy_Sales_Channel, Region_Code, Vintage`. 
Further, we categorized `age as youngAge, middleAge, and oldAge` and also categorized `policy_sales_channel` and `region_code`. 
From here we observed that customers belonging to the youngAge group are less interested in taking vehicle insurance. 

Similarly, Region_C, Channel_A have the highest number of customers who are not interested in insurance. 

From the vehicle_Damage feature, we were able to conclude that customers with vehicle damage are more likely to take vehicle insurance. 

Similarly, the Annual Premium for customers with vehicle damage history is higher.

## 4. Encoding categorical values
We used one-hot encoding for converting the categorical columns such as 'Gender', 'Previously_Insured','Vehicle_Age','Vehicle_Damage', 'Age_Group', 'Policy_Sales_Channel_Categorical', 'Region_Code_Categorical' into numerical values so that our model can understand and extract valuable information from these columns.

## 5. Feature Selection
At first, we obtained the correlation between numeric features through Kendall’s Rank Correlation to understand their relation. We had two numerical features, i.e. Annual_Premium and Vintage. 
For categorical features, we tried to see the feature importance through Mutual Information.  It measures how much one random variable tells us about another.



## 6. Model Fitting
For modeling, we tried the various classification algorithms like:

#### i. Decision Tree 
Decision Trees are non-parametric supervised learning methods, capable of finding complex non-linear relationships in the data. Decision trees are a type of algorithm that uses a tree-like system of conditional control statements to create the machine learning model. A decision tree observes features of an object and trains a model in the structure of a tree to predict data in the future to produce output.
For classification trees, it is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

#### ii. Gaussian Naive Bayes
Gaussian Naive Bayes is based on Bayes’ Theorem and has a strong assumption that predictors should be independent of each other. For example, Should we give a Loan applicant depending on the applicant’s income, age, previous loan, location, and transaction history? In real-life scenarios, it is most unlikely that data points don’t interact with each other but surprisingly Gaussian Naive Bayes performs well in that situation. Hence, this assumption is called class conditional independence.

#### iii. AdaBoost Classifier
Boosting is a class of ensemble machine learning algorithms that involve combining the predictions from many weak learners. A weak learner is a very simple model, although has some skill on the dataset. Boosting was a theoretical concept long before a practical algorithm could be developed, and the AdaBoost (adaptive boosting) algorithm was the first successful approach for the idea.
The AdaBoost algorithm involves using very short (one-level) decision trees as weak learners that are added sequentially to the ensemble. Each subsequent model attempts to correct the predictions made by the model before it in the sequence. This is achieved by weighing the training dataset to put more focus on training examples on which prior models made prediction errors.

#### iv. Bagging Classifier
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

#### v. LightGBM 
Light GBM is a gradient boosting framework that uses tree-based learning algorithms. Light GBM grows trees vertically while other algorithms grow trees horizontally meaning that Light GBM grows trees leaf-wise while other algorithms grow level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, a Leaf-wise algorithm can reduce more loss than a level-wise algorithm. Light GBM is prefixed as ‘Light’ because of its high speed. Light GBM can handle the large size of data and takes lower memory to run.

#### vi. Logistic Regression
Logistic regression is named for the function used at the core of the method, the logistic function.

The logistic function, also called the sigmoid function, was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.


## 7. Hyperparameter Tuning
Tuning of hyperparameters is necessary for modeling to obtain better accuracy and to avoid overfitting. In our project, we used following techniques:

#### - GridSearchCV
#### - RandomizedSearchCV
#### - HalvingRandomizedSearchCV




## 8. Metrics Evaluation

To evaluate our model and to obtain the accuracy and error rate of our models before and after hyperparameter tuning. We used some metric evaluation technique.
They are:
#### i. Confusion Matrix
#### ii. Accuracy
#### iii. Precision
#### iv. Recall
#### v. F1-Score
#### vi. ROC-AUC Score
#### vii. Log Loss


# Challenges Faced
- Handling Large dataset.
- Already available methods of hyper-parameter tuning were taking a huge amount of time to process.
- Memory Optimization during hyperparameter tuning.


# Conclusion
In problem, we have to find those customers only who are interested for taking next (Vehicle) Insurance. so we can targeted those customers only for markiting...

so here we focus on the correctly Predicted Positive value, (What proportion of Correctly Positive Predicted by the Actual Positive)...

Means we can check our model has High Recall score. (Actual number of customer who are interested in take Insurance, what number of customers predicted Positive by ML model who are intersted....)

So Highest Recall at Before tuning GaussianNB Classifier, But After tuning classifier Accuracy increases and Log_loss decreases.

Precision
Our stackholder is interested in All the Customers who might be interested in taking Insurance Predicted by ML model. means taking some risk bases on we can Use Precision as evalution matrix in this stage.

F1-score
If Both (Precision And Recall) Equal Important for model evaluation, then can use F1-score, which gives Harmonic mean of Precision and Recall.

GaussianNB Classifier is the Best model, F1-score is 0.42

For Balanced (by Under-sampling) data set, All the matrix increases but due to undersampling reduce majority class data upto minority class, so more amount data is not used.
Here also GaussianNB Classifier is the Best model, F1-score is 0.82, and Recall and Accuracy also increases.

---
