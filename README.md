# Walmart Store Sales ForeCasting
As part of this assignment for my Master's class at UCSD, we were provided with historical sales data for 45 Walmart stores
located in different regions, with the objective being to predict forecast sales for all given stores.
The predictive models we have implemented for this dataset try to best capture the most important aspects of the data and make the best possible sales forecast, evaluated by a metric known as the weighted mean absolute error (WMAE).
For Preprocessing we followed the below mentioned step :
* We explored different data features from the training data to pointout which features actually ahve an impact and vary across various data points.
* The intuition behind using these features was which of them worked well, and which of them did not, and the probable reasons for the same have been covered in the pdf attached.
# Model Selection and their WMAE Score
* Simple Multiple Linear Regression - 9304.87
* Support Vector Regression - 8899.96
* Random Forest Regression - 8437.63
* Gradient Boosting Regression - 8335.61
* Simple Multiple Linear Regression - 9304.87
# Libraries used
* Numpy
* Sklearn
