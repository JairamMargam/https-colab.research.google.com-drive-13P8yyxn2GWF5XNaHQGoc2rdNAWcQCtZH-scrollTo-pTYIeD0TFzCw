# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:18:29 2024

@author: Jai Ram
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE # SMOTE - Synthetic Minority Over-sampling Technique
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skmet
import joblib

# Loading the data set
data = pd.read_csv(r"C:\Users\Jai Ram\Documents\mini project\sms_raw_NB.csv", encoding = "Latin-1")

# Mapping the type to numeric values 1 and 0.
# This step is required for metric calculations in the model evaluation phase.

data['spam'] = np.where(data.type == 'spam', 1, 0)

# assigning data dataset to email_data
email_data = data


# In[ ]:


# Imbalance check
email_data.type.value_counts()
email_data.type.value_counts() / len(email_data.type)  # values in percentages


# In[ ]:


# Splitting dataset into Training and testing with 80 - 20 %
# Data Split
email_train, email_test = train_test_split(email_data, test_size = 0.2, stratify = email_data[['spam']], random_state = 0) 


# In[ ]:


# CountVectorizer: Convert a collection of text documents to a matrix of token counts
countvectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

# creating a matrix of token counts for the entire text document
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)

# Defining BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text)

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)


# In[ ]:


# SMOTE technique to handle class imbalance.
# Oversampling can be a good option when we have a class imbalance.creates duplicats examples
smote = SMOTE(random_state = 0)

# Transform the dataset
X_train, y_train = smote.fit_resample(train_emails_matrix, email_train.spam) # defining Input and Output.

y_train.unique()
y_train.values.sum()   # Number of 1s
y_train.size - y_train.values.sum()  # Number of 0s
# The data is now balanced


# In[ ]:


# Model Builging - Multinomial Naive Bayes
# Traning model 
classifier_mb = MultinomialNB()
classifier_mb.fit(X_train, y_train)

# predicting on unseen dataset
test_pred_m = classifier_mb.predict(test_emails_matrix)

# Crosstable
pd.crosstab(email_test.spam, test_pred_m)

# Accuracy Score
accuracy_test_m = skmet.accuracy_score(email_test.spam, test_pred_m)
print(accuracy_test_m) # 0.9703237410071942


# In[ ]:


# Heat map
cm = skmet.confusion_matrix(email_test.spam, test_pred_m)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not Spam', 'Spam'])
cmplot.plot()
cmplot.ax_.set(title = 'Spam Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# In[ ]:


# once predicting on seen dataset to compare accuracy ,how it learned 
train_pred_m = classifier_mb.predict(train_emails_matrix)

# Crosstab
pd.crosstab(email_train.spam, train_pred_m)

# Accuracy
accuracy_train_m = skmet.accuracy_score(email_train.spam, train_pred_m)
print(accuracy_train_m) # 0.991005172026085


# In[ ]:


# Hyperparameter Tuning
# Multinomial Naive Bayes changing default alpha for Laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# The smoothing process mainly solves the emergence of a zero probability problem in the dataset.

# Traning model-
mnb_lap = MultinomialNB(alpha = 5)
mnb_lap.fit(X_train, y_train)


# In[ ]:


# Evaluation on Test Data after applying Laplace
# predicting on test data
test_pred_lap = mnb_lap.predict(test_emails_matrix)
# crosstab
pd.crosstab(test_pred_lap, email_test.type)

# Accuracy
accuracy_test_lap = skmet.accuracy_score(email_test.spam, test_pred_lap)
print(accuracy_test_lap) # 0.966726618705036


# In[ ]:


# Heat Map
cm = skmet.confusion_matrix(email_test.spam, test_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not Spam', 'Spam'])
cmplot.plot()
cmplot.ax_.set(title = 'Spam Detection Confusion Matrix',
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# In[ ]:


# Training Data accuracy
# once predicting on seen dataset to compare accuracy ,how it learned 
# Trailing on Train dataset
train_pred_lap = mnb_lap.predict(train_emails_matrix)
# crosstab
pd.crosstab(train_pred_lap, email_train.spam)
# Accuracy
accuracy_train_lap = skmet.accuracy_score(email_train.spam, train_pred_lap)
print(accuracy_train_lap) # 0.9804362491567349


# In[ ]:


# Feature Engineering


# In[ ]:


# Saving the Best Model using Pipelines
# Preparing a Naive Bayes model on a training data set
nb = MultinomialNB()

# Defining Pipeline
pipe1 = make_pipeline(countvectorizer, smote, nb)

# Fit the train data
processed = pipe1.fit(email_train.text.ravel(), email_train.spam.ravel()) # ravel is to convert data shape into 1d-array coz its processing time is fast.

# Save the trained model
joblib.dump(processed, 'processed1')

# load the saved model for predictions
model = joblib.load('processed1')


#example:
predicts = model.predict(email_test.text.ravel())


# In[ ]:


dense_matrix = X_train.toarray()

# Create a DataFrame
X_DATAFRAME = pd.DataFrame(dense_matrix)
X_DATAFRAME


from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

prepo = ColumnTransformer(
    transformers=[('standard', StandardScaler(with_mean=False), X_DATAFRAME.columns)], 
    remainder='passthrough'  # Optional for non-scaled features
)

X1 = pd.DataFrame(prepo.fit_transform(X_DATAFRAME), columns=X_DATAFRAME.columns)


X1 = sm.add_constant(X1)  # Add intercept term
logit_model = sm.Logit(y_train, X1).fit()





# ++++++++++++++++++++++++++++++++++++++++++++++++++# # import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # train and test 
 
# import pylab as pl
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


logit_model = sm.Logit(y_train, X1).fit()

# logit_model = sm.Logit(X_train, y_train).fit()

import pickle
pickle.dump(logit_model, open('logistic.pkl', 'wb'))

# Summary
logit_model.summary()

logit_model.summary2() # for AIC


# Prediction
pred = logit_model.predict(X_train)
pred  # Probabilities

# ROC Curve to identify the appropriate cutoff value
fpr, tpr, thresholds = roc_curve(y_train, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % auc)

# Filling all the cells with zeroes
X_train["pred"] = np.zeros(1340)

# taking threshold value and above the prob value will be treated as correct value 
X_train.loc[pred > optimal_threshold, "pred"] = 1


# Confusion Matrix
confusion_matrix(X_train.pred, y_train)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(X_train.pred, y_train))

# Classification report
classification = classification_report(X_train["pred"], y_train)
print(classification)

import matplotlib.pyplot as plt

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()
'for logistic model'
'on train data we got around 98% and on test data 95%'













------------------------------------

# X = all_emails_matrix
# Y_ = email_data["spam"]


# # import statsmodels.formula.api as smf
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split # train and test 
 
# # import pylab as pl
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import classification_report

# X_train = pd.DataFrame(X_train.toarray(), columns=emails_bow.get_feature_names_out())

# X_train = sm.add_constant(X_train)

# y_train = np.array(y_train).flatten()

# logit_model = sm.Logit(y_train, X_train).fit()

# logit_model = sm.Logit(X_train, y_train).fit()

# import pickle
# pickle.dump(logit_model, open('logistic.pkl', 'wb'))

# # Summary
# logit_model.summary()

# logit_model.summary2() # for AIC


# # Prediction
# pred = logit_model.predict(X_train)
# pred  # Probabilities

# # ROC Curve to identify the appropriate cutoff value
# fpr, tpr, thresholds = roc_curve(y_train, pred)
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx]
# optimal_threshold


# auc = metrics.auc(fpr, tpr)
# print("Area under the ROC curve : %f" % auc)

# # Filling all the cells with zeroes
# X_train["pred"] = np.zeros(1340)

# # taking threshold value and above the prob value will be treated as correct value 
# X_train.loc[pred > optimal_threshold, "pred"] = 1


# # Confusion Matrix
# confusion_matrix(X_train.pred, y_train)

# # Accuracy score of the model
# print('Test accuracy = ', accuracy_score(X_train.pred, y_train))

# # Classification report
# classification = classification_report(X_train["pred"], y_train)
# print(classification)

# import matplotlib.pyplot as plt

# ### PLOT FOR ROC
# plt.plot(fpr, tpr, label = "AUC="+str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc = 4)
# plt.show()
# 'for logistic model'
# 'on train data we got around 98% and on test data 95%'


