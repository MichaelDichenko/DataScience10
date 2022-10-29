#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import scipy.optimize as opt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics


# In[2]:


#notice: Disable all warnings 
import warnings
warnings.filterwarnings('ignore')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[3]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[4]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[5]:


df.shape


# ### Convert to date time object
# 

# In[6]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[7]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[8]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('pip install seaborn')


# In[9]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[10]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[11]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[12]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[13]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[14]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[15]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[16]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[17]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[18]:


X = Feature
X[0:5]


# In[97]:


df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
df.head()


# What are our lables?
# 

# In[98]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[99]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[100]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[101]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[102]:


n = 25
accuracy = np.zeros(n)
for i in range(1,n+1):
    clf = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    accuracy[i-1] = (accuracy_score(y_test, y_test_predicted))
accuracy


# In[103]:


plt.plot(range(1,n+1),accuracy,'g')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.show()

accuracy = pd.DataFrame(accuracy)
print("Maximum Accuracy Got is - " )
accuracy.sort_values(by = 0, ascending = False)[0:1]


# In[104]:


clf_KNN = KNeighborsClassifier(n_neighbors = 24).fit(X_train, y_train)
# y_test_pred_KNN = clf_KNN.predict(X_test)
yhatKNN = clf_KNN.predict(X_test)
yhatKNN[0:5]


# # Decision Tree
# 

# In[105]:


from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(criterion = 'gini').fit(X_train, y_train)
y_test_pred_KNN = clf.predict(X_test)
print("Accuracy using criterion as gini - ", accuracy_score(y_test, y_test_pred_KNN))
clf3 = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)
y_test_pred_KNN = clf2.predict(X_test)
print("Accuracy using criterion as entropy - ", accuracy_score(y_test, y_test_pred_KNN))


# In[106]:


# using criterion as gini
clf_DT = DecisionTreeClassifier(criterion = 'gini').fit(X_train, y_train)
yhatDT = clf_DT.predict(X_test)
yhatDT[0:5]


# # Support Vector Machine
# 

# In[107]:


from sklearn.svm import SVC


# In[108]:


clf4 = SVC(kernel = 'poly').fit(X_train, y_train)
print("accuracy using polynomial kernel - ", accuracy_score(y_test, clf2.predict(X_test)))
clf5 = SVC(kernel = 'rbf').fit(X_train, y_train)
print("accuracy using Radial Basis function kernel - ", accuracy_score(y_test, clf3.predict(X_test)))


# In[109]:


# using linear for kernel in our SVM model
clf_SVM = SVC(kernel = 'poly', random_state = 4).fit(X_train, y_train)
yhatSVM = clf_SVM.predict(X_test)
yhatSVM[0:5]


# In[110]:


from sklearn.metrics import classification_report, confusion_matrix


# In[111]:


print("Avg F1-score: %.4f" % f1_score(y_test, yhatSVM, average='weighted'))


# # Logistic Regression
# 

# In[112]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
clf_LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
clf_LR


# In[113]:


yhatLR = clf_LR.predict(X_test)
yhatLR


# In[114]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[115]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# # Model Evaluation using Test set
# 

# In[116]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# In[117]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[118]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[119]:


# conversion to datetime object
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df.head()


# In[120]:


# creating weekend column
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()


# In[121]:


# replacing male and female string with 0 and 1
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()


# In[122]:


# creaing dummies and dropping one column

X_t = test_df[['Principal','terms','age','Gender','weekend']]
X_t = pd.concat([X_t,pd.get_dummies(test_df['education'])], axis=1)
X_t.drop(['Master or Above'], axis = 1,inplace=True)
X_t.head()


# In[123]:


# Feature Scaling
X_t = preprocessing.StandardScaler().fit(X_t).transform(X_t)
X_t[0:5]


# In[124]:


test_df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)


# In[125]:


y_t = test_df['loan_status'].values
y_t[0:20]


# In[126]:


# np arrays to store intermediate result
Jaccard = np.full(4, np.nan)
F1_score = np.full(4, np.nan)
LogLoss = np.full(4, np.nan)
Algorithm = np.array(4)
Algorithm = ["KNN", "Decision Tree", "SVM", "LogisticRegression"]


# In[131]:


Jaccard[0] = jaccard_score(y_t, clf_KNN.predict(X_t))
Jaccard[1] = jaccard_score(y_t, clf_DT.predict(X_t))
Jaccard[2] = jaccard_score(y_t, clf_SVM.predict(X_t))
Jaccard[3] = jaccard_score(y_t, clf_LR.predict(X_t))


# In[132]:


F1_score[0] = f1_score(y_t, clf_KNN.predict(X_t))
F1_score[1] = f1_score(y_t, clf_DT.predict(X_t))
F1_score[2] = f1_score(y_t, clf_SVM.predict(X_t))
F1_score[3] = f1_score(y_t, clf_LR.predict(X_t))


# In[133]:


LogLoss[3] = log_loss(y_t, clf_LR.predict_proba(X_t))


# In[134]:


Report = pd.DataFrame({"Jaccard":Jaccard, "F1-score":F1_score, "LogLoss":LogLoss}, index=Algorithm)


# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# In[136]:


Report


# | Algorithm          | Jaccard | F1-score | LogLoss |
# | ------------------ | ------- | -------- | ------- |
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 
