#importing the models
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('heart.csv')
#print(df.head())
# print(df.describe())
# print(df.info())
print(df['target'].value_counts())

# sns.countplot(x='target',data=df,palette="hls")
# plt.show()
x = df.iloc[:,:-1]    # all rows, all columns except the last
y = df.iloc[:, -1]     # all rows, only the last column
x=pd.DataFrame(x)
y=pd.DataFrame(y)

#splting the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

# Fit the model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Predicting
y_pred = logmodel.predict(x_test)
# print(y_pred)  #to print the prdict data

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
# print(cm)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(cm).plot()
plt.show()

probs=logmodel.predict_proba(x_test)

from sklearn.metrics import roc_curve, roc_auc_score

# Get probability scores for the positive class
probs = logmodel.predict_proba(x_test)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])

# Calculate AUC score
auc_score = roc_auc_score(y_test, probs[:, 1])

print("FPR:", fpr)


#draw roc curve
plt.figure()
plt.plot(fpr,tpr)
plt.show()
#area_under_curve
auc=roc_auc_score(y_test,y_pred)
print(auc)

#classification 

#check the acuuracy
print(logmodel.score(x_test,y_test))
