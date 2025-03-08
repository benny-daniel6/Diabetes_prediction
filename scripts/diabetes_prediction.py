import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('diabetes.csv')
print(df.head(2))
# df.hist(figsize=(12,10))
# plt.suptitle('Feature Distribution')
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

smote=SMOTE(random_state=42)
scaler=StandardScaler()
x=df.drop('Outcome',axis=1)
y=df['Outcome']
x_scaled=scaler.fit_transform(x)
x_resampled,y_resampled=smote.fit_resample(x_scaled,y)
xtrain,xtest,ytrain,ytest=train_test_split(x_resampled,y_resampled,test_size=0.3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)
# model.fit(xtrain,ytrain)
# ypred=model.predict(xtest)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("ACCURACY SCORE : ",accuracy_score(ytest,ypred))
print("CLASSIFICATION REPORT : \n",classification_report(ytest,ypred))
print("CONFUSION MATRIX : \n",confusion_matrix(ytest,ypred))
import joblib
joblib.dump(rf,'diabetes_model.pkl')
joblib.dump(scaler,'scaler.pkl')

