import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("CC GENERAL.csv")

x=df.iloc[:,1:]
# print(x.head())
# sns.pairplot(data=x,palette=__annotations__)
# plt.show()
ss=StandardScaler()

x["CREDIT_LIMIT"]=x["CREDIT_LIMIT"].fillna(x["CREDIT_LIMIT"].mean())
x["MINIMUM_PAYMENTS"]=x["MINIMUM_PAYMENTS"].fillna(x["MINIMUM_PAYMENTS"].mean())
x_s=ss.fit_transform(x)

db=DBSCAN(eps=2,min_samples=5)
db.fit_predict(x_s)
x["spender"]=db.fit_predict(x_s)
label=db.labels_
print("unique labels",np.unique(label))

score=silhouette_score(x,label)
print("score",score)

# for eps in [0.5, 1, 1.5, 2, 3, 5,7,8,10,12,13,15,17,18,20,100,200,500,1000,2000,2500]:
#     db = DBSCAN(eps=eps, min_samples=5)
#     labels = db.fit_predict(x_s)
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print(f"eps={eps}, clusters={n_clusters}, unique={np.unique(labels)}")


# sns.pairplot(data=x)
# plt.show()

x_k=x.iloc[:,:-1]
y=x["spender"]

X=ss.fit_transform(x_k)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knc=DecisionTreeClassifier(max_depth=10)
knc.fit(x_train,y_train)

print("Test Score",knc.score(x_test,y_test))
print("Train score",knc.score(x_train,y_train))
print("Precision Score",precision_score(y_test,knc.predict(x_test),pos_label=-1))
print("Recall Score",recall_score(y_test,knc.predict(x_test),pos_label=-1))
print("F1 score",f1_score(y_test,knc.predict(x_test),pos_label=-1))

con=confusion_matrix(y_test,knc.predict(x_test))
sns.heatmap(data=con,annot=True)
plt.show()
