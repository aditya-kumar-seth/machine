import numpy as np
from sklearn.datasets import load_iris#data set from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cvs2
# my libraries
from knn import KNN
from preprocessing import train_test_split as tts
from preprocessing import cross_val_score as cvs


x,y = load_iris().data,load_iris().target
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2)

model = KNN()
model2 = KNeighborsClassifier()

cv_score = cvs(model,x_train,y_train)
cv_score_sklearn = cvs2(model2,x_train,y_train)
print(f"my models accuracy {cv_score}")
print(f"sklearn accuracy {np.mean(cv_score_sklearn)} ")