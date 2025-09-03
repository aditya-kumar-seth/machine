from knn import KNN
import numpy as np
from sklearn.datasets import load_iris
from preprocessing import train_test_split as tts
from preprocessing import cross_val_score as cvs
x,y = load_iris().data,load_iris().target
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2)
model = KNN()
cv_score = cvs(model,x_train,y_train,k=3)
print(cv_score)