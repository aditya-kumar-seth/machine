# knn algorithm implementation 
import numpy as np
from collections import Counter
def dis(x1,x2):
    distance = np.sqrt(np.sum(x2-x1)**2)
    return distance

class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,x,y):
        # we dont need to do much here in knn
        self.x_train = x 
        self.y_train = y 

    def predict(self,X):
        # find the distance of x from all 
        # the data points and find the k nearest ones 
        predictions = [ self.pred(x) for x in X]
        return predictions 
    def pred(self,x):
        # find the distance of x from all 
        distances = [dis(x,x_train) for x_train in self.x_train ]
        # the data points and find the k nearest ones
        indices = np.argsort(distances)[:self.k]#return indecies for first k neighbours
        k_nearest_labels = [self.y_train[i] for i in indices] 
        # find the most commmon class labels 
        majority_vote = Counter(k_nearest_labels).most_common()
        return majority_vote
