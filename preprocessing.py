# this is for train test split
import numpy as np
from knn import KNN

def train_test_split(x, y, test_size = 0.25, random_state = 42):
    np.random.seed(random_state)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    split_idx = int(test_size*len(x))
    train_idx = indices[split_idx:]
    test_idx = indices[:split_idx]
    x_train, y_train = x[train_idx],y[train_idx]
    x_test, y_test = x[test_idx],y[test_idx]
    return x_train, x_test, y_train, y_test   


def cross_val_score(model, x_train, y_train, k = 5):
    fold_size = int(len(x_train)/k)
    start=0
    acc = []
    for i in range(k):
        start = i * fold_size
        # FIX: last fold includes all remaining
        end = (i + 1) * fold_size if i != k - 1 else len(x_train)
        x_train = np.concatenate( [x_train[:start], x_train[start:]] )
        x_val=x_train[start:end]
        y_train =  np.concatenate( [y_train[:start], y_train[start:]] )
        y_val = y_train[start:end]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_val)
        acc.append(model.accuracy(y_val,y_pred))
    return np.mean(acc)
