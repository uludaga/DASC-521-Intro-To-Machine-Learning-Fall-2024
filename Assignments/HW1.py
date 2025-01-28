import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-300))

X_train = np.genfromtxt("pendigits_sta16_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("pendigits_label_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("pendigits_sta16_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("pendigits_label_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    K = np.max(y_train)
    
    class_priors = [np.mean(y == (c + 1)) for c in range(K)]
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)

# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    # your implementation starts below
    K = np.max(y_train)
    
    P = np.array([np.mean(X[y == (c + 1)], axis=0) for c in range(K)])
    # your implementation ends above
    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)

# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    # your implementation starts below
    N = P.shape[0]

    # (x_i * log(pi) + (1-x_i) * log(1 - pi)) + log(class_priors)
    score_values = np.array([[np.sum(x * safelog(P[c]) + (1 - x) * safelog(1 - P[c])) 
                              + safelog(class_priors[c]) for c in range(N)] for x in X])
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)

# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = np.max(y_truth)
    
    y_pred = np.argmax(scores, axis=1) + 1
    
    # Confusion Matrix with shape (K, K)
    confusion_matrix = np.zeros((K, K), dtype=int)
    
    for true_label, pred_label in zip(y_truth, y_pred):
        confusion_matrix[pred_label - 1, true_label - 1] += 1
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))
confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))