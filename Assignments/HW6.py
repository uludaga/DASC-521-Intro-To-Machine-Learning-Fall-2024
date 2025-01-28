import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("hw06_predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("hw06_predicted_probabilities2.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    # your implementation starts below
    sorted_probs = np.sort(predicted_probabilities)
    first_threshold = sorted_probs[0] / 2
    last_threshold = sorted_probs[-1] + (1 - sorted_probs[-1]) / 2
    mid_vals = []
    
    for i in range(len(sorted_probs) - 1):
        mid_val = (sorted_probs[i] + sorted_probs[i + 1]) / 2
        mid_vals.append(mid_val)
    
    thresholds = np.array([first_threshold] + mid_vals + [last_threshold])
    # your implementation ends above
    return thresholds

thresholds1 = calculate_threholds(predicted_probabilities1)
print(thresholds1)

thresholds2 = calculate_threholds(predicted_probabilities2)
print(thresholds2)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    # your implementation starts below
    true_labels = np.array(true_labels)
    predicted_probabilities = np.array(predicted_probabilities)
    fp_rates = []
    tp_rates = []
    P = np.sum(true_labels == 1)
    N = np.sum(true_labels == -1)
    
    for threshold in thresholds:
        predicted_labels = np.where(predicted_probabilities >= threshold, 1, -1)
        
        TP = np.sum((predicted_labels == 1) & (true_labels == 1))
        FP = np.sum((predicted_labels == 1) & (true_labels == -1))
        fp_rate = FP / N
        tp_rate = TP / P
        fp_rates.append(fp_rate)
        tp_rates.append(tp_rate)
    # your implementation ends above
    return fp_rates, tp_rates

fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    # your implementation starts below
    fp_rates = np.array(fp_rates)
    tp_rates = np.array(tp_rates)
    sorted_indices = np.argsort(fp_rates)
    fp_rates = fp_rates[sorted_indices]
    tp_rates = tp_rates[sorted_indices]
    auroc = 0.0
    
    for i in range(len(fp_rates) - 1):
        x = fp_rates[i + 1] - fp_rates[i]
        y = (tp_rates[i] + tp_rates[i + 1]) / 2
        auroc += x * y
    # your implementation ends above
    return auroc

auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))