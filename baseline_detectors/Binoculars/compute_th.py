from sklearn.metrics import roc_curve, f1_score
import numpy as np


def compute(pred, scores):

    fpr, tpr, thresholds = roc_curve(pred, scores)

    # F1-Score
    f1_scores = [f1_score(pred, [1 if y >= thr else 0 for y in scores]) for thr in thresholds]
    #print(f1_scores)
    #print(thresholds)
    optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
    return optimal_threshold_f1
