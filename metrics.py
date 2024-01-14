import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import antiVectorize
from scipy.stats import pearsonr

def calculate_jaccard_distance(gts, preds):
    jds = []
    for i in range(len(gts)):
        gt = gts[i]
        pred = preds[i]
        mins = 0
        maxs = 0

        for j in range(len(gt)):
            mins += min(gt[j], pred[j])
            maxs += max(gt[j], pred[j])

        jd = 1 - mins/maxs

        jds.append(jd)

    return np.array(jds).mean()

def calculate_pcc(gts, preds):
    pcs = []
    for i in range(len(gts)):
        gt = gts[i]
        pred = preds[i]
        pc, _ = pearsonr(gt,pred)
        pcs.append(pc)

    return np.array(pcs).mean()

def calculate_mae_ns(true, predicted, dim):
    ns_samples_true = []
    ns_samples_predicted = []
    for i in range(len(true)):
        real = true[i]
        fake = predicted[i]

        real_M = antiVectorize(real, dim)
        fake_M = antiVectorize(fake, dim)

        ns_real = np.sum(real_M, axis=1)
        ns_fake = np.sum(fake_M, axis=1)

        ns_samples_true.append(ns_real)
        ns_samples_predicted.append(ns_fake)

    ns_samples_true = np.array(ns_samples_true)
    ns_samples_predicted = np.array(ns_samples_predicted)

    return mean_absolute_error(ns_samples_true, ns_samples_predicted)
