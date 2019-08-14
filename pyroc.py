"""
The pyroc package provides tools for working with ROC curves and the area under the ROC.
"""

__author__ = "Alistair Johnson <aewj@mit.edu>"
__version__ = "0.1.0"

from collections import OrderedDict

import numpy as np
from scipy.stats import norm, chi2

def auroc(target, pred):
    """
    Calculates the area under the receiver operator characteristic curve (AUROC).
    Equivalent to the Wilcoxon statistic W.

    Inputs:
        target - the target variable, 0 or 1, predicted by pred
        pred   - the prediction made (the probability that target=1).

    Outputs:
        W - Probability( PRED|target=1 > PRED|target=0 )
               Calculation: sum(sum(PRED|target=1 > PRED|target=0))
               Equivalent to the AUROC.
    """

    idx = np.argsort(pred)
    pred = pred[idx]
    target = target[idx]

    N, P = pred.shape

    W = np.zeros(1, P)  # 1xP where P is # of AUROCs to calculate
    negative = np.zeros(N, P, dtype=bool)

    for p in range(P):
        negative[:, p] = target[:, p] == 0

        # Count the number of negative targets below each element
        negativeCS = np.cumsum(negative[:, p], 1)

        # Only get positive targets
        pos = negativeCS[~negative[:, p]]

        W[n] = np.sum(pos)

    count = np.sum(negative, axis=0)  # count number who are negative
    count = count * (N - count)  # multiply by positives
    W = W/count

    return W


def auroc_ci(target, pred, alpha=0.05):
    """
    Calculates confidence intervals around an AUROC.

    Inputs:
        target - the target variable, 0 or 1
        pred - the prediction made (the probability that target == 1)
        alpha - Critical value for confidence intervals (default 0.05)

    Outputs:
        auroc - AUROC/Wilcoxon statistic
        auroc_ci - Confidence interval (length 2 list)

    References:
        Sen 1960
        DeLong et al, Biometrics, September 1988
    
    Copyright 2019 Alistair Johnson
    """
    # First parse the inputs into matrices X and Y
    #   X: predictions for target = 1
    #   Y: predictions for target = 0
    # Each prediction will be stored in a column of X and Y

    # Parse first two inputs to get sizes(m, n) and vectors(X, Y)
    K = 1
    # Only one prediction/target pair
    idx = target == 1
    X = pred[idx]
    m = len(X)
    Y = pred[~idx]
    n = len(Y)

    # # # -------------- AUC, V10, V01 - ------------- # # #
    # Using matrices X and Y, calculate estimated Wilcoxon statistic(auc)
    # Also Calculate the mxK and nxK V10 and V01 matrices

    auc = np.zeros([1, K])
    V10 = np.zeros([m, K])
    V01 = np.zeros([n, K])

    for r in range(K):  # For each X/Y column pair
        # compare 0s to 1s
        for i in range(m):
            phi1 = np.sum(X[i, r] > Y[:, r])
            # Xi > Y
            phi2 = np.sum(X[i, r] == Y[:, r])
            # Xi = Y
            V10[i, r] = (phi1+phi2*0.5)/n
            auc[r] = auc[r]+phi1+phi2*0.5

        auc[r] = auc[r]/(n*m)
        for j in range(n):
            # correct classifications (X>Y)
            phi1 = X[:, r] > Y[j, r]
            # ties (X==Y) get half points
            phi2 = X[:, r] == Y[j, r]
            V01[j, r] = (sum(phi1)+sum(phi2)*0.5)/m

    #  Calculate S01 and S10, covariance matrices of V01 and V10
    S01 = ((np.transpose(V01)*V01)-n*(np.transpose(auc) * auc))/(n-1)
    S10 = ((np.transpose(V10)*V10)-m*(np.transpose(auc) * auc))/(m-1)
    # Alternative equivalent formulation:
    # S01a = (np.transpose(V01-auc)*(V01-auc))/(n-1)
    # S10a = (np.transpose(V10-auc)*(V10-auc))/(m-1)

    # Combine for S, covariance matrix of auc
    S = (1/m)*S10 + (1/n)*S01
    auc_ci = norm.ppf([alpha/2, 1-(alpha/2)], auc, np.sqrt(S))

    return auc, auc_ci

class AUROC(object):
    """
    Use this class if you are interested in statistically comparing AUROCs.

    Initialize with:
        target - a vector of 0s and 1s
        preds - a vector of predictions for the target
          *or* a list of these vectors if comparing multiple predictions
          *or* a dictionary of these vectors
    """

    def __init__(self, target, preds):
        if type(preds) is list:
            # convert preds into a dictionary
            self.predictors = [x for x in range(len(preds))]
            self.preds = OrderedDict([
                [i, x] for i, x in enumerate(preds)
            ])
        elif 'array' in str(type(preds)):
            self.predictors = [0]
            self.preds = OrderedDict([
                [0, preds]
            ])
        elif type(preds) is dict:
            # preds is a dict - convert to ordered
            self.predictors = list(preds.keys())
            self.predictors.sort()

            self.preds = OrderedDict([
                [c, preds[c]] for c in self.predictors
            ])
        elif type(preds) is not OrderedDict:
            raise ValueError('Unrecognized type "%s" for predictions.',
                            str(type(preds)))
        else:
            self.preds = preds
            self.predictors = list(preds.keys())

        

        # TODO: validate inputs
        self.target = target

        # initialize vars
        self.K = len(preds)
        self.n_obs = len(target)
        self.n_pos = np.sum(target == 1)
        self.n_neg = self.n_obs - self.n_pos

        # First parse the predictions into matrices X and Y
        #   X: predictions for target = 1
        #   Y: predictions for target = 0
        # Each prediction will be stored in a column of X and Y

        # create "X" - the predictions when target == 1
        idx = self.target == 1
        self.X = np.zeros([self.n_pos, len(preds)])
        for i, p in enumerate(self.preds.keys()):
            self.X[:, i] = self.preds[p][idx]

        # create "Y" - the predictions when target == 0
        idx = ~idx
        self.Y = np.zeros([self.n_neg, len(preds)])
        for i, p in enumerate(self.preds.keys()):
            self.Y[:, i] = self.preds[p][idx]

        # calculate auroc, V10, V01
        self.calculate_auroc()

        # calculate S01, S10, S
        self.calculate_covariance()

    def calculate_auroc(self):
        m = self.X.shape[0]
        n = self.Y.shape[0]

        theta = np.zeros([1, self.K])
        V10 = np.zeros([m, self.K])
        V01 = np.zeros([n, self.K])

        for r in range(self.K):  # For each X/Y column pair
            # compare 0s to 1s
            for i in range(m):
                phi1 = np.sum(self.X[i, r] > self.Y[:, r])
                # Xi > Y
                phi2 = np.sum(self.X[i, r] == self.Y[:, r])
                # Xi = Y
                V10[i, r] = (phi1+phi2*0.5)/n
                theta[r] = theta[r]+phi1+phi2*0.5

            theta[r] = theta[r]/(n*m)
            for j in range(n):
                # correct classifications (X>Y)
                phi1 = self.X[:, r] > self.Y[j, r]
                # ties (X==Y) get half points
                phi2 = self.X[:, r] == self.Y[j, r]
                V01[j, r] = (np.sum(phi1)+np.sum(phi2)*0.5)/m

        self.auroc = theta
        self.V10 = V10
        self.V01 = V01

    def calculate_covariance(self):
        # Calculates the covariance for R sets of predictions and outcomes

        m = self.X.shape[0]
        n = self.Y.shape[0]

        if self.auroc is None:
            self.calculate_auroc()

        V01, V10, theta = self.V01, self.V10, self.auroc

        #  Calculate S01 and S10, covariance matrices of V01 and V10
        self.S01 = ((np.transpose(V01)*V01)-n *
                    (np.transpose(theta) * theta))/(n-1)
        self.S10 = ((np.transpose(V10)*V10)-m*(np.transpose(theta) * theta))/(m-1)
        # Alternative equivalent formulation:
        # S01a = (np.transpose(V01-theta)*(V01-theta))/(n-1)
        # S10a = (np.transpose(V10-theta)*(V10-theta))/(m-1)

        # Combine for S, covariance matrix of AUROCs
        self.S = (1/m)*self.S10 + (1/n)*self.S01

    def compare(self, contrast, alpha=0.05):
        # Compare predictions given a contrast
        # If there are two predictions, you can compare as:
        #   contrast = [1, -1]

        if (alpha <= 0) | (alpha >= 1): 
            raise ValueError('alpha must be in the range (0, 1), exclusive.')
        
        if alpha > 0.5:
            alpha = 1 - alpha

        if self.S is None:
            self.calculate_covariance()
        
        S = self.S
        L = np.array(contrast, dtype=float)
        L_sz = L.shape
        S_sz = S.shape

        # is not equal to number of classifiers
        if S_sz[1] != L_sz[1]: # Contrast column
            raise ValueError(
                'Contrast should have %d elements (number of predictors)', S_sz[1]
            )
        
        if np.sum(L) != 0:
            raise ValueError(
                'Contrast must sum to 0', S_sz[1]
            )
            
        LSL = L*S*np.transpose(L)
        if L_sz[0] == 1:
            # One row
            # Compute using the normal distribution
            mu = L*self.auroc
            sigma = np.sqrt(LSL)
            # self.auroc1 = normpdf(0, mu, sigma)
            thetaP = norm.cdf(0, mu, sigma)
            # 2-sided test, double the tails -> double the p-value
            if mu < 0:
                thetaP = 2*(1-thetaP)
            else:
                thetaP = 2*thetaP
            theta2 = norm.ppf([alpha/2, 1-alpha/2], self.auroc[L == 1], sigma)
        else:
            # Calculate chi2 stat with DOF = rank(L*S*L')

            # first invert the LSL matrix
            inv_LSL = np.linalg.inv(LSL)

            # then calculate the chi2
            w_chi2 = np.transpose(self.auroc)*np.transpose(L)*inv_LSL*L*self.auroc
            w_df = np.linalg.matrix_rank(LSL)
            thetaP = 1 - chi2.cdf(w_chi2, w_df)
            theta2 = w_chi2

        return thetaP, theta2
