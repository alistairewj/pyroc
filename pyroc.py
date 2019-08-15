"""
The pyroc package provides tools for working with ROC curves and the area under the ROC.
"""

__author__ = "Alistair Johnson <aewj@mit.edu>"
__version__ = "0.1.0"

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


class ROC(object):
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
            self.preds = OrderedDict([[i, x] for i, x in enumerate(preds)])
        elif 'array' in str(type(preds)):
            # convert preds into a dictionary
            self.predictors = [0]
            self.preds = OrderedDict([[0, preds]])
        elif type(preds) is dict:
            # preds is a dict - convert to ordered
            self.predictors = list(preds.keys())
            self.predictors.sort()
            self.preds = OrderedDict([[c, preds[c]] for c in self.predictors])
        elif type(preds) is not OrderedDict:
            raise ValueError(
                'Unrecognized type "%s" for predictions.', str(type(preds)))
        else:
            # is already a ordered dict
            self.preds = preds
            self.predictors = list(preds.keys())

        # TODO: validate inputs
        self.target = target

        # initialize vars
        self.K = len(self.preds)
        self.n_obs = len(target)
        self.n_pos = np.sum(target == 1)
        self.n_neg = self.n_obs - self.n_pos

        # First parse the predictions into matrices X and Y
        #   X: predictions for target = 1
        #   Y: predictions for target = 0
        # Each prediction will be stored in a column of X and Y

        # create "X" - the predictions when target == 1
        idx = self.target == 1
        self.X = np.zeros([self.n_pos, self.K])
        for i, p in enumerate(self.preds.keys()):
            self.X[:, i] = self.preds[p][idx]

        # create "Y" - the predictions when target == 0
        idx = ~idx
        self.Y = np.zeros([self.n_neg, self.K])
        for i, p in enumerate(self.preds.keys()):
            self.Y[:, i] = self.preds[p][idx]

        # calculate auc, V10, V01
        self.__calculate_auc()

        # calculate S01, S10, S
        self.__calculate_covariance()

    def __calculate_auc(self):
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
                theta[0, r] = theta[0, r]+phi1+phi2*0.5

            theta[0, r] = theta[0, r]/(n*m)
            for j in range(n):
                # correct classifications (X>Y)
                phi1 = np.sum(self.X[:, r] > self.Y[j, r])
                # ties (X==Y) get half points
                phi2 = np.sum(self.X[:, r] == self.Y[j, r])
                V01[j, r] = (phi1+phi2*0.5)/m

        self.auc = theta
        self.V10 = V10
        self.V01 = V01

        return self.auc

    def __calculate_covariance(self):
        # Calculates the covariance for R sets of predictions and outcomes

        m = self.V10.shape[0]
        n = self.V01.shape[0]

        if self.auc is None:
            self.__calculate_auc()

        V01, V10, theta = self.V01, self.V10, self.auc

        # Calculate S01 and S10, covariance matrices of V01 and V10
        self.S01 = ((np.transpose(V01)@V01) -
                    n*(np.transpose(theta)@theta))/(n-1)
        self.S10 = ((np.transpose(V10)@V10) -
                    m*(np.transpose(theta)@theta))/(m-1)
        # Alternative equivalent formulations:
        # self.S01 = (np.transpose(V01-theta)@(V01-theta))/(n-1)
        # self.S10 = (np.transpose(V10-theta)@(V10-theta))/(m-1)
        # self.S01 = np.cov(np.transpose(self.V01))
        # self.S10 = np.cov(np.transpose(self.V10))

        # Combine for S, covariance matrix of AUCs
        self.S = (1/m)*self.S10 + (1/n)*self.S01

    def ci(self, alpha=0.05):
        # Calculates the confidence intervals for each auroc separetely
        if self.auc is None:
            self.__calculate_auc()

        # Calculate CIs
        itvs = np.transpose([[alpha/2, 1-(alpha/2)]])
        ci = norm.ppf(itvs, self.auc, np.sqrt(np.diagonal(self.S)))

        return ci

    def compare(self, contrast, alpha=0.05):
        # Compare predictions given a contrast
        # If there are two predictions, you can compare as:
        #   contrast = [1, -1]

        # Validate alpha
        if (alpha <= 0) | (alpha >= 1):
            raise ValueError('alpha must be in the range (0, 1), exclusive.')
        elif alpha > 0.5:
            alpha = 1 - alpha

        # Verify if covariance was calculated
        if self.S is None:
            self.__calculate_covariance()

        # L as matrix
        L = np.array(contrast, dtype=float)
        if len(L.shape) == 1:
            L = L.reshape(1, L.shape[0])

        # Shapes
        L_sz = L.shape
        S_sz = self.S.shape

        # is not equal to number of classifiers
        if (S_sz[1] != L_sz[1]):  # Contrast column
            raise ValueError(
                'Contrast should have %d elements (number of predictors)',
                S_sz[1]
            )

        # Validate contrast
        if np.any(np.sum(L, axis=1) != 0):
            raise ValueError('Contrast rows must sum to 0', S_sz[1])

        # Calculate LSL matrix
        LSL = L @ self.S @ np.transpose(L)

        # Normal vs chi^2 distribution
        if L_sz[0] == 1:
            # Compute using the normal distribution
            mu = L @ np.transpose(self.auc)
            sigma = np.sqrt(LSL)
            thetaP = norm.cdf(0, mu, sigma)

            # 2-sided test, double the tails -> double the p-value
            if mu < 0:
                thetaP = 2*(1-thetaP)
            else:
                thetaP = 2*thetaP

            # Confidence intervals
            theta2 = norm.ppf([alpha/2, 1-alpha/2], self.auc[L == 1], sigma)
        else:
            # Calculate chi2 stat with DOF = rank(L*S*L')
            # first invert the LSL matrix
            inv_LSL = np.linalg.inv(LSL)

            # then calculate the chi2
            w_chi2 = np.transpose(self.auc) @ np.transpose(
                L) @ inv_LSL @ L @ self.auc
            w_df = np.linalg.matrix_rank(LSL)
            thetaP = 1 - chi2.cdf(w_chi2, w_df)
            theta2 = w_chi2

        return thetaP, theta2

    def __figure(self):
        # Create figure
        fig, ax = plt.subplots(figsize=(36, 30))
        fig.tight_layout()

        # Stylying
        ax.tick_params(labelsize=60)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

        # Set axes limits
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([-0.02, 1.05])

        # Title and labels
        ax.set_title('ROC Curve', fontsize=60)
        ax.set_xlabel('False Positive Rate', fontsize=60)
        ax.set_ylabel('True Positive Rate', fontsize=60)

        # Adjust figure border
        plt.gcf().subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.98)

        return (fig, ax)

    def __roc(self, pred):
        # Transform to matrices
        y_prob = np.array([pred])
        target = np.array([self.target])

        # Calculate predictions for all thresholds
        thresholds = np.transpose([np.unique(y_prob)])
        y_pred = np.greater_equal(y_prob, thresholds)

        # FPR and TPR
        P = target == 1
        N = target == 0
        FP = np.logical_and(y_pred == 1, N)
        TP = np.logical_and(y_pred == 1, P)
        fpr = np.sum(FP, axis=1)/np.sum(N)
        tpr = np.sum(TP, axis=1)/np.sum(P)

        return fpr, tpr

    def curve(self, labels=None):
        # Create figure
        fig, ax = self.__figure()

        # Calculate auc
        if self.auc is None:
            self.__calculate_auc()

        # Get colormap
        viridis = plt.cm.get_cmap("viridis", len(labels))

        # Calculate confidence intervals
        ci = self.ci()

        # Set default labels
        if labels is None:
            labels = {i: "ROC{0}".format(i) for i in range(self.K)}

        for i, label in labels.items():
            # Get prediction for current iteration
            pred = self.preds[i]

            # Calculate FPRs and TPRs
            fpr, tpr = self.__roc(pred)
            roc = ax.plot(fpr, tpr, lw=12, color=viridis(i))[0]

            # Line legend
            legend = '{0}, AUC = {1:0.2f} ({2:0.2f}-{3:0.2f})'.format(
                label, self.auc[0, i], ci[0, i], ci[1, i])
            roc.set_label(legend)

        # Legend stylying
        ax.legend(fontsize=50, loc=4)

        return (fig, ax)
