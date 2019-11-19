"""
Tools for working with ROC curves and the area under the ROC.
"""

__author__ = "Alistair Johnson <aewj@mit.edu>, Lucas Bulgarelli"
__version__ = "0.1.0"

from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


class ROC(object):
    """Class for calculating receiver operator characteristic curves (ROCs).

    Also facilitates statistically comparing the area under the ROC for
    multiple predictors.

    Attributes
    ----------
    K : int
        Calculated number of individual predictors.
    n_obs : int
        Calculated number of observations
    n_pos : int
        Number of observations with a positive class (== 1)
    n_neg : int
        Number of observations with a negative class (== 0)
    X : np.ndarray
        Numpy array of predictions for observations in the positive class
    Y : np.ndarray
        Numpy array of predictions for observations in the negative class
    """
    def __init__(self, target, preds):
        """ROC class for comparing predictors for a common set of targets.

        Parameters
        ----------
        target : np.ndarray
            A length N vector of binary values (0s or 1s).
        preds
            A vector of predictions which correspond to the targets
            *or* a list of vectors,
            *or* an NxD matrix,
            *or* a dictionary of vectors.
        """
        self.preds, self.target = self._parse_inputs(preds, target)

        # TODO: validate that target/predictors work
        # self._validate_inputs()

        # initialize vars
        self.K = len(self.preds)
        self.n_obs = len(self.target)
        self.n_pos = np.sum(self.target == 1)
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
        self._calculate_auc()

        # calculate S01, S10, S
        self._calculate_covariance()

    def _parse_inputs(self, preds, target):
        """Parse various formats of preds into dictionary/numpy array.

        Parameters
        ----------
        preds
            A vector of predictions which correspond to the targets
            *or* a list of vectors,
            *or* an NxD matrix,
            *or* a dictionary of vectors.

        Returns
        -------
        (OrderedDict, np.ndarray)
            - An ordered dictionary with the values for each predictor.
              If no predictor names are provided, then predictor names are
              monotonically increasing integers.
            - A list of names for each predictor and a dictionary with
              the values for each predictor. If no predictor names are
              provided, then predictor names are monotonically increasing
              integers.

        """
        if type(preds) is OrderedDict:
            # is already a ordered dict
            pass
        elif type(preds) is list:
            if hasattr(preds[0], '__len__'):
                # convert preds into a dictionary
                preds = OrderedDict(
                    [[i, np.asarray(p)] for i, p in enumerate(preds)]
                )
            elif type(preds[0]) in (float, int):
                preds = OrderedDict([(0, np.asarray(preds))])
            else:
                raise TypeError(
                    'unable to parse preds list with element type %s',
                    type(preds[0])
                )
        elif type(preds) is pd.DataFrame:
            # preds is a dict - convert to ordered
            preds = OrderedDict(zip(preds.columns, preds.T.values))
        elif type(preds) is np.ndarray:
            if len(preds.shape) <= 1:
                # numpy vector
                preds = OrderedDict([[0, np.asarray(preds)]])
            else:
                # numpy matrix
                preds = OrderedDict(
                    [[i, preds[:, i]] for i in range(preds.shape[1])]
                )
        elif type(preds) is dict:
            # preds is a dict - convert to ordered
            names = sorted(preds.keys())
            preds = OrderedDict([[c, np.asarray(preds[c])] for c in names])
        else:
            raise ValueError(
                'Unrecognized type "%s" for predictions.', str(type(preds))
            )

        if type(target) is pd.Series:
            target = target.values
        elif type(target) in (list, tuple):
            target = np.asarray(target)
        elif type(target) is not np.ndarray:
            raise TypeError(
                'target should be type np.ndarray, was %s', type(target)
            )

        return preds, target

    def _calculate_auc(self):
        """Calculates the area under the ROC and the variances of each predictor.
        """
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
                V10[i, r] = (phi1 + phi2 * 0.5) / n
                theta[0, r] = theta[0, r] + phi1 + phi2 * 0.5

            theta[0, r] = theta[0, r] / (n * m)
            for j in range(n):
                # correct classifications (X>Y)
                phi1 = np.sum(self.X[:, r] > self.Y[j, r])
                # ties (X==Y) get half points
                phi2 = np.sum(self.X[:, r] == self.Y[j, r])
                V01[j, r] = (phi1 + phi2 * 0.5) / m

        self.auc = theta
        self.V10 = V10
        self.V01 = V01

        return self.auc

    def _calculate_covariance(self):
        """Calculate the covariance for K sets of predictions and outcomes
        """

        m = self.V10.shape[0]
        n = self.V01.shape[0]

        if self.auc is None:
            self._calculate_auc()

        V01, V10, theta = self.V01, self.V10, self.auc

        # Calculate S01 and S10, covariance matrices of V01 and V10
        self.S01 = (
            (np.transpose(V01) @ V01) - n * (np.transpose(theta) @ theta)
        ) / (n - 1)
        self.S10 = (
            (np.transpose(V10) @ V10) - m * (np.transpose(theta) @ theta)
        ) / (m - 1)
        # Alternative equivalent formulations:
        # self.S01 = (np.transpose(V01-theta)@(V01-theta))/(n-1)
        # self.S10 = (np.transpose(V10-theta)@(V10-theta))/(m-1)
        # self.S01 = np.cov(np.transpose(self.V01))
        # self.S10 = np.cov(np.transpose(self.V10))

        # Combine for S, covariance matrix of AUCs
        self.S = (1 / m) * self.S10 + (1 / n) * self.S01

    def ci(self, alpha=0.05):
        """Calculates the confidence intervals for each auroc separetely.
        """
        if self.auc is None:
            self._calculate_auc()

        # Calculate CIs
        itvs = np.transpose([[alpha / 2, 1 - (alpha / 2)]])
        ci = norm.ppf(itvs, self.auc, np.sqrt(np.diagonal(self.S)))

        return ci

    def compare(self, contrast, alpha=0.05):
        """Compare predictions given a contrast

        If there are two predictions, you can compare as:
            roc.compare(contrast=[1, -1], alpha=0.05)
        """

        # Validate alpha
        if (alpha <= 0) | (alpha >= 1):
            raise ValueError('alpha must be in the range (0, 1), exclusive.')
        elif alpha > 0.5:
            alpha = 1 - alpha

        # Verify if covariance was calculated
        if self.S is None:
            self._calculate_covariance()

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
                thetaP = 2 * (1 - thetaP)
            else:
                thetaP = 2 * thetaP

            # Confidence intervals
            theta2 = norm.ppf([alpha / 2, 1 - alpha / 2], mu, sigma)
        else:
            # Calculate chi2 stat with DOF = rank(L*S*L')
            # first invert the LSL matrix
            inv_LSL = np.linalg.inv(LSL)

            # then calculate the chi2
            w_chi2 = self.auc @ np.transpose(L) @ inv_LSL @ L @ np.transpose(
                self.auc
            )
            w_df = np.linalg.matrix_rank(np.transpose(LSL))
            thetaP = 1 - chi2.cdf(w_chi2, w_df)
            theta2 = chi2.ppf([alpha / 2, 1 - alpha / 2], w_df)

        return np.ndarray.item(thetaP), theta2

    def _roc(self, pred):
        """Calculate false positive rate and true positive rate for ROC curve.

        Returns
        -------
        (fpr, tpr)
            np.ndarrays containing the false positive rate and the true
            positive rate, respectively.
        
        """
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
        fpr = np.sum(FP, axis=1) / np.sum(N)
        tpr = np.sum(TP, axis=1) / np.sum(P)

        return fpr, tpr

    def __figure(self, figsize=(36, 30), **kwargs):
        """Initialize a figure for plotting the ROC curve.
        """
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
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

    def plot(self, labels=None, fontsize=50, **kwargs):
        """Plot the ROC curve.
        """

        # Init figure with axes labels, etc.
        fig, ax = self.__figure(**kwargs)

        # Calculate auc
        if self.auc is None:
            self._calculate_auc()

        # Calculate confidence intervals
        ci = self.ci()

        # Set default labels
        if labels is None:
            labels = self.preds.keys()

        # Get colormap
        viridis = plt.cm.get_cmap("viridis", len(labels))

        for i, label in enumerate(labels):
            # Get prediction for current iteration
            pred = self.preds[label]

            # Calculate FPRs and TPRs
            fpr, tpr = self._roc(pred)
            roc = ax.plot(fpr, tpr, lw=12, color=viridis(i))[0]

            # Line legend
            legend = '{0}, AUC = {1:0.2f} ({2:0.2f}-{3:0.2f})'.format(
                label, self.auc[0, i], ci[0, i], ci[1, i]
            )
            roc.set_label(legend)

        # Legend stylying
        ax.legend(fontsize=fontsize, loc=4)

        return (fig, ax)
