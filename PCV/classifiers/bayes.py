from __future__ import print_function
import numpy as np


class BayesClassifier(object):
    def __init__(self):
        """ Initialize classifier with training data. """

        self.labels = []  # class labels
        self.mean = []  # class mean
        self.var = []  # class variances
        self.n = 0  # nbr of classes

    def train(self, data, labels=None):
        """ Train on data (list of arrays n*dim). 
            Labels are optional, default is 0...n-1. """

        if labels is None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)

        for c in data:
            self.mean.append(np.mean(c, axis=0))
            self.var.append(np.var(c, axis=0))

    def classify(self, points):
        """ Classify the points by computing probabilities 
            for each class and return most probable label. """

        # compute probabilities for each class
        est_prob = np.array([gauss(m, v, points) for m, v in zip(self.mean, self.var)])

        print('est prob', est_prob.shape, self.labels)
        # get index of highest probability, this gives class label
        ndx = est_prob.argmax(axis=0)

        est_labels = np.array([self.labels[n] for n in ndx])

        return est_labels, est_prob


def gauss(m, v, x):
    """ Evaluate Gaussian in d-dimensions with independent 
        mean m and variance v at the points in (the rows of) x. 
        http://en.wikipedia.org/wiki/Multivariate_normal_distribution """

    if len(x.shape) == 1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape

    # covariance matrix, subtract mean
    S = np.diag(1 / v)
    x = x - m
    # product of probabilities
    y = np.exp(-0.5 * np.diag(np.dot(x, np.dot(S, x.T))))

    # normalize and return
    return y * (2 * np.pi) ** (-d / 2.0) / (np.sqrt(np.prod(v)) + 1e-6)
