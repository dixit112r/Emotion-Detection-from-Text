# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:53:53 2019

@author: dixit
"""
from collections import defaultdict

from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
from nltk.classify.api import ClassifierI

##//////////////////////////////////////////////////////
##  Naive Bayes Classifier
##//////////////////////////////////////////////////////

class NaiveBayes(ClassifierI):
    
    def __init__(self, label_probdist, feature_probdist):
        
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                #print 'Ignoring unseen feature %s' % fname
                del featureset[fname]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label, fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    # nb: This case will never come up if the
                    # classifier was created by
                    # NaiveBayesClassifier.train().
                    logprob[label] += sum_logs([]) # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)



        

    @classmethod
    def train(cls, labeled_featuresets, estimator=ELEProbDist):
        """
        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        """
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist[label] += 1
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname][fval] += 1
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                # Only add a None key when necessary, i.e. if there are
                # any samples with feature 'fname' missing.
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return cls(label_probdist, feature_probdist)
