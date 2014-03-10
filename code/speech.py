#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import bob
import scipy.io.wavfile as wavfile
import numpy
import csv

SAMPLES_PATH = '../agender_distribution/'
TRAIN_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_train.txt'
TEST_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_devel.txt'


class Classifier(object):
    RATE = 8000

    def __init__(self):
        self.create_mfcc(self.RATE)

    def classify(self, age, gender):
        """Returns class computed of age and gender"""
        result = -1
        if age <= 14:
            return 1
        ages = [24, 54, 80]
        for index, ag in enumerate(ages):
            if age <= ag:
                result = index
        if gender == 'm':
            result += 1
        return result


    def sample_generator(self, file_path):
        """Generate tuple(path, class) of sample from given txt file"""
        with open(file_path) as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                path = '/'.join([SAMPLES_PATH, line[0]])
                sample_class = self.classify(int(line[3]), line[4])
                yield path, sample_class

    def create_mfcc(self, rate):
        """Create mfcc extractor"""
        win_length_ms = 20 # The window length of the cepstral analysis in milliseconds
        win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
        n_filters = 24 # The number of filter bands
        n_ceps = 19 # The number of cepstral coefficients
        f_min = 0. # The minimal frequency of the filter bank
        f_max = 4000. # The maximal frequency of the filter bank
        delta_win = 2 # The integer delta value used for computing the first and second order derivatives
        pre_emphasis_coef = 0.97 # The coefficient used for the pre-emphasis
        dct_norm = True # A factor by which the cepstral coefficients are multiplied
        mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
        self.c = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps,
                        f_min, f_max, delta_win, pre_emphasis_coef, mel_scale,
                        dct_norm)
        self.c.with_energy = True # VAD

    def get_mfcc(self, signal):
        """Returns MFCC of given signal"""
        signal = numpy.cast['float'](signal) # vector should be in **float**
        mfcc = self.c(signal)
        return mfcc

    def train_kmeans(self, data):
        kmeans = bob.machine.KMeansMachine(7, 20)
        kmeansTrainer = bob.trainer.KMeansTrainer()
        kmeansTrainer.max_iterations = 200
        kmeansTrainer.convergence_threshold = 1e-5
        kmeansTrainer.train(self.kmeans, data)
        return kmeans

    def get_machine(self):
        # 7 diagonal (classes), dimension 20
        gmm = bob.machine.GMMMachine(7, 20)
        gmm.means = self.kmeans.means
        # gmm.weights = numpy.array([0.4, 0.6], 'float64')
        # gmm.means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
        # gmm.variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64')
        return gmm

    def get_trainer(self):
        """Create trainer for gmm machine"""
        trainer = bob.trainer.ML_GMMTrainer(True, True, True)
        trainer.convergence_threshold = 1e-5
        trainer.max_iterations = 200
        return trainer

    def extract_data(self):
        """Extract mfcc from samples and create dataset"""
        for file_path, sample_class in self.sample_generator(TRAIN_SAMPLES_FILE):
            rate, signal =  wavfile.read(file_path)
            #  VAD  & MFCC extraction
            mfcc = self.get_mfcc(signal)
            data = 0 # FIXME create data from mfcc and class
        return data

    def train(self):
        """Trains gmm machine with data from train part"""
        self.data = self.extract_data()
        self.kmeans = self.train_kmeans(self.data)
        self.gmm = self.get_machine()
        self.trainer = self.get_trainer(self.gmm)
        self.trainer.train(self.gmm, self.data)

if __name__ == "__main__":
    classifier = Classifier()
    classifier.train()
