#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import bob
import scipy.io.wavfile as wavfile
import numpy as np
import csv
import logging

SAMPLES_PATH = '../agender_distribution/'
TRAIN_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_train.txt'
TEST_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_devel.txt'

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class Trainer(object):
    RATE = 8000
    DIAGONALS = 35

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
        signal = np.cast['float'](signal) # vector should be in **float**
        mfcc = self.c(signal)
        return mfcc

    def get_kmeans_means(self, data):
        """Returns means of given data"""
        kmeans = bob.machine.KMeansMachine(self.DIAGONALS, 20)
        kmeansTrainer = bob.trainer.KMeansTrainer()
        kmeansTrainer.max_iterations = 200
        kmeansTrainer.convergence_threshold = 1e-5
        kmeansTrainer.train(self.kmeans, data)
        return kmeans.means

    def get_machine(self, means):
        gmm = bob.machine.GMMMachine(self.DIAGONALS, 20)
        gmm.means = means
        return gmm

    def get_trainer(self):
        """Create trainer for gmm machine"""
        trainer = bob.trainer.ML_GMMTrainer(True, True, True)
        trainer.convergence_threshold = 1e-5
        trainer.max_iterations = 200
        return trainer

    def extract_data(self, class_number):
        """Extract mfcc from samples and create dataset"""
        logging.debug("Extracting data for class: {0}".format(class_number))
        data = None
        file_number = 0
        for file_path, sample_class in self.sample_generator(TRAIN_SAMPLES_FILE):
            if sample_class != class_number:
                continue
            file_number += 1
            rate, signal =  wavfile.read(file_path)
            #  VAD  & MFCC extraction
            mfcc = self.get_mfcc(signal)
            try:
                data = np.vstack((data, mfcc))
            except ValueError:
                data = mfcc
            if file_number % 100 == 0:
                logging.debug("File number {0}".format(file_number))
        logging.debug("Extracting data FINISHED for class: {0}".format(class_number))
        return data

    def train_machine(self, class_number):
        """Trains one gmm machine with class depicted by class_number"""
        logging.debug("Training machine #{0}".format(class_number))
        data = self.extract_data(class_number)
        means = self.get_kmeans_means(data)
        gmm = self.get_machine(means)
        trainer = self.get_trainer(gmm)
        trainer.train(gmm, data)
        logging.debug("Machine #{0} training FINISHED".format(class_number))
        return gmm


    def train(self):
        """Trains gmm machine with data from train part"""
        for class_number in range(1, 8):
            gmm = self.train_machine(class_number)
            hdf5_file = bob.io.HDF5FILE('gmm{0}.hdf5'.format(class_number), 'w')
            gmm.save(hdf5_file)
            del hdf5_file # close descriptor


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
