#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import bob
import scipy.io.wavfile as wavfile
import numpy as np
import csv

SAMPLES_PATH = '../agender_distribution/'
TRAIN_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_train.txt'
TEST_SAMPLES_FILE = SAMPLES_PATH + 'trainSampleList_devel.txt'

class Core(object):
    @staticmethod
    def classify(age, gender):
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

    @staticmethod
    def filter_vad(mfcc):
        energy = mfcc[:, 19]
        energy = np.expand_dims(energy, 0).T
        kmeansTrainer = bob.trainer.KMeansTrainer()
        kmeans = bob.machine.KMeansMachine(2, 1)
        kmeansTrainer = bob.trainer.KMeansTrainer()
        kmeansTrainer.max_iterations = 100
        kmeansTrainer.convergence_threshold = 1e-5
        kmeansTrainer.train(kmeans, energy)
        threshold = (kmeans.means[0] + kmeans.means[1]) / 2
        mfcc = mfcc[mfcc[:, 19] < threshold]
        return mfcc

    @staticmethod
    def create_mfcc(rate):
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
        c = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps,
                        f_min, f_max, delta_win, pre_emphasis_coef, mel_scale,
                        dct_norm)
        c.with_energy = True # VAD
        return c

    @staticmethod
    def get_mfcc(c, signal):
        """Returns MFCC of given signal
        :param c:
        """
        signal = np.cast['float'](signal) # vector should be in **float**
        mfcc = c(signal)
        return mfcc


class Trainer(object):
    RATE = 8000
    DIAGONALS = 35

    def __init__(self):
        self.c = Core.create_mfcc(self.RATE)

    def sample_generator(self, file_path):
        """Generate tuple(path, class) of sample from given txt file"""
        with open(file_path) as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                path = '/'.join([SAMPLES_PATH, line[0]])
                sample_class = Core.classify(int(line[3]), line[4])
                yield path, sample_class


    def get_kmeans_means(self, data):
        """Returns means of given data"""
        kmeans = bob.machine.KMeansMachine(self.DIAGONALS, 20)
        kmeansTrainer = bob.trainer.KMeansTrainer()
        # https://groups.google.com/forum/#!topic/bob-devel/VOi8k0Ts1gw
        kmeansTrainer.initialization_method = kmeansTrainer.KMEANS_PLUS_PLUS
        kmeansTrainer.max_iterations = 200
        kmeansTrainer.convergence_threshold = 1e-5
        kmeansTrainer.train(kmeans, data)
        if np.any(np.isnan(kmeans.means)):
            import pdb; pdb.set_trace()
        return kmeans.means

    def get_empty_machine(self, means):
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
        print "Extracting data for class: {0}".format(class_number)
        data = None
        file_number = 0
        for file_path, sample_class in self.sample_generator(TRAIN_SAMPLES_FILE):
            if sample_class != class_number:
                continue
            file_number += 1
            rate, signal =  wavfile.read(file_path)
            #  VAD  & MFCC extraction
            mfcc = Core.filter_vad(Core.get_mfcc(self.c, signal))
            if np.any(np.isnan(mfcc)):
                import pdb; pdb.set_trace()
            try:
                data = np.vstack((data, mfcc))
            except ValueError:
                data = mfcc
            if file_number % 100 == 0:
                print "File number {0}".format(file_number)
                break
        print "Extracting data FINISHED for class: {0}".format(class_number)
        with open('mfccs_{0}.npy'.format(class_number), 'w') as f:
            np.save(f, data)
        return data

    def train_machine(self, class_number):
        """Trains one gmm machine with class depicted by class_number"""
        print "Training machine #{0}".format(class_number)
        data = self.extract_data(class_number)
        means = self.get_kmeans_means(data)
        gmm = self.get_empty_machine(means)
        trainer = self.get_trainer()
        trainer.train(gmm, data)
        print "Machine #{0} training FINISHED".format(class_number)
        return gmm

    def save_machine(self, gmm, file_path):
        hdf5_file = bob.io.HDF5FILE(file_path, 'w')
        gmm.save(hdf5_file)
        del hdf5_file  # close descriptor

    def train(self, machine=0):
        """Trains gmm machine with data from train part"""
        if machine is 0:
            for class_number in range(1, 8):
                gmm = self.train_machine(class_number)
                self.save_machine(gmm, 'gmm{0}.hdf5'.format(class_number))
        else:
            class_number = machine
            gmm = self.train_machine(class_number)
            self.save_machine(gmm, 'gmm{0}.hdf5'.format(class_number))


class Classifier(object):
    def __init__(self):
        self.load_machines(['gmm{0}.hdf5'.format(i) for i in range(1,8)])
        self.c = Core.create_mfcc()

    def load_machines(self, paths):
        """Load all machines from list of paths sorted by theirs classes"""
        self.machines = []
        for file_path in paths:
            hdf5file = bob.io.HDF5FILE(file_path)
            self.machines.append(bob.machine.gmm(hdf5file))
            del hdf5file

    def get_log_likehoods(self, mfcc):
        """Return array of log likehood of each mfcc for each machine"""
        log_likehoods = np.zeros([mfcc.shape[0], len(self.machines)])
        for chunk_index, chunk_mfcc in enumerate(mfcc):
            for machine_index, machine in enumerate(self.machines):
                log_likehoods[chunk_index, machine_index] = machine(chunk_mfcc)
        return log_likehoods


    def classify_file(self, path):
        """Returns class of given wav file in path"""
        rate, signal =  wavfile.read(path)
        mfcc = Core.filter_vad(Core.get_mfcc(self.c, signal))
        log_likehoods = self.get_log_likehoods(mfcc)
        # TODO get overall probabilities that this file is in particulary class

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-machine', type=int, help='Train machine with given index 1-7', default=0)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--train', action='store_true')
    operation.add_argument('--classify')
    args = parser.parse_args()
    if args.train:
        trainer = Trainer()
        trainer.train(args.train_machine)
    elif args.classify:
        wav_path = args.classify
        classifier = Classifier()
        print classifier.classify_file(wav_path)

