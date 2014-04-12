#!/usr/bin/env python2.7
#-*- encoding: utf-8 -*-

import os
import sys
try:
    import bob
except ImportError:
    sys.path.append(os.path.dirname(os.path.expanduser('/datapool/home/miikapi/bob/bob/lib64/python2.7/site-packages/')))
    import bob

import numpy
from matplotlib import pyplot

# Training is a 3-step thing
data = bob.db.iris.data()
trainer = bob.trainer.FisherLDATrainer()
machine, eigen_values = trainer.train(data.values())

# A simple way to forward the data
output = {}
for key in data.keys():
    output[key] = machine(data[key])

# Performance
negatives = numpy.vstack([output['setosa'], output['versicolor']])[:, 0]
positives = output['virginica'][:, 0]
import pdb; pdb.set_trace()

# Plot ROC curve
bob.measure.plot.roc(negatives, positives)
pyplot.xlabel("False Virginica Acceptance (%)")
pyplot.ylabel("False Virginica Rejection (%)")
pyplot.title("ROC Curve for Virginica Classification")
pyplot.grid()
pyplot.axis([0, 5, 0, 15])  #xmin, xmax, ymin, ymax
