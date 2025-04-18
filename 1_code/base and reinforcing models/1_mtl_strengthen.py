# coding: utf-8

# Introduction

"""
The multi-task model extended the RESNET convolutional base with three fully connected layers.
The model has two outputs with different network heads: one head is the classification output,
the other represents a demographic (age / gender etc.)

Loss functions:
A mean squared error loss together with a last layer linear mapping is used for the annotation-regression task.
For the binary classification task, again a cross-entropy loss and
sigmoid activation function of the nodes are used. The contribution of the different losses are equal.
The resulting loss values are summed and minimised during network training.
"""
from mtl_strengthen_model import MTLStrengthenModel
from environment_variables import EnvVars

NAME = '1_mtl_strengthen'
PROJECT = 'HINTS'

env = EnvVars.getInstance()
model = MTLStrengthenModel()

model.build_model()
if env.get_env_var('load_model')=='True':
    # load model
    model.read_test_data()
    model.load_model()
    model.predict_model()
    model.report_performance_metrics()
else:
    # train model
    model.read_data()
    model.fit_model()
    if env.get_env_var('save_model')=="True":
        model.save_model()
    model.predict_model()
    model.report_metrics()