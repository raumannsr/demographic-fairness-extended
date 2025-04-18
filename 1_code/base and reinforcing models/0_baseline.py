# coding: utf-8
"""
Skin Lesion Classification Baseline Model
-----------------------------------------
The baseline model predicts a binary label (melanoma or not) from a 
skin lesion image. The model uses a convolutional base with additional specific
layers. We used ResNet50 as the encoder, which consists of pooling and 
convolution layers with fixed pre-trained ImageNet weights.

Author: Ralf Raumanns
Date: February 21, 2025
"""

from stl_model import STLModel
from environment_variables import EnvVars

NAME = '0_baseline'
PROJECT = 'HINTS'

env = EnvVars.getInstance()
model = STLModel()

model.build_model()
if env.get_env_var('load_model')=='True':
    # load model
    model.read_test_data()
    model.load_model()
    model.predict_model()
    model.report_performance_metrics()
else:
    # train model
    model.read_train_val_test_data()
    model.fit_model()
    if env.get_env_var('save_model')=="True":
        model.save_model()
    model.predict_model()
    model.report_metrics()