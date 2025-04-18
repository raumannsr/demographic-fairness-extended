import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

import base_model
from environment_variables import EnvVars
from generate_data import isic_generate_data_stl
from keras import applications, layers, models, optimizers
from hints_utils import get_input_shape

class STLModel(base_model.BaseModel):
    def __init__(self):
        self.env = EnvVars.getInstance()

    def read_train_val_test_data(self):
        path = self.env.get_env_var('train_val_test_path') + '/'
        self.train = pd.read_csv(path + self.env.get_env_var('train_file'))
        self.validation = pd.read_csv(path + self.env.get_env_var('validation_file'))
        self.test = pd.read_csv(path + self.env.get_env_var('test_file'))
        self.train_data = isic_generate_data_stl(directory=self.env.get_env_var('image_path'), augmentation=True, batch_size=int(self.env.get_env_var('batch_size')), file_list=self.train.isic_id, label_1=self.train.target)
        self.validation_data = isic_generate_data_stl(directory=self.env.get_env_var('image_path'), augmentation=False, batch_size=int(self.env.get_env_var('batch_size')), file_list=self.validation.isic_id, label_1=self.validation.target)
        y = self.train['target'].to_numpy()
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights = {i: class_weights[i] for i in range(2)}
        return class_weights
    
    def read_test_data(self):
        path = self.env.get_env_var('train_val_test_path') + '/'
        self.test = pd.read_csv(path + self.env.get_env_var('test_file'))
        if self.env.get_env_var('archive') == "PAD-UFES":
            self.test['isic_id'] = self.test['img_id'].str.split('.').str[0]

    def build_model(self):
        w, h, d = get_input_shape(self.env.get_env_var('input_shape'))
        lr = float(self.env.get_env_var('lr'))
        momentum = float(self.env.get_env_var('momentum'))
        conv_base = applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(w, h, d))
        x = layers.Flatten()(conv_base.output)
        out_class = layers.Dense(1, activation='sigmoid', name='out_class')(x)
        self.model = models.Model(conv_base.input, outputs=[out_class])
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=lr, momentum=momentum), metrics=['acc'])
        if self.env.get_env_var('verbose')=="True":
            self.model.summary()

    def predict_model(self):
        test = isic_generate_data_stl(directory=self.env.get_env_var('image_path'), augmentation=False,
                                              batch_size=int(self.env.get_env_var('batch_size')), file_list=self.test.isic_id,
                                              label_1=self.test.target)
        predictions = self.model.predict(test, steps=int(self.env.get_env_var('pred_steps')))
        y_true = self.test.target
        delta_size = predictions.size - y_true.count()
        scores = np.resize(predictions, predictions.size - delta_size)
        df = pd.DataFrame({'isic_id': self.test.isic_id, 'prediction': scores, 'true_label': y_true})
        df.to_csv(self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/' + self.env.get_env_var('prediction_file'), index=False)
        auc = roc_auc_score(df['true_label'], df['prediction'])
        self.df_auc = self.df_auc.append({'seed': self.env.get_env_var('seed'), 'auc': auc}, ignore_index=True)