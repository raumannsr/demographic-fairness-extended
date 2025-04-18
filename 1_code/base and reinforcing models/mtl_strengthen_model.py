import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

import base_model
from environment_variables import EnvVars
from generate_data import isic_generate_data_mtl
from hints_utils import get_input_shape
from keras import applications, layers, models, optimizers
from keras.callbacks import EarlyStopping

class MTLStrengthenModel(base_model.BaseModel):
    def __init__(self):
        self.env = EnvVars.getInstance()

    def read_data(self):
        demographic_feature = self.env.get_env_var('demographic_feature')        
        path = self.env.get_env_var('train_val_test_path') + '/'

        self.train = pd.read_csv(path + self.env.get_env_var('train_file'))
        self.validation = pd.read_csv(path + self.env.get_env_var('validation_file'))
        self.test = pd.read_csv(path + self.env.get_env_var('test_file'))
        
        if demographic_feature == 'sex':
            self.train['dmg'] = self.train['sex'].apply(lambda x: 1 if x == 'female' else 0)
            self.validation['dmg'] = self.validation['sex'].apply(lambda x: 1 if x == 'female' else 0)
            self.test['dmg'] = self.test['sex'].apply(lambda x: 1 if x == 'female' else 0)
        else:
            # normalisation of age_approx values, maximum age over all available instances is 85
            self.train['dmg'] = self.train['age_approx'] / 85.0
            self.validation['dmg'] = self.validation['age_approx'] / 85.0
            self.test['dmg'] = self.test['age_approx'] / 85.0
    
        self.train_data = isic_generate_data_mtl(directory=self.env.get_env_var('image_path'), augmentation=True, batch_size=int(self.env.get_env_var('batch_size')), file_list=self.train.isic_id, label_1=self.train.target, label_2=self.train.dmg)
        self.validation_data = isic_generate_data_mtl(directory=self.env.get_env_var('image_path'), augmentation=False, batch_size=int(self.env.get_env_var('batch_size')), file_list=self.validation.isic_id, label_1=self.validation.target, label_2=self.validation.dmg)
        
        y = self.train['target'].to_numpy()
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights = {i: class_weights[i] for i in range(2)}
        return class_weights
    
    def build_model(self):
        demographic_feature = self.env.get_env_var('demographic_feature')
        if demographic_feature == 'sex':
            out_demographic_loss = 'binary_crossentropy'
        else:
            out_demographic_loss = 'mse'

        w, h, d = get_input_shape(self.env.get_env_var('input_shape'))
        conv_base = applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(w, h, d))
        lr = float(self.env.get_env_var('lr'))
        momentum = float(self.env.get_env_var('momentum'))

        x = layers.Flatten()(conv_base.output)
        x = layers.Dense(256, activation='relu')(x)
        out_class = layers.Dense(1, activation='sigmoid', name='out_class')(x)
        out_demographic = layers.Dense(1, activation='sigmoid', name='out_demographic')(x)
        self.model = models.Model(conv_base.input, outputs=[out_class, out_demographic])
        self.model.compile(
            optimizer=optimizers.RMSprop(learning_rate=lr, momentum=momentum),
            loss={'out_class': 'binary_crossentropy', 'out_demographic': out_demographic_loss},
            loss_weights={'out_class': 0.5, 'out_demographic': 0.5},
            metrics={'out_class': 'accuracy'})
        if self.env.get_env_var('verbose')=="True":
            self.model.summary()
    
    def fit_model(self):
        early_stopping = EarlyStopping(monitor='val_out_class_accuracy', patience=10, verbose=1, restore_best_weights=True)
        steps_per_epoch = int(self.env.get_env_var('steps_per_epoch'))
        epochs = int(self.env.get_env_var('num_epochs'))
        validation_steps = int(self.env.get_env_var('val_steps'))

        self.history = self.model.fit(
            self.train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps,
            validation_data=self.validation_data,
            callbacks=[early_stopping])
        
    def read_test_data(self):
        demographic_feature = self.env.get_env_var('demographic_feature') 
        path = self.env.get_env_var('train_val_test_path') + '/'
        self.test = pd.read_csv(path + self.env.get_env_var('test_file'))
        if self.env.get_env_var('archive') == "PAD-UFES":
            self.test['isic_id'] = self.test['img_id'].str.split('.').str[0]
        if demographic_feature == 'sex':
            self.test['dmg'] = self.test['gender'].apply(lambda x: 1 if x == 'FEMALE' else 0)
        else:
            # normalisation of age_approx values, maximum age over all available instances is 85
            self.test['dmg'] = self.test['age'] / 85.0

    def predict_model(self):
        test = isic_generate_data_mtl(directory=self.env.get_env_var('image_path'),
                                          augmentation=False,
                                          batch_size=int(self.env.get_env_var('batch_size')),
                                          file_list=self.test.isic_id,
                                          label_1=self.test.target,
                                          label_2=self.test.dmg)
        predictions = self.model.predict(test, steps=int(self.env.get_env_var('pred_steps')))
        y_true = self.test.target
        delta_size = predictions[0].size - y_true.count()
        scores = np.resize(predictions[0], predictions[0].size - delta_size)
        df = pd.DataFrame({'isic_id': self.test.isic_id, 'prediction': scores, 'true_label': y_true, 'dmg' : self.test.dmg})
        df.to_csv(self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/' + self.env.get_env_var('prediction_file'), index=False)
        auc = roc_auc_score(df['true_label'], df['prediction'])
        self.df_auc = self.df_auc.append({'seed': self.env.get_env_var('seed'), 'auc': auc}, ignore_index=True)