import pandas as pd
import keras

from get_data import ISICDataset
from environment_variables import EnvVars
from keras.callbacks import EarlyStopping

class BaseModel:
    train = 0
    validation = 0
    test = 0
    train_data = 0
    validation_data = 0
    train_dmg = 0
    val_dmg = 0
    test_dmg = 0
    class_weights = 0
    pipeline = 0
    history = 0
    df_auc = pd.DataFrame(columns=['seed', 'auc'])
    experiment_id = ""

    def __init__(self):
        self.dataset = ISICDataset()
        self.model = keras.Model()
        self.env = EnvVars.getInstance()

    def load_model(self):
        path = self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/'
        model_json = self.model.to_json()
        filename = path + self.env.get_env_var('model_weights_file')
        self.model.load_weights(filename + '.h5')
        self.model.summary()

    def fit_model_with_class_weights(self, class_weights):
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
        steps_per_epoch = int(self.env.get_env_var('steps_per_epoch'))
        epochs = int(self.env.get_env_var('num_epochs'))
        validation_steps = int(self.env.get_env_var('val_steps'))

        self.history = self.model.fit(
            self.train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps,
            validation_data=self.validation_data,
            class_weight=class_weights,
            callbacks=[early_stopping])        

    def fit_model(self):
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=True)
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

    def save_model(self):
        path = self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/'
        model_json = self.model.to_json()
        filename = path + self.env.get_env_var('model_weights_file')
        with open(filename + '.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename + '.h5')

    def report_metrics(self):
        path = self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/'
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(path + self.env.get_env_var('model_history_file'))
        self.df_auc.to_csv(path + self.env.get_env_var('model_performance_file'))

    def report_performance_metrics(self):
        path = self.env.get_env_var('experiment_path') + '/' + self.env.get_env_var('experiment_id') + '/'
        self.df_auc.to_csv(path + self.env.get_env_var('model_performance_file'))