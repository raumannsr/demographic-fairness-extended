import os

class EnvVars:
    __instance = None

    @staticmethod
    def getInstance():
        if EnvVars.__instance == None:
            EnvVars()
        return EnvVars.__instance

    def __init__(self):
        if EnvVars.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            EnvVars.__instance = self
        self.vars = {
            'image_path': os.getenv('HINTS_IMAGE_PATH', None),
            'train_val_test_path': os.getenv('HINTS_TRAIN_VAL_TEST_PATH', None),
            'batch_size': os.getenv('HINTS_BATCH_SIZE', None),
            'num_epochs': os.getenv('HINTS_NUM_EPOCHS', None),
            'steps_per_epoch': os.getenv('HINTS_STEPS_PER_EPOCH', None),
            'val_steps': os.getenv('HINTS_VAL_STEPS', None),
            'pred_steps': os.getenv('HINTS_PRED_STEPS', None),
            'learning_rate': os.getenv('HINTS_LEARNING_RATE', None),
            'momentum': os.getenv('HINTS_MOMENTUM', None),
            'input_shape': os.getenv('HINTS_INPUT_SHAPE', None),
            'train_file': os.getenv('HINTS_TRAIN_FILE', None),
            'validation_file': os.getenv('HINTS_VALIDATION_FILE', None),
            'pipeline_path': os.getenv('HINTS_PIPELINE_PATH', None),
            'experiment_path': os.getenv('HINTS_EXPERIMENT_PATH', None),
            'experiment_id': os.getenv('HINTS_EXPERIMENT_ID', None),
            'test_file': os.getenv('HINTS_TEST_FILE', None),
            'prediction_file': os.getenv('HINTS_PREDICTION_FILE', None),
            'lamda': os.getenv('HINTS_LAMDA', None),
            'save_model': os.getenv('HINTS_SAVE_MODEL', None),
            'model_weights_file': os.getenv('HINTS_MODEL_WEIGHTS_FILE', None),
            'model_history_file': os.getenv('HINTS_MODEL_HISTORY_FILE', None),
            'seed': os.getenv('HINTS_SEED', None),
            'lr': os.getenv('HINTS_LEARNING_RATE', None),
            'momentum': os.getenv('HINTS_MOMENTUM', None),
            'model_performance_file': os.getenv('HINTS_MODEL_PERFORMANCE_FILE', None),
            'demographic_feature': os.getenv('HINTS_DEMOGRAPHIC_FEATURE', 'sex'),
            'verbose': os.getenv('HINTS_VERBOSE', None)
        }

    def set_env_var(self, key, value):
        self.vars[key] = value

    def get_env_var(self, key):
        return self.vars.get(key)