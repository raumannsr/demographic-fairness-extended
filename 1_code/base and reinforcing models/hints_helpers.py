from environment_variables import EnvVars

class LoggerSingleton:
    def __init__(self):
        self.__verbose = EnvVars.getInstance().get_env_var('verbose')

    @classmethod
    def instance(cls):
        if not hasattr(LoggerSingleton, "_instance"):
            LoggerSingleton._instance = LoggerSingleton()
        return LoggerSingleton._instance

    def log(self, string):
        if self.__verbose:
            print(string)