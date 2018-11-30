from sklearn.externals import joblib

class RegressorPred():
    def load_model(self, path):
        self.model = joblib.load(path) 
        return self.model
