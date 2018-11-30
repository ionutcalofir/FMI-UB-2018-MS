from sklearn.externals import joblib

class ClassifierPred():
    def load_model(self, path):
        self.model = joblib.load(path) 
        return self.model
