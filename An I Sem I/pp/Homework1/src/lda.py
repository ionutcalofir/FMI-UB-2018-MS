import pickle
from lda_model import LDAModel

class LDA():
    pass
    def fit(self, data_path, lda_type, n_topics, plot=False, n_period=None):
        self.lda = LDAModel(n_topics,
                            lda_type,
                            n_period)
        self.lda.fit(data_path, plot)

    def save_model(self, model_path='models/lda.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.lda, f)

    def load_model(self, model_path='models/lda.pkl'):
        with open(model_path, 'rb') as f:
            self.lda = pickle.load(f)

    def get_similarity(self, plot):
        self.lda.get_similarity(plot)

    def predict(self, data_path, plot):
        self.lda.predict(data_path, plot)
