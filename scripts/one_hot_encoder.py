from sklearn.base import BaseEstimator, TransformerMixin

class CustomizedOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories):
        self.categories = categories
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        one_hot_encoded = []
        for seq in X:
            one_hot_encoded_seq = []
            for aa in seq.upper():
                encoding = [1 if cat == aa else 0 for cat in self.categories]
                one_hot_encoded_seq.extend(encoding)
            one_hot_encoded.append(one_hot_encoded_seq)
        return one_hot_encoded