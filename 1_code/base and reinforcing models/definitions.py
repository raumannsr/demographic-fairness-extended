import pandas as pd

class Dataset:
    def __init__(self, image_id, label):
        self.id = image_id
        self.label = label

    def to_csv(self, filename):
        df = pd.DataFrame()
        df['id'] = self.id
        df['label'] = self.label
        df.to_csv(filename)
