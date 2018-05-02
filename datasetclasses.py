import utils
import imageutils
from collections import Counter
from sklearn.utils import shuffle


class Dataset:
    def __init__(self,
                 X_data,
                 y_data,
                 X_valid,
                 y_valid,
                 X_test,
                 y_test,
                 label_mapping_filename):
        self.train = DatasetElements(X_data, y_data)
        self.valid = DatasetElements(X_valid, y_valid)
        self.test = DatasetElements(X_test, y_test)
        self.label_mapping = self.get_label_mapping(label_mapping_filename)

    def get_label_mapping(self, file_path):
        return utils.read_csv(file_path=file_path)

    def add_testing_data(self, X_test, y_test):
        self.test = DatasetElements(X_test, y_test)


class DatasetElements:
    def __init__(self, X_data, y_data):
        self.X = X_data
        self.y = y_data

    @property
    def label_counter(self):
        # Create a counter to count the occurences of a sign (label)
        # values = list(set(self.y))
        # data_counter = Counter(values)
        # # We count each label occurence and store it in our label_counter
        # for label in self.y:
        #     data_counter[label] += 1
        return utils.get_data_count(self.X, self.y)

    @property
    def len(self):
        return 0 if len(self.X) != len(self.y) else len(self.X)

    # def mapping(self):
    #     return self.label_mapping[str(self.y)]

    def shuffle_data(self):
        self.X, self.y = shuffle(self.X, self.y)

    def preprocess(self, normalize, equalize, grayscale):
        return imageutils.preprocess_images(self.X,
                                            grayscale=grayscale,
                                            equalize=equalize,
                                            normalize=True)
