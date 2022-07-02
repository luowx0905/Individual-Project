import pandas as pd
import numpy as np
import parse


class GeneralizeDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = ["parking", "outside_space", "heating", "accessibility"]

    def get_feature_types(self, feature: str):
        """
        Obtain feature types for every property
        :param feature: parking, outside_space
        :return: dict
        """
        if feature not in self.features:
            raise ValueError("{} is invalid".format(feature))

        # Select all the columns related to the feature
        names = list(filter(lambda s: feature in s, self.data.columns))
        raw_data = self.data[names].copy()

        # Fill all the missing values to None
        raw_data = raw_data.fillna("None")
        types = np.unique(raw_data.to_numpy().flatten())

        # Create a dict for every feature types except None
        result = {key: [] for key in types if key != "None"}

        # Obtain parking types for every property
        for i in range(len(raw_data)):
            temp = {key: 0 for key in types if key != "None"}

            first = raw_data.iloc[i, 0]
            second = raw_data.iloc[i, 1]
            third = raw_data.iloc[i, 2]
            temp = self.record_types(first, second, third, temp)

            list(map(lambda d: result[d].append(temp[d]), temp.keys()))

        return result

    def get_feature_num(self, feature: str):
        if feature not in self.features:
            raise ValueError("{} is invalid".format(feature))

        names = list(filter(lambda s: feature in s, self.data.columns))
        raw_data = self.data[names]

        result = []
        list(map(lambda d: result.append(raw_data.iloc[d].count()), range(len(raw_data))))

        return result

    @staticmethod
    def record_types(first: str, second: str, third: str, types: dict):
        if first in types.keys():
            types[first] += 1
        if second in types.keys():
            types[second] += 1
        if third in types.keys():
            types[third] += 1

        return types


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    generalize = GeneralizeDataset(data)

    types = generalize.get_feature_types("accessibility")
    for k, v in types.items():
        print("{:20s}{:5d}".format(k, sum(v)))
