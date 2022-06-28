import pandas as pd
import numpy as np


class GeneralizeDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        self.parking_types = None
        self.parking_num = None

        self.outside_space_types = None
        self.outside_space_num = None

    def get_parking_types(self):
        # Select all the columns related to parking spaces
        parking_name = list(filter(lambda s: "parking" in s, self.data.columns))
        parking = self.data[parking_name].copy()

        # Fill all the missing values to None
        parking = parking.fillna("None")
        types, counts = np.unique(parking.to_numpy().flatten(), return_counts=True)

        # Create a dict for every parking types except None
        self.parking_types = {key: [] for key in types if key != "None"}

        # Obtain parking types for every property
        for i in range(len(parking)):
            types = {key: 0 for key in types if key != "None"}

            first = parking.iloc[i, 0]
            second = parking.iloc[i, 1]
            third = parking.iloc[i, 2]
            types = self.record_types(first, second, third, types)

            list(map(lambda d: self.parking_types[d].append(types[d]), types.keys()))

    def get_parking_num(self):
        parking_name = list(filter(lambda s: "parking" in s, self.data.columns))
        parking = self.data[parking_name]

        self.parking_num = [parking.iloc[i].count() for i in range(len(parking))]

    def get_outside_space_types(self):
        outside_name = list(filter(lambda s: "outside_space" in s, self.data.columns))
        outside = self.data[outside_name].copy()

        outside = outside.fillna("None")
        types, counts = np.unique(outside.to_numpy().flatten(), return_counts=True)

        self.outside_space_types = {key: [] for key in types if key != "None"}

        for i in range(len(outside)):
            types = {key: 0 for key in types if key != "None"}

            first = outside.iloc[i, 0]
            second = outside.iloc[i, 1]
            third = outside.iloc[i, 2]
            types = self.record_types(first, second, third, types)

            list(map(lambda d: self.outside_space_types[d].append(types[d]), types.keys()))

    def get_feature_types(self, feature: str):
        """
        Obtain feature types for every property
        :param feature: parking, outside_space
        :return: None
        """
        names = list(filter(lambda s: feature in s, self.data.columns))
        raw_data = self.data[names].copy()

        raw_data = raw_data.fillna("None")
        types = np.unique(raw_data.to_numpy().flatten())

        result = None
        if feature.lower() == "parking":
            self.parking_types = {key: [] for key in types if key != "None"}
            result = self.parking_types
        elif feature.lower() == "outside_space":
            self.outside_space_types = {key: [] for key in types if key != "None"}
            result = self.outside_space_types
        else:
            raise ValueError("{} is invalid".format(feature))

        for i in range(len(raw_data)):
            types = {key: 0 for key in types if key != "None"}

            first = raw_data.iloc[i, 0]
            second = raw_data.iloc[i, 1]
            third = raw_data.iloc[i, 2]
            types = self.record_types(first, second, third, types)

            list(map(lambda d: result[d].append(types[d]), types.keys()))

    def get_outside_space_num(self):
        outside_name = list(filter(lambda s: "outside_space" in s, self.data.columns))
        outside = self.data[outside_name]

        self.outside_space_num = [outside.iloc[i].count() for i in range(len(outside))]

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

    generalize.get_outside_space_num()
    for k, v in enumerate(generalize.outside_space_num):
        print("{:15d}{:5d}".format(k, v))
