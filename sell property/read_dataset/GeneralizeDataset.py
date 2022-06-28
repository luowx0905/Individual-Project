import pandas as pd
import numpy as np


class GeneralizeDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        self.parking_types = None
        self.parking_num = None

    def get_parking_types(self):
        # Select all the columns related to parking spaces
        parking_name = list(filter(lambda s: "parking" in s, data.columns))
        parking = data[parking_name].cpoy()

        # Fill all the missing values to None
        parking = parking.fillna("None")
        types, counts = np.unique(parking.to_numpy().flatten(), return_counts=True)

        # Create a dict for every parking types except None
        self.parking_types = {key: [] for key in types if key != "None"}

        # Obtain parking types for every property
        for i in range(len(parking)):
            first = parking.iloc[i, 0]
            second = parking.iloc[i, 1]
            third = parking.iloc[i, 2]
            types = self.get_parking_spaces_for_property(first, second, third)

            list(map(lambda value, d: self.parking_types[d].append(types[value]),
                     types.keys(), types.keys()))

    def get_parking_num(self):
        parking_name = list(filter(lambda s: "parking" in s, data.columns))
        parking = data[parking_name]

        self.parking_num = [parking.iloc[i].count() for i in range(len(parking))]

    @staticmethod
    def get_parking_spaces_for_property(first: str, second: str, third: str):
        parking_types = {"Allocated": 0, "Off Street": 0, "Residents": 0, "On Street": 0, "Driveway": 0, "Garage": 0,
                         "Permit": 0, "Private": 0, "Gated": 0, "Covered": 0, "Communal": 0, "Rear": 0}

        if first in parking_types.keys():
            parking_types[first] += 1
        if second in parking_types.keys():
            parking_types[second] += 1
        if third in parking_types.keys():
            parking_types[third] += 1

        return parking_types


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    generalize = GeneralizeDataset(data)

    generalize.get_parking_num()
    for i, num in enumerate(generalize.parking_num):
        print("{:5d}{:5d}".format(i, num))
