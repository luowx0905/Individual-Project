from CreateInputDataset import CreateInputDataset
from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms
from GeneralizeDataset import GeneralizeDataset
from unittest import TestCase
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import unittest


class TestCreateInputDataset(TestCase):
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    creation = CreateInputDataset(data.copy())

    def test_get_general_dataset(self):
        column_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description",
                        "# of Enquiry or viewings", "# of Apps/Offers"]

        result = self.creation.get_general_dataset()
        truth = self.data[column_names].copy()

        encode_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description"]
        encoder = LabelEncoder()
        for name in encode_names:
            encoder.fit(self.data[name])
            truth[name] = pd.DataFrame(encoder.transform(truth[name]))
        truth = truth.iloc[result.index]

        self.assertEqual(result.equals(truth), True)

    def test_get_room_dataset(self):
        room_mapping = {"bedroom": ["bedroom"],
                        "kitchen": ["kitchen"],
                        "living": ["living", "reception"],
                        "bathroom": ["bathroom", "wc", "washroom"],
                        "dining": ["dining", "diner"]}
        operations = {"bedroom": "number",
                      "kitchen": "number",
                      "living": "number",
                      "bathroom": "number",
                      "dining": "number"}

        result = self.creation.get_room_dataset().round(1)

        count = 0
        for i in result.index:
            handler = ProcessHTML()
            handler.EweMove_Description_S3_Rooms(self.data["EweMove Description S3 Rooms"].iloc[i])
            extract = ExtractRooms(handler.s3_rooms, handler.s3_rooms_set, "{} ({} sqm){}")

            info = []
            for k in room_mapping.keys():
                info.append(*extract.get_rooms(*room_mapping[k], operation=operations[k]))
            temp = extract.get_rest_rooms()
            info.append(temp[0][0])
            info.append(temp[0][1])
            info = [round(i, 1) for i in info]

            if not all(info[j] == result.loc[i].values[j] for j in range(len(info))):
                count += 1

        print("Round errors = {}".format(count))
        if count > 0.005 * len(result):
            self.fail()

    def test_get_categorical_dataset(self):
        result = self.creation.get_categorical_dataset()

        generalize = GeneralizeDataset(self.data)
        features = ["parking", "outside_space", "heating", "accessibility"]
        for feature in features:
            temp = generalize.get_feature_types(feature)
            temp = pd.DataFrame(temp).iloc[result.index]
            self.assertEqual(result[temp.columns].equals(temp), True)

    def test_get_labels(self):
        result = self.creation.get_labels()
        encoder = LabelEncoder()

        complete = pd.DataFrame(self.data["Completed"])
        complete = pd.DataFrame(np.where(complete.isna(), "Not Completed", "Completed"), columns=complete.columns)
        encoder.fit(complete)
        complete = pd.DataFrame({"Completed": encoder.transform(complete)}).iloc[result.index]

        handler = ProcessHTML()
        for p in self.data["Price / Rent"]:
            handler.price_rent(p)

        price = [i[0] for i in handler.price_or_rent]
        price = pd.DataFrame({"Price": price}).iloc[result.index]

        truth = pd.concat([complete, price], axis=1)

        self.assertEqual(result.equals(truth), True)


if __name__ == '__main__':
    unittest.main()
