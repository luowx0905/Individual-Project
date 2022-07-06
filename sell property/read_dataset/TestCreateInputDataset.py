from CreateInputDataset import CreateInputDataset
from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms
from GeneralizeDataset import GeneralizeDataset
from unittest import TestCase
import pandas as pd
import unittest


class TestCreateInputDataset(TestCase):
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    creation = CreateInputDataset(data)

    def test_get_general_dataset(self):
        column_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "# of Enquiry or viewings", "# of Apps/Offers"]

        result = self.creation.get_general_dataset()
        truth = self.data.iloc[result.index][column_names]
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
        if count > 0.003 * len(result):
            self.fail()

    def test_get_categorical_dataset(self):
        result = self.creation.get_categorical_dataset()

        generalize = GeneralizeDataset(self.data)
        features = ["parking", "outside_space", "heating", "accessibility", "condition"]
        for feature in features:
            temp = generalize.get_feature_types(feature)
            temp = pd.DataFrame(temp).iloc[result.index]
            self.assertEqual(result[temp.columns].equals(temp), True)

    def test_get_labels(self):
        result = self.creation.get_labels()

        for i in result.index:
            complete = self.data.Completed.loc[i]
            complete = "Not Completed" if pd.isna(complete) else "Completed"

            price = self.data["Price / Rent"].loc[i]
            handler = ProcessHTML()
            handler.price_rent(price)
            price = handler.price_or_rent[0][0]

            self.assertEqual(result.Completed.loc[i] == complete, True)
            self.assertEqual(result.Price.loc[i] == price, True)


if __name__ == '__main__':
    unittest.main()
