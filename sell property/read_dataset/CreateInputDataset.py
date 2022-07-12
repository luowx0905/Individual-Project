import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms
from GeneralizeDataset import GeneralizeDataset


class CreateInputDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        if "Postcode" not in self.data.columns:
            self.data = CreateInputDataset.extract_postcode(self.data)

        self.data = CreateInputDataset.extract_prices(self.data)
        self.valid_indices = CreateInputDataset.remove_invalid_values(self.data)

        self.data = self.data.iloc[self.valid_indices]
        self.data = self.data.rename(index={i: j for i, j in zip(self.data.index, range(len(self.data)))})

    @staticmethod
    def extract_postcode(data: pd.DataFrame) -> pd.DataFrame:
        pattern = "[A-Za-z]{1,2}[0-9Rr][0-9A-Za-z]? [0-9][ABD-HJLNP-UW-Zabd-hjlnp-uw-z]{2}"
        postcodes = []
        for i in data["Full Address"]:
            postcode = re.findall(pattern, i)
            if len(postcode) == 0:
                postcodes.append(np.nan)
            else:
                postcodes.append(postcode[0])
        data["Full Address"] = postcodes
        data = data.rename(columns={"Full Address": "Postcode"})
        data = data[data["Postcode"].notna()]
        data = data.rename(index={i: j for i, j in zip(data.index, range(len(data)))})

        return data

    @staticmethod
    def extract_prices(data: pd.DataFrame) -> pd.DataFrame:
        handler = ProcessHTML()
        for price in data["Price / Rent"]:
            handler.price_rent(price)

        prices = [i[0] for i in handler.price_or_rent]
        data["Price / Rent"] = prices

        return data

    @staticmethod
    def remove_invalid_values(data: pd.DataFrame) -> list:
        indices = set(range(len(data)))
        room_indices = set(i for i in range(len(data)) if data["EweMove Description S3 Rooms"].notna()[i])
        price_indices = set(i for i in range(len(data)) if data["Price / Rent"][i] > 0)
        valid_indices = indices & room_indices & price_indices

        postcode_indices = set(i for i in indices if data["Postcode"].notna()[i])
        condition_indices = set(i for i in indices if data["RTD3316_condition1 - Condition Description"].notna()[i])
        qualifier_indices = set(i for i in indices if data["Price Qualifier"].notna()[i])
        council_tax_indices = set(i for i in indices if data["DESC Council Tax Band"].notna()[i])
        valid_indices = valid_indices & condition_indices & qualifier_indices & council_tax_indices & postcode_indices

        return sorted(list(valid_indices))

    def get_general_dataset(self) -> pd.DataFrame:
        column_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description",
                        "# of Enquiry or viewings", "# of Apps/Offers"]
        encode_names = ["Postcode", "Price Qualifier", "Sale or Let", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description"]

        encoder = LabelEncoder()
        for name in encode_names:
            encoder.fit(self.data[name])
            self.data[name] = pd.DataFrame(encoder.transform(self.data[name]))

        result = self.data[column_names]

        return result.rename(index={i: j for i, j in zip(result.index, self.valid_indices)})

    def get_room_dataset(self) -> pd.DataFrame:
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

        # Obtain HTML info
        handler = ProcessHTML()
        rooms = self.data["EweMove Description S3 Rooms"]
        for room in rooms:
            handler.EweMove_Description_S3_Rooms(room)
        extract_room = ExtractRooms(handler.s3_rooms, handler.s3_rooms_set, "{} ({} sqm){}")

        room_info = []
        for room in room_mapping.keys():
            rooms = extract_room.get_rooms(*room_mapping[room], operation=operations[room])

            rooms = pd.DataFrame(rooms)
            _, col = rooms.shape
            rename_dict = None
            if operations[room] == "split":
                rename_dict = {str(i): "{} {}".format(room, i + 1) for i in range(col)}
            elif operations[room] == "sum":
                rename_dict = {0: "{} area".format(room)}
            elif operations[room] == "number":
                rename_dict = {0: "{} number".format(room)}
            elif operations[room] == "mean":
                rename_dict = {0: "{} average".format(room)}
            rooms = rooms.rename(columns=rename_dict)
            room_info.append(rooms)

        others = pd.DataFrame(extract_room.get_rest_rooms())
        others = others.rename(columns={0: "other number", 1: "other area"})
        others = others["other number"]
        room_info.append(others)

        result = pd.concat(room_info, axis=1)
        return result.rename(index={k: v for k, v in zip(result.index, self.valid_indices)})

    def get_categorical_dataset(self, operation: str = "types") -> pd.DataFrame:
        generalize = GeneralizeDataset(self.data)
        column_names = ["parking", "outside_space", "accessibility", "heating"]

        operation = operation.lower()
        if operation == "types":
            operation_func = generalize.get_feature_types
        elif operation == "number":
            operation_func = generalize.get_feature_num
        else:
            raise ValueError("{} is an invalid operation".format(operation))

        info = []
        for name in column_names:
            if operation == "number":
                info.append(pd.DataFrame({name: operation_func(name)}))
            else:
                info.append(pd.DataFrame(operation_func(name)))

        result = pd.concat(info, axis=1)
        return result.rename(index={k: v for k, v in zip(result.index, self.valid_indices)})

    def get_labels(self) -> pd.DataFrame:
        # Encode completeness
        complete = pd.DataFrame(self.data["Completed"])
        complete = pd.DataFrame(np.where(complete.isna(), 0, 1), columns=complete.columns)
        self.data.Completed = complete

        complete = self.data["Completed"]
        price = self.data["Price / Rent"]

        result = pd.concat([complete, price], axis=1)
        return result.rename(index={i: j for i, j in zip(result.index, self.valid_indices)})

    def get_source(self) -> pd.DataFrame:
        if "Source" not in self.data.columns:
            raise ValueError("No valid source")

        result = self.data["Source"]
        return result.rename(index={i: j for i, j in zip(result.index, self.valid_indices)})

    @staticmethod
    def create_dataset(*paths) -> (pd.DataFrame, pd.DataFrame):
        files = [CreateInputDataset.add_origin(path) for path in paths]
        for i in range(len(files)):
            if "Postcode" not in files[i].columns:
                files[i] = CreateInputDataset.extract_postcode(files[i])

        all_datasets = pd.concat(files, axis=0, ignore_index=True)
        creation = CreateInputDataset(all_datasets)

        general = creation.get_general_dataset()
        room = creation.get_room_dataset()
        categorical = creation.get_categorical_dataset()
        labels = creation.get_labels()
        sources= creation.get_source()

        features = pd.concat([general, room, categorical], axis=1)
        return features, labels, sources

    @staticmethod
    def add_origin(path: str) -> pd.DataFrame:
        data = pd.read_csv(path, encoding="ISO8859-1")
        source = ["{}@{}".format(path, i) for i in range(len(data))]
        data["Source"] = source

        return data


if __name__ == '__main__':
    #filename = "../datasets/PropertyData_wDesc.csv"
    filename = "../datasets/H1 2020.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")

    #creation = CreateInputDataset(data)

    folder = "../datasets"
    paths = [os.path.join(folder, path) for path in os.listdir(folder) if "final" not in path]
    features, labels, sources = CreateInputDataset.create_dataset(*paths)

