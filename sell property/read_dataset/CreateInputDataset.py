import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms
from GeneralizeDataset import GeneralizeDataset


class CreateInputDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # Obtain HTML info
        self.handler = ProcessHTML()
        rooms = data["EweMove Description S3 Rooms"]
        price = data["Price / Rent"]
        for room in rooms:
            self.handler.EweMove_Description_S3_Rooms(room)
        for p in price:
            self.handler.price_rent(p)

        # Remove rows with NaN values
        indices = set(range(len(data)))
        room_indices = set(i for i in range(len(self.handler.s3_rooms)) if self.handler.s3_rooms[i] is not None)
        price_indices = set(i for i in range(len(self.handler.price_or_rent)) if self.handler.price_or_rent[i][0] != 0)
        self.valid_indices = indices & room_indices & price_indices

        # Obtain valid indices
        condition_indices = set(i for i in indices if self.data["RTD3316_condition1 - Condition Description"].notna()[i])
        qualifier_indices = set(i for i in indices if self.data["Price Qualifier"].notna()[i])
        council_tax_indices = set(i for i in indices if self.data["DESC Council Tax Band"].notna()[i])
        self.valid_indices = self.valid_indices & condition_indices & qualifier_indices & council_tax_indices
        self.valid_indices = sorted(list(self.valid_indices))

        # Encode categorical feature
        encode_names = ["Postcode", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description"]
        encoder = LabelEncoder()
        for name in encode_names:
            encoder.fit(self.data[name])
            self.data[name] = pd.DataFrame(encoder.transform(self.data[name]))

        # Encode completeness
        complete = pd.DataFrame(self.data["Completed"])
        complete = pd.DataFrame(np.where(complete.isna(), "Not Completed", "Completed"), columns=complete.columns)
        encoder.fit(complete)
        self.data.Completed = pd.DataFrame(encoder.transform(complete))

        # Obtain rooms for valid indices
        rooms = [self.handler.s3_rooms[i] for i in self.valid_indices]
        self.extract_room = ExtractRooms(rooms, self.handler.s3_rooms_set, "{} ({} sqm){}")

        self.generalize = GeneralizeDataset(self.data.iloc[self.valid_indices])

    def __call__(self, general_file, room_file, categorical_file, label_file, operation="types"):
        return self.get_general_dataset().to_csv(general_file, index=False), \
               self.get_room_dataset().to_csv(room_file, index=False), \
               self.get_categorical_dataset(operation).to_csv(categorical_file, index=False), \
               self.get_labels().to_csv(label_file, index=False)

    def get_general_dataset(self) -> pd.DataFrame:
        column_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description",
                        "# of Enquiry or viewings", "# of Apps/Offers"]

        return self.data[column_names].iloc[self.valid_indices]

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

        room_info = []
        for room in room_mapping.keys():
            rooms = self.extract_room.get_rooms(*room_mapping[room], operation=operations[room])

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

        others = pd.DataFrame(self.extract_room.get_rest_rooms())
        others = others.rename(columns={0: "other number", 1: "other area"})
        room_info.append(others)

        result = pd.concat(room_info, axis=1)
        return result.rename(index={k: v for k, v in zip(result.index, self.valid_indices)})

    def get_categorical_dataset(self, operation: str = "types") -> pd.DataFrame:
        column_names = ["parking", "outside_space", "accessibility", "heating"]

        operation = operation.lower()
        if operation == "types":
            operation_func = self.generalize.get_feature_types
        elif operation == "number":
            operation_func = self.generalize.get_feature_num
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
        complete = self.data["Completed"].iloc[self.valid_indices]

        price = [i[0] for i in self.handler.price_or_rent]
        price = pd.DataFrame({"Price": price}).iloc[self.valid_indices]

        return pd.concat([complete, price], axis=1)


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")

    creation = CreateInputDataset(data)
    creation("../datasets/general.csv", "../datasets/room.csv", "../datasets/categorical.csv", "../datasets/label.csv")
