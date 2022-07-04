import pandas as pd
from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms


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

        condition_indices = set(i for i in indices if self.data["RTD3316_condition1 - Condition Description"].notna()[i])
        qualifier_indices = set(i for i in indices if self.data["Price Qualifier"].notna()[i])
        council_tax_indices = set(i for i in indices if self.data["DESC Council Tax Band"].notna()[i])

        self.valid_indices = self.valid_indices & condition_indices & qualifier_indices & council_tax_indices

        rooms = [self.handler.s3_rooms[i] for i in self.valid_indices]
        self.extract_room = ExtractRooms(rooms, self.handler.s3_rooms_set, "{} ({} sqm){}")

    def __call__(self, no_nan_filename, room_filename):
        return self.remove_nan_rows().to_csv(no_nan_filename, index=False), \
               self.get_room_dataset().to_csv(room_filename, index=False)

    def remove_nan_rows(self, *col_names) -> pd.DataFrame:
        if len(col_names) != 0:
            column_names = col_names
        else:
            parking_names = [i for i in self.data.columns if "parking" in i]
            outside_names = [i for i in self.data.columns if "outside" in i]
            heating_names = [i for i in self.data.columns if "heating" in i]
            accessibility_names = [i for i in self.data.columns if "accessibility" in i]
            condition_names = [i for i in self.data.columns if "condition" in i]
            column_names = ["Postcode", "Sale or Let", "EweMove Description S3 Rooms", "Price / Rent",
                            "Price Qualifier", "DESC Council Tax Band", "# of Enquiry or viewings", "# of Apps/Offers"]
            column_names += parking_names + outside_names + heating_names + accessibility_names + condition_names

        valid_indices = sorted(list(self.valid_indices))
        input_data = self.data.iloc[valid_indices][column_names]
        return input_data.loc[:, ~input_data.columns.isin(["EweMove Description S3 Rooms", "Price / Rent"])]

    def get_room_dataset(self, *operation, exclude_other_room=[], **rooms) -> pd.DataFrame:
        # The number of operations should be the same as types of room
        # if default values are not used
        if exclude_other_room is None:
            exclude_other_room = []
        if len(rooms) != 0 and len(rooms) != len(operation):
            raise ValueError("Length of input arguments mismatch, len(rooms) = {} and len(operation) = {}".
                             format(len(rooms), len(operation)))

        if len(rooms) != 0:
            room_mapping = {k: v for k, v in rooms.items()}
            operations = {k: v for k, v in zip(rooms.keys(), operation)}
        else:
            room_mapping = {"bedroom": ["bedroom"],
                            "kitchen": ["kitchen"],
                            "living": ["living", "reception"],
                            "bathroom": ["bathroom", "wc", "washroom"],
                            "dining": ["dining"]}
            operations = {"bedroom": "split",
                          "kitchen": "number",
                          "living": "sum",
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

        others = pd.DataFrame(self.extract_room.get_rest_rooms(*exclude_other_room))
        others = others.rename(columns={0: "Other Number", 1: "Other Area"})
        room_info.append(others)

        return pd.concat(room_info, axis=1)


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")

    creation = CreateInputDataset(data)
    creation("../datasets/no_nan_general_test.csv", "../datasets/no_nan_room_test.csv")
