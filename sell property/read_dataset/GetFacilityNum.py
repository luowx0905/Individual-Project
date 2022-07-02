import re
import pandas as pd
from ProcessHTML import ProcessHTML
from GeneralizeDataset import GeneralizeDataset


class GetFacilityNum:
    def __init__(self, s1_features: list, s3_rooms: list):
        self.s1_features = s1_features
        self.s3_rooms = s3_rooms

        self.facility = ["parking", "outside_space", "heating", "accessibility"]
        self.extract_numeric = "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
        self.mapping = {"one": 1, "two": 2, "single": 1, "double": 2}

    def get_parking_num(self, search_index: list):
        result = {str(i): -1 for i in search_index}
        keywords = ["car ", "parking ", "garage"]

        for i in search_index:
            found = False
            for feature in self.s1_features[i]:
                match = list(map(lambda word: word in feature.lower(), keywords))
                if True in match:
                    for key in self.mapping.keys():
                        if key in feature.lower():
                            result[str(i)] = self.mapping[key]
                            found = True
                            break

                    num = re.findall(self.extract_numeric, feature)
                    if len(num) != 0:
                        result[str(i)] = int(num[0])
                        found = True
                        break

            if found or self.s3_rooms[i] is None:
                continue

            count = 0
            for k in self.s3_rooms[i].keys():
                match = list(map(lambda word: word in k.lower(), keywords))
                if True in match:
                    count += 1
            if count != 0:
                result[str(i)] = count

        return result


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    handler = ProcessHTML()

    description = data["EweMove Description S1 Features"]
    rooms = data["EweMove Description S3 Rooms"]

    for d in description:
        handler.EweMove_Description_S1_Features(d)
    for r in rooms:
        handler.EweMove_Description_S3_Rooms(r)

    generalize = GeneralizeDataset(data)
    parking = generalize.get_feature_num("parking")
    search = [i for i in range(len(parking)) if parking[i] != 0]

    facility_num = GetFacilityNum(handler.s1_description, handler.s3_rooms)
    result = facility_num.get_parking_num(search)
    for k, v in result.items():
        print("{:5s}\t{}".format(k, v))