import pandas as pd
import re
import parse
import random
from bs4 import BeautifulSoup


class ProcessHTML:
    def __init__(self):
        self.s1_description = []
        self.s3_rooms_set = set()
        self.s3_rooms = []
        self.s4_summary = []
        self.price_or_rent = []

        self.extract_numeric = "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"

    def EweMove_Description_S1_Features(self, info: str) -> None:
        if pd.isna(info):
            self.s1_description.append(None)
            return None

        soup = BeautifulSoup(info, "html.parser")
        features = soup.select("li")

        result = []
        for feature in features:
            result.append(feature.string.strip())

        self.s1_description.append(result)

    def EweMove_Description_S3_Rooms(self, info: str) -> None:
        if pd.isna(info):
            self.s3_rooms.append(None)
            return

        soup = BeautifulSoup(info, "html.parser")
        rooms = soup.select("li")

        result = {}
        for room in rooms:
            name = room.strong.string.split('-')[-1].strip()
            # In case there are two rooms with the same name
            if name in result.keys():
                name = "{}{}".format(name, random.randint(0, 100))
            self.s3_rooms_set.add(name)

            try:
                area_or_number = room.i.string
            except AttributeError:
                area_or_number = 1

            # In case there are two rooms with the same name
            result[name] = area_or_number

        self.s3_rooms.append(result)

    def EweMove_Description_S4_Summary(self, info: str) -> None:
        if pd.isna(info):
            self.s4_summary.append(None)
            return

        summary = list(filter(lambda s: s.startswith("<b>"), info.split("<li>")))
        single = parse.compile("<b>{}</b><br>")
        double = parse.compile("<b>{}</b><br><br>{}<br><br>")
        final1 = parse.compile("<b>{}</b><br><br>{}<br></li>")
        final2 = parse.compile("<b>{}</b><br><br>{}<br></li><br><br>{}")
        final3 = parse.compile("<b>{}</b></li>")
        final4 = parse.compile("<b>{}</b></li><br><br>{}")
        parsers = [single, double, final1, final2, final3, final4]

        result = []
        for s in summary:
            res = list(filter(lambda r: r is not None, map(lambda p: p.parse(s), parsers)))[0]
            result.append(res.fixed)

        self.s4_summary.append(result)

    def price_rent(self, info: str) -> None:
        if pd.isna(info):
            self.price_or_rent.append(None)
            return

        price_qualifier = info.split("<br>")[-1]
        price = re.findall(self.extract_numeric, info)[0]

        self.price_or_rent.append((price, price_qualifier))


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    handler = ProcessHTML()

    description = data["EweMove Description S1 Features"]
    rooms = data["EweMove Description S3 Rooms"]
    summary = data["EweMove Description S4 Summary"]
    price = data["Price / Rent"]

    count = 0
    for d, r, s, p in zip(description, rooms, summary, price):
        print(count)
        count += 1
        #handler.EweMove_Description_S1_Features(d)
        handler.EweMove_Description_S3_Rooms(r)
        #handler.EweMove_Description_S4_Summary(s)
        #handler.price_rent(p)

    for room in handler.s3_rooms:
        if room is None:
            print("None\n\n")
            continue

        for k, v in rooms.items():
            print("{:30s}{}".format(k, v))

        print("\n\n")
