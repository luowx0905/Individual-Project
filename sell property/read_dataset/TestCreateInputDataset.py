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
    filename = "../datasets/H1 2020.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    creation = CreateInputDataset(data.copy())

    def test_get_general_dataset(self):
        column_names = ["Postcode", "Sale or Let", "Price Qualifier", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description",
                        "# of Enquiry or viewings", "# of Apps/Offers"]
        encode_names = ["Postcode", "Price Qualifier", "Sale or Let", "DESC Council Tax Band",
                        "RTD3316_condition1 - Condition Description"]
        result = self.creation.get_general_dataset()

        truth = CreateInputDataset.extract_prices(self.data.copy())
        if "Postcode" not in truth.columns:
            truth = CreateInputDataset.extract_postcode(truth)
        valid_indices = CreateInputDataset.remove_invalid_values(truth)
        truth = truth.iloc[valid_indices]
        truth = truth.rename(index={i: j for i, j in zip(truth.index, range(len(truth)))})

        encoder = LabelEncoder()
        for name in encode_names:
            encoder.fit(truth[name])
            truth[name] = pd.DataFrame(encoder.transform(truth[name]))

        self.assertEqual(result.equals(truth[column_names]), True)

    def test_get_room_dataset(self):
        self.fail()

    def test_get_categorical_dataset(self):
        self.fail()

    def test_get_labels(self):
        self.fail()

    def test_extract_postcode(self):
        data = pd.DataFrame({"Full Address": ["3 Cromwell Crescent SW5 9QN",
                                              "146 Queens Road NG9 2FF"],
                             "City": ["London", "Nottingham"]})
        truth = pd.DataFrame({"Postcode": ["SW5 9QN", "NG9 2FF"],
                              "City": ["London", "Nottingham"]})

        extracted = CreateInputDataset.extract_postcode(data)
        self.assertEqual(extracted.equals(truth), True)

    def test_extract_price(self):
        data = pd.DataFrame({"Price / Rent": ["<font color='blue'>£650</font><br>Monthly",
                                              "<font color='blue'>£141,375</font><br>Guide Price",
                                              "<font color='blue'>£275,000</font><br>Offers Invited",
                                              "<font color='blue'>£1</font><br>Monthly",
                                              "<font color='blue'>£600</font><br>Monthly",
                                              "<font color='blue'>£1</font><br>Monthly",
                                              "<font color='blue'>£0</font><br>",
                                              np.nan,
                                              "<font color='blue'>£0</font><br>Monthly"]})
        truth = pd.DataFrame({"Price / Rent": [650.0, 141375.0, 275000.0, 1.0, 600.0, 1.0, 0.0, -1.0, 0.0]})

        extracted = CreateInputDataset.extract_prices(data)
        self.assertEqual(extracted.equals(truth), True)

    def test_remove_invalid_values(self):
        rooms = ["This home includes:<ul><li><strong>01 - Approach</strong><br><br>Becketts Lane is a quiet, one-way, suburban street just off Stocks Lane. This property is  a few doors in, on the left.<br><br></li...",
                 "1203	RH2 8JB	15-05-2018	17-05-2018	NaN	NaN	03-03-2020	Sale	NaN	<ul><li>Wonderful Family Home</li><li>Very Popular Location</li><li>Three Good Sized Bedrooms</li><li>Off-Road Parking</li><li>Good Sized South Westerly Facing Rear Garden</li><li>Close to Excelle...	You could be just the second owner of this great, solid, family home!<br><br>For sale by the first owner after 40 or so years, this has been a very happy home. Built in the 1970s this house has se...	This home includes:<ul><li><strong>01 - Approach</strong><br><br>Bordered by a picket fence with a metal gate, a path through the front lawn with mature shrubs leads us  to the front door with a p...	Additional Information:<br><li><b>Windows and doors</b><br><br>uPVC windows and doors are installed throughout<br><br><li><b>Council Tax: </b><br><br>Band D<br><br><li><b>Energy Performance Certif...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	<font color='blue'>£375,000</font><br>	NaN	NaN	10-10-2027	Band D	NaN	NaN	NaN	0	0",
                 "This home includes:<ul><li><strong>01 - Approach</strong><br><br>Driveway with off street parking, access to garage, partly fenced, planted borders, path to front door and to side with gated acces...",
                 "This home includes:<ul><li><strong>01 - Approach</strong><br><br>Located on the town centre side of the wide open space of Cotmandene, the entrance to the apartment is down a driveway to the rear ...",
                 "This home includes:<ul><li><strong>01 - Porch</strong><br><br>uPVC double glazed French doors open in to the porch area, with tiled floor, ceiling light point and uPVC double glazed side window pa...",
                 "This home includes:<ul><li><strong>01 - Porch</strong><br><br>uPVC double glazed French doors open in to the porch area, with tiled floor, ceiling light point and uPVC double glazed side window pa...",
                 "This home includes:<ul><li><strong>01 - Porch</strong><br><br>uPVC double glazed French doors open in to the porch area, with tiled floor, ceiling light point and uPVC double glazed side window pa...",
                 np.nan]
        prices = ["<font color='blue'>£650</font><br>Monthly",
                  "<font color='blue'>£275,000</font><br>Offers Invited",
                  "<font color='blue'>£275,000</font><br>Offers Invited",
                  "<font color='blue'>£1</font><br>Monthly",
                  "<font color='blue'>£1</font><br>Monthly",
                  "<font color='blue'>£1</font><br>Monthly",
                  np.nan,
                  "<font color='blue'>£0</font><br>Monthly"]
        postcode = ["SW5 9QN", "SW5 9QN", "NG9 2FF", "CB2 1TQ", "NG7 2RD", np.nan, "SW7 2BX", "W2 4QW"]
        condition = ["Good", "Good", "Good", "Good", np.nan, "Good", "Good", "Good"]
        price_qualifier = ["Monthly", "Daily", "Monthly", np.nan, "Monthly", "Monthly", "Monthly", "Monthly"]
        council_tax = ["Band A", "Band Z", np.nan, "Band A", "Band A", "Band A", "Band A", "Band A"]

        data = pd.DataFrame({"EweMove Description S3 Rooms": rooms, "Price / Rent": prices,
                             "Postcode": postcode, "RTD3316_condition1 - Condition Description": condition,
                             "Price Qualifier": price_qualifier, "DESC Council Tax Band": council_tax})
        data = CreateInputDataset.extract_prices(data)
        truth = [0, 1]

        extracted = CreateInputDataset.remove_invalid_values(data)
        self.assertEqual(truth == extracted, True)


if __name__ == '__main__':
    unittest.main()
