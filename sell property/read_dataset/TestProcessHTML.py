from unittest import TestCase
from ProcessHTML import ProcessHTML
import pandas as pd
import unittest


class TestHandleHTML(TestCase):
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")

    def test_ewe_move_description_s1_features(self):
        info = self.data["EweMove Description S1 Features"].iloc[0]
        handler = ProcessHTML()
        handler.EweMove_Description_S1_Features(info)

        truth = ['CASH BUYERS ONLY', 'No Upper Chain!', 'Very Rare 2 Car Parking Spaces',
                   'Great Commuter Links by Road & Rail', 'Communal Gardens', '2 Bedrooms',
                   'Cul-De-Sac Location', 'Brand New Bathroom', 'Lovely Residential Area']

        self.assertEqual(len(truth), len(handler.s1_description[0]))
        for t, result in zip(truth, handler.s1_description[0]):
            self.assertEqual(t, result)

    def test_ewe_move_description_s3_rooms(self):
        info = self.data["EweMove Description S3 Rooms"].iloc[0]
        handler = ProcessHTML()
        handler.EweMove_Description_S3_Rooms(info)

        truth = {'Entrance Hall': 1, 'Living/Dining Room': '6.58m x 3.78m (24.8 sqm) - 21\' 7" x 12\' 4" (267 sqft)',
                 'Kitchen': '2.68m x 2.14m (5.7 sqm) - 8\' 9" x 7\' (61 sqft)',
                 'Bedroom 1': "3.37m x 2.45m (8.2 sqm) - 11' x 8' (88 sqft)",
                 'Bedroom 2': '2.54m x 2.45m (6.2 sqm) - 8\' 4" x 8\' (67 sqft)',
                 'Bathroom': '2.14m x 2.04m (4.3 sqm) - 7\' x 6\' 8" (46 sqft)', 'Garden': 1, 'Parking': 1}

        self.assertEqual(len(truth), len(handler.s3_rooms[0]))
        for k in truth.keys():
            self.assertEqual(truth[k], handler.s3_rooms[0][k])

    def test_ewe_move_description_s4_summary(self):
        info = self.data["EweMove Description S4 Summary"].iloc[0]
        handler = ProcessHTML()
        handler.EweMove_Description_S4_Summary(info)

        truth = [('Council Tax: ', 'Band B'), ('Energy Performance Certificate (EPC) Rating:', 'Band C (69-80)')]

        self.assertEqual(len(truth), len(handler.s4_summary[0]))
        for true, result in zip(truth, handler.s4_summary[0]):
            self.assertEqual(len(true), len(result))
            for t, res in zip(true, result):
                self.assertEqual(t, res)

    def test_price_rent(self):
        info = self.data["Price / Rent"].iloc[0]
        handler = ProcessHTML()
        handler.price_rent(info)

        truth = (140000, 'Offers In Excess Of')

        self.assertEqual(len(truth), len(handler.price_or_rent[0]))
        for t, result in zip(truth, handler.price_or_rent[0]):
            self.assertEqual(t, result)


if __name__ == '__main__':
    unittest.main()
