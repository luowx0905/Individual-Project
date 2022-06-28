from unittest import TestCase
from GeneralizeDataset import GeneralizeDataset
import pandas as pd
import unittest


class TestGeneralizeDataset(TestCase):
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")

    def test_get_parking_types(self):
        generalize = GeneralizeDataset(self.data)
        types = generalize.get_feature_types("parking")

        result = {"Allocated": 343, "Communal": 81, "Covered": 36, "Driveway": 1257, "Garage": 842,
                  "Gated": 89, "Off Street": 1209, "On Street": 458, "Permit": 70, "Private": 258,
                  "Rear": 27, "Residents": 90}

        self.assertEqual(len(types), 12)
        for k in types.keys():
            self.assertEqual(sum(types[k]), result[k])

    def test_get_parking_num(self):
        generalize = GeneralizeDataset(self.data)
        num = generalize.get_feature_num("parking")

        self.assertEqual(sum(num), 4760)

    def test_get_outside_space_types(self):
        generalize = GeneralizeDataset(self.data)
        types = generalize.get_feature_types("outside_space")

        result = {"Back Garden": 1366, "Communal Garden": 227, "Enclosed Garden": 735, "Front Garden": 682,
                  "Patio": 431, "Private Garden": 599, "Rear Garden": 802, "Terrace": 93}

        self.assertEqual(len(types), 8)
        for k in types.keys():
            self.assertEqual(sum(types[k]), result[k])

    def test_get_outside_space_num(self):
        generalize = GeneralizeDataset(self.data)
        num = generalize.get_feature_num("outside_space")

        self.assertEqual(sum(num), 4935)


if __name__ == '__main__':
    unittest.main()
