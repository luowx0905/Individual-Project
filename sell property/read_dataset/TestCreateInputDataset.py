from CreateInputDataset import CreateInputDataset
from unittest import TestCase
import pandas as pd
import unittest


class TestCreateInputDataset(TestCase):
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    creation = CreateInputDataset(data)

    def test_remove_nan_rows(self):
        result = self.creation.remove_nan_rows()
        result = result.rename(index={i: j for i, j in zip(result.index, range(len(result)))})
        truth = pd.read_csv("../datasets/no_nan_general_test.csv")

        self.assertEqual(result.shape, truth.shape)
        self.assertEqual(result.equals(truth), True)

    def test_get_room_dataset(self):
        result = self.creation.get_room_dataset().round(1)
        result = result.rename(index={i: j for i, j in zip(result.index, range(len(result)))})
        truth = pd.read_csv("../datasets/no_nan_room_test.csv").round(1)

        self.assertEqual(result.shape, truth.shape)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if abs(result.iloc[i, j] - truth.iloc[i, j]) > 0.1 * truth.iloc[i, j]:
                    print("({}, {}) result = {}\t truth = {}".format(i, j, result.iloc[i, j], truth.iloc[i, j]))
                    self.fail()


if __name__ == '__main__':
    unittest.main()
