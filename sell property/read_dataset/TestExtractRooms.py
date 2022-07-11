from ProcessHTML import ProcessHTML
from ExtractRooms import ExtractRooms
from unittest import TestCase
import unittest


class TestExtractRooms(TestCase):
    info1 = """This home includes:
    <ul>
        <li>
            <strong>01 - Entrance Hall</strong><br><br>
        </li>
        <li>
            <strong>02 - Living/Dining Room</strong><br><br>
            <i>6.58m x 3.78m (24.8 sqm) - 21' 7" x 12' 4" (267 sqft)</i><br><br>
        </li>
        <li>
            <strong>03 - Kitchen</strong><br><br>
            <i>2.68m x 2.14m (5.7 sqm) - 8' 9" x 7' (61 sqft)</i><br><br>
        </li>
        <li>
            <strong>04 - Bedroom 1</strong><br><br>
            <i>3.37m x 2.45m (8.2 sqm) - 11' x 8' (88 sqft)</i><br><br>
        </li>
        <li>
            <strong>05 - Bedroom 2</strong><br><br>
            <i>2.54m x 2.45m (6.2 sqm) - 8' 4" x 8' (67 sqft)</i><br><br>
            The second double bedroom is bright and well-sized, with room for all required furniture.<br><br>
        </li>
        <li>
            <strong>06 - Bathroom</strong><br><br>
            <i>2.14m x 2.04m (4.3 sqm) - 7' x 6' 8" (46 sqft)</i><br><br>
        </li>
        <li>
            <strong>07 - Garden</strong><br><br>
            Communal Gardens.<br><br>
        </li>
        <li>
            <strong>08 - Parking</strong><br><br>
            2 allocated parking spaces.<br><br>
        </li>
    </ul>"""
    info2 = """This home includes:
        <ul>
            <li>
                <strong>01 - Entrance Porch</strong><br><br>
            </li>
            <li>
                <strong>02 - Lounge Diner</strong><br><br>
                <i>6.76m x 4.04m (27.3 sqm) - 22' 2" x 13' 3" (293 sqft)</i><br><br>
            </li>
            <li>
                <strong>03 - Kitchen</strong><br><br>
                <i>2.97m x 2.36m (7 sqm) - 9' 8" x 7' 8" (75 sqft)</i><br><br>
            </li>
            <li>
                <strong>05 - Bathroom</strong><br><br>
            </li>
            <li>
                <strong>07 - Bedroom (Double)</strong><br><br>
                <i>4.05m x 3.25m (13.1 sqm) - 13' 3" x 10' 7" (142 sqft)</i><br><br>
            </li>
            <li>
                <strong>08 - Bedroom (Double)</strong><br><br>
                <i>3.28m x 2.36m (7.7 sqm) - 10' 9" x 7' 8" (83 sqft)</i><br><br>
            </li>
            <li>
                <strong>09 - Bedroom (Double)</strong><br><br>
                <i>4.3m x 2.44m (10.4 sqm) - 14' 1" x 8' (112 sqft)</i><br><br>
            </li>
            <li>
                <strong>10 - Bathroom</strong><br><br>
            </li>
        </ul>"""
    handler = ProcessHTML()
    handler.EweMove_Description_S3_Rooms(info1)
    handler.EweMove_Description_S3_Rooms(info2)

    extract = ExtractRooms(handler.s3_rooms, handler.s3_rooms_set, "{} ({} sqm){}")

    def test_get_rooms(self):
        result = self.extract.get_rooms("bedroom", operation="mean")
        self.assertEqual(round(result[0], 1), 7.2)
        self.assertEqual(round(result[1], 1), 10.4)

        result = self.extract.get_rooms("bedroom", operation="sum")
        self.assertEqual(round(result[0], 1), 14.4)
        self.assertEqual(round(result[1], 1), 31.2)

        result = self.extract.get_rooms("bedroom", operation="number")
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 3)

        result = self.extract.get_rooms("bedroom")
        self.assertEqual(result[0]["0"], 8.2)
        self.assertEqual(result[0]["1"], 6.2)
        self.assertEqual(result[0]["2"], 0)
        self.assertEqual(result[1]["0"], 13.1)
        self.assertEqual(result[1]["1"], 7.7)
        self.assertEqual(result[1]["2"], 10.4)

    def test_get_rest_rooms(self):
        self.extract.get_rooms("bedroom")
        result = self.extract.get_rest_rooms()
        self.assertEqual(result[0][0], 6)
        self.assertEqual(result[0][1], 34.8)
        self.assertEqual(result[1][0], 5)
        self.assertEqual(result[1][1], 34.3)

    def test_get_room_statistics(self):
        result = self.extract.get_room_statistics([i for i in self.handler.s3_rooms_set if "bedroom" in i.lower()])
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], 2)

    def test_get_room_areas(self):
        result = self.extract.get_room_areas([i for i in self.handler.s3_rooms_set if "bedroom" in i.lower()])
        self.assertEqual(round(result, 2), 9.12)


if __name__ == '__main__':
    unittest.main()
