import pandas as pd
import parse

from ProcessHTML import ProcessHTML


class ExtractRooms:
    def __init__(self, rooms: list, room_set: set, extract_rule: str):
        self.rooms = rooms
        self.room_set = room_set

        self.current_rooms = set()
        self.extract_area = parse.compile(extract_rule)

    def get_rooms(self, *args: str, operation: str = "split", handle_na: str = "None") -> list:
        """
        Get info about the rooms based on input names
        :param args: room names
        :param operation: method to process room info in a property
        :param handle_na: methods to handle missing value (no record for the property)
        :return: A list containing info about rooms of input names
        """
        # Obtain all room names of this type
        room_names = [i for i in self.room_set for j in args if j in i.lower()]
        self.current_rooms |= set(room_names)
        # Get the average area of these rooms
        average_room_area = self.get_room_areas(room_names)
        # Obtain the maximum number of rooms in a property and the average number of rooms in a property
        max_room_num, avg_room_num = self.get_room_statistics(room_names)

        all_room_info = []
        # Iterate all the properties in the dataset
        for rooms in self.rooms:
            # Key is the ID of current room and value is its area
            room_info_per_property = {str(k): 0.0 for k in range(max_room_num)}
            count = 0
            if rooms is not None:
                for k, v in rooms.items():
                    if k in room_names:
                        if isinstance(v, str):
                            # Extract the area if applicable
                            room_info_per_property[str(count)] = float(self.extract_area.parse(v)[1])
                        else:
                            # Fill in the average area
                            room_info_per_property[str(count)] = average_room_area
                        count += 1
            # If there is no data about this property
            elif rooms is None and handle_na.lower() == "mean":
                count = avg_room_num
                # Fill in the average area for average number of rooms
                for i in range(int(avg_room_num)):
                    room_info_per_property[str(i)] = average_room_area

            # Calculate the total area of this type of rooms in a property and
            # append it to the final list
            total_room_area = sum(i for i in room_info_per_property.values())
            if operation.lower() == "sum":
                all_room_info.append(total_room_area)
            elif operation.lower() == "mean":
                try:
                    all_room_info.append(total_room_area / count)
                except ZeroDivisionError:
                    all_room_info.append(0)
            elif operation.lower() == "split":
                all_room_info.append(room_info_per_property)
            elif operation.lower() == "number":
                all_room_info.append(count)
            else:
                raise ValueError("{} is invalid".format(operation))

        return all_room_info

    def get_rest_rooms(self, *args: str, handle_na: str = "None") -> list:
        """
        Get the number and total area of the other rooms
        If there is no data for a property and handle_na is set to mean, then both values
        will be set to average, if there are no other rooms then both values are 0
        :param args: room names that should be excluded
        :param handle_na: method to handle missing value (no record for the property)
        :return: A list containing info about other rooms
        """
        other_room_names = self.room_set - self.current_rooms
        exclude_names = [i for i in other_room_names for j in args if j in i.lower()]
        other_room_names -= set(exclude_names)

        average_room_area = self.get_room_areas(other_room_names)
        max_room_num, avg_room_num = self.get_room_statistics(other_room_names)

        all_other_rooms = []
        for rooms in self.rooms:
            other_room_per_property = [0, 0.0]
            if rooms is not None:
                for k, v in rooms.items():
                    if k in other_room_names:
                        other_room_per_property[0] += 1
                        if isinstance(v, str):
                            other_room_per_property[1] += float(self.extract_area.parse(v)[1])
                        #else:
                            #other_room_per_property[1] += average_room_area
            elif handle_na.lower() == "mean":
                other_room_per_property[0] = avg_room_num
                other_room_per_property[1] = avg_room_num * average_room_area

            all_other_rooms.append(other_room_per_property)

        return all_other_rooms

    def get_room_statistics(self, room_names) -> tuple:
        """
        Obtain the statistics of rooms base on input names
        :param room_names: the name of rooms for getting statistics
        :return: maximum number of the room and average number of room in a property
        """
        room_num = []
        for rooms in self.rooms:
            if rooms is not None:
                room_num.append(len([i for i in rooms.keys() if i in room_names]))
            else:
                room_num.append(0)

        return max(room_num), round(sum(room_num) / len(room_num))

    def get_room_areas(self, room_names) -> float:
        """
        obtain the room average area based on the input names
        :param room_names: the names of rooms for averaging
        :return: average room area
        """
        areas = []
        for rooms in self.rooms:
            if rooms is None:
                continue

            for room, area in rooms.items():
                if room in room_names and isinstance(area, str):
                    areas.append(float(self.extract_area.parse(area)[1]))

        return sum(areas) / len(areas)


if __name__ == '__main__':
    filename = "../datasets/PropertyData_wDesc.csv"
    data = pd.read_csv(filename, encoding="ISO8859-1")
    handler = ProcessHTML()

    for r in data["EweMove Description S3 Rooms"]:
        #handler.EweMove_Description_S3_Rooms(r)
        pass

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

    result = extract.get_rooms("bedroom", operation="split")
    result = extract.get_rest_rooms()

    print(result)

