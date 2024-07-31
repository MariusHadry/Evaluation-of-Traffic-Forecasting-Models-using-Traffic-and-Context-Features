import json
import os
import zipfile
import psycopg2
from tqdm import tqdm

extracted_weather_data: None | list = None
path_to_zip_files = r"data/"
connection = psycopg2.connect(host="", port="",
                              user="", password="",
                              database="lowerFranconia")
cursor = connection.cursor()
last_index = -1


def create_weather_table():
    query = """CREATE TABLE IF NOT EXISTS weather_data (
                id serial NOT NULL PRIMARY KEY,
                "timestamp" timestamp NOT NULL,
                "type" integer,
                "level" integer,
                geometry geometry(MultiPolygon,4326),
                simple_geometry geometry(Polygon,4326)
                );"""
    cursor.execute(query)
    connection.commit()


def insert_weather_community_warnings():
    value_arg = ""

    batch_counter = 0
    batch_size = 5
    extracted_length = len(extracted_weather_data)

    for value in extracted_weather_data:
        batch_counter += 1
        value_arg += f" ('{value['time']}', {value['type']}, {value['level']}, '{value['geometry']}', '{value['simple_geometry']}'), "

        if batch_counter % batch_size == 0 or batch_counter == extracted_length:
            try:
                query = "INSERT INTO weather_data(\"timestamp\", \"type\", \"level\", \"geometry\", \"simple_geometry\") VALUES" + value_arg[:-2] + " ON CONFLICT DO NOTHING"
                cursor.execute(query)
                connection.commit()
                value_arg = ""
            except psycopg2.Error as e:
                print(str(e))


def parse_zip(input_zip):
    with zipfile.ZipFile(input_zip) as input_zip:
        for name in input_zip.namelist():
            if name.endswith(".json"):
                year_month_day = name.split("/")[0]
                time_hours = name.split("/")[1].split("_")[0].split("-")[0]
                time_minutes = name.split("/")[1].split("_")[0].split("-")[1]

                with input_zip.open(name) as file:
                    data = file.read().decode("UTF-8")

                    # check file size and skip empty files
                    if len(data) == 0:
                        continue

                    extract_weather_information_community_warnings(data,
                                                                 f"{year_month_day} {time_hours}:{time_minutes}:00")


def get_simple_geometry_string(geometry) -> str:
    global cursor

    query = "SELECT ST_Transform(ST_ConcaveHull('" + geometry + "', 0.95, False),4326)"
    try:
        cursor.execute(query)
        simple_geometry_string = cursor.fetchone()[0]
        return simple_geometry_string
    except Exception as error:
        print(error)
        cursor.execute("ROLLBACK")
        connection.commit()
        return ""


def get_geometry_string(points_raw, triangles_raw) -> str:
    points = []
    triangles = []

    for i in range(0, len(points_raw), 2):
        points.append([points_raw[i], points_raw[i + 1]])

    for i in range(0, len(triangles_raw), 3):
        triangles.append([points[triangles_raw[i]], points[triangles_raw[i + 1]], points[triangles_raw[i + 2]]])

    multi_polygon = "MULTIPOLYGON("

    for t in triangles:
        # Areas of effect are given as a set of triangles that are constructed here and merged into a polygon
        multi_polygon += f"(({t[0][1]} {t[0][0]}, {t[1][1]} {t[1][0]}, {t[2][1]} {t[2][0]}, {t[0][1]} {t[0][0]})),"

    # make use of postGIS to convert the polygon string into a geojson string
    multi_polygon = multi_polygon[:-1] + ")"
    query = "SELECT ST_GeomFromText('" + multi_polygon + "', 4326)"
    cursor.execute(query)
    geometry_string = cursor.fetchone()[0]

    return geometry_string


def delete_unnecessary_data():
    """
        Deletes weather data outside the bounding box. The bounding box is extracted from the metadata table.
        ST_INTERSECTS returns true for every geometry that shares area with the given geometry.
    """
    global last_index

    filter_query = """DELETE FROM weather_data
                        USING metadata
                        WHERE NOT ST_INTERSECTS(bbox, geometry) and id > """ + str(last_index)
    cursor.execute(filter_query)
    connection.commit()

    query_get_last_index = """ SELECT * FROM weather_data ORDER BY "id" DESC LIMIT 1"""
    cursor.execute(query_get_last_index)
    last_index = int(cursor.fetchone()[0])


def extract_weather_information_community_warnings(data, time):
    try:
        parsed = json.loads(data)
    except ValueError as e:
        return

    for entry in parsed["warnings"]:
        if entry["isVorabinfo"] is False:
            for r in entry["regions"]:
                points = r["polygon"]
                triangles = r["triangles"]
                geom = get_geometry_string(points, triangles)
                simple_geom = get_simple_geometry_string(geom)

                if len(simple_geom) != 0:
                    to_add = {
                        'time': time,
                        'type': entry["type"],
                        'level': entry["level"],
                        'polygon': points,
                        'triangles': triangles,
                        'geometry': geom,
                        'simple_geometry': simple_geom
                    }

                    extracted_weather_data.append(to_add)


if __name__ == '__main__':
    create_weather_table()
    flow_path = path_to_zip_files + "communityWarnings"
    files = [f for f in os.listdir(flow_path)]
    print(f"{len(files)} zip files will be analyzed")
    files.sort()

    for file in tqdm(files):
        if file.endswith(".zip"):
            # clear list before inserting new data
            extracted_weather_data = []
            parse_zip(flow_path + "/" + file)
            insert_weather_community_warnings()
            delete_unnecessary_data()
