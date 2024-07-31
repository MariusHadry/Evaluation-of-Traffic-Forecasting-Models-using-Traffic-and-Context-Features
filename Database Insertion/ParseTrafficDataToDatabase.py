import argparse
import json
import os
import zipfile
import psycopg2

flow_data = {}
incident_data = {}
path_to_zip_files = "data/"

connection = psycopg2.connect(host="", port="",
                              user="", password="",
                              database="lowerFranconia")


def get_frequent_olr_codes():
    global flow_data

    # TODO: adjust frequency depending on data!
    query = """SELECT DISTINCT olr_eid_mapping.olr_code
                FROM olr_eid_mapping, olr_codes
                WHERE olr_eid_mapping.olr_code = olr_codes.olr_code and frequency >= 100"""

    flow_data = {}
    cursor = connection.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    while row:
        flow_data[row[0]] = []
        row = cursor.fetchone()


def reset_incident_data():
    global incident_data
    incident_data = {}


def create_flow_table():
    query = """CREATE TABLE IF NOT EXISTS flow_data (
                olr_code VARCHAR(256) NOT NULL,
                "timestamp" timestamp NOT NULL,
                speed numeric(10,7),
                speedUncapped numeric(10,7),
                freeFlow numeric(10,7),
                jamFactor numeric(4,2),
                confidence numeric(4,3),
                traversability VARCHAR(32),
                speed_percentage numeric(10,7),
                PRIMARY KEY(olr_code, "timestamp"))"""
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()


def create_incidents_table():
    query = """ CREATE TABLE IF NOT EXISTS incident_data (
                incident_id bigint NOT NULL PRIMARY KEY,
                olr_code VARCHAR(256) NOT NULL,
                original_id bigint,
                "start_time" timestamp,
                "end_time" timestamp,
                road_closed boolean,
                criticality VARCHAR(32),
                "type" VARCHAR(32),
                summary VARCHAR(1024))"""
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()


def date_already_in_database(date_str):
    query = f"""SELECT timestamp
                FROM public.flow_data
                WHERE timestamp = '{date_str}'
                LIMIT 1"""

    cursor = connection.cursor()
    cursor.execute(query)

    row = cursor.fetchone()
    if row:
        return True
    return False

def insert_flow():
    cursor = connection.cursor()
    value_arg = ""
    progress_counter = 0
    size = len(flow_data.keys())

    for olr, values in flow_data.items():
        progress_counter += 1

        if len(values) == 0:
            continue

        for v in values:
            value_arg += f" ('{olr}', '{v['time']}', {v['speed']}, {v['speedUncapped']}, {v['freeFlow']}, {v['jamFactor']}, {v['confidence']}, '{v['traversability']}', '{v['speed']/v['freeFlow']:.6f}'), "

        if progress_counter % 100 == 0 or progress_counter == size:
            if progress_counter % 500 == 0:
                print(f"\t{progress_counter}/{size}")

            try:
                # cursor.execute(insert_query, (olr, v["time"], v["speed"], v["speedUncapped"], v["freeFlow"], v["jamFactor"], v["confidence"], v["traversability"]))
                query = "INSERT INTO flow_data(olr_code, \"timestamp\", speed, speedUncapped, freeFlow, jamFactor, confidence, traversability, speed_percentage) VALUES" + value_arg[:-2] + " ON CONFLICT DO NOTHING"
                cursor.execute(query)
                connection.commit()
                value_arg = ""
            except psycopg2.Error as e:
                print(str(e))


def insert_incidents():
    cursor = connection.cursor()
    for incident_id, v in incident_data.items():
        value_arg = f" ({incident_id}, '{v['olr_code']}', {v['original_id']}, '{v['start_time']}', '{v['end_time']}', " \
                    f"{v['road_closed']}, '{v['criticality']}', '{v['incident_type']}', '{v['summary']}'), "
        try:
            # cursor.execute(insert_query, (olr, v["time"], v["speed"], v["speedUncapped"], v["freeFlow"], v["jamFactor"], v["confidence"], v["traversability"]))
            query = "INSERT INTO incident_data(incident_id, olr_code, original_id, start_time, end_time, road_closed, criticality, \"type\", summary) VALUES" + value_arg[:-2] + " ON CONFLICT DO NOTHING"
            cursor.execute(query)
            connection.commit()
        except psycopg2.Error as e:
            print(str(e))


def parse_zip(input_zip, is_flow_data):
    try:
        with zipfile.ZipFile(input_zip) as input_zip:
            for name in input_zip.namelist():
                if name.endswith(".txt"):
                    year_month_day = name.split("/")[0]
                    time_hours = name.split("/")[1].split("_")[0].split("-")[0]
                    time_minutes = name.split("/")[1].split("_")[0].split("-")[1]

                    with input_zip.open(name) as file:
                        data = file.read().decode("UTF-8")

                        # check file size and skip empty files
                        if len(data) == 0:
                            continue

                        if is_flow_data:
                            # extract information from files
                            extract_flow_information(data, f"{year_month_day} {time_hours}:{time_minutes}:00")
                        else:
                            extract_incidents_information(data, f"{year_month_day} {time_hours}:{time_minutes}:00")
    except:
        print(f"could not read file {file}")


def extract_flow_information(data, time):
    parsed = json.loads(data)
    for entry in parsed["results"]:
        olr = entry["location"]["olr"]
        if olr in flow_data:
            to_add = {
                'time': time,
                'speed': entry["currentFlow"]["speed"] if "speed" in entry["currentFlow"] else -1,
                'speedUncapped': entry["currentFlow"]["speedUncapped"] if "speedUncapped" in entry["currentFlow"] else -1,
                'freeFlow': entry["currentFlow"]["freeFlow"] if "freeFlow" in entry["currentFlow"] else -1,
                'jamFactor': entry["currentFlow"]["jamFactor"] if "jamFactor" in entry["currentFlow"] else -1,
                'confidence': entry["currentFlow"]["confidence"] if "confidence" in entry["currentFlow"] else -1,
                'traversability': entry["currentFlow"]["traversability"] if "traversability" in entry["currentFlow"] else "unknown"
            }
            flow_data[olr].append(to_add)


def extract_incidents_information(data, time):
    parsed = json.loads(data)
    # incident_id, olr_code, start_time, end_time, road_closed, criticality, type, summary
    for entry in parsed["results"]:
        incident_id = entry["incidentDetails"]["id"]
        olr_code = entry["location"]["olr"]
        origianl_id = entry["incidentDetails"]["originalId"]
        start_time = entry["incidentDetails"]["startTime"]
        end_time = entry["incidentDetails"]["endTime"]
        road_closed = entry["incidentDetails"]["roadClosed"]
        criticality = entry["incidentDetails"]["criticality"]
        incident_type = entry["incidentDetails"]["type"]
        summary = entry["incidentDetails"]["summary"]["value"]

        if incident_id not in incident_data:
            incident_data[incident_id] = {
                'olr_code': olr_code,
                'original_id': origianl_id,
                'start_time': convert_time(start_time),
                'end_time': convert_time(end_time),
                'road_closed': road_closed,
                'criticality': criticality,
                'incident_type': incident_type,
                'summary': summary
            }


def convert_time(time_string):
    year_month_day = time_string.split('T')[0]
    hour_min_sec = time_string.split('T')[1][:-1]
    return year_month_day + " " + hour_min_sec


def delete_unresolved_olr_entries():
    query = """DELETE FROM incident_data
                WHERE incident_data.olr_code NOT IN (
                    SELECT DISTINCT olr_eid_mapping.olr_code from olr_eid_mapping) 
                """
    cursor = connection.cursor()
    cursor.execute(query)

    query = """DELETE FROM olr_codes
                    WHERE olr_codes.olr_code NOT IN (
                        SELECT DISTINCT olr_eid_mapping.olr_code from olr_eid_mapping) 
                    """
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--insert_flow', action='store_true')
    parser.add_argument('--insert_incidents', action='store_true')
    args = parser.parse_args()

    if args.insert_flow:
        create_flow_table()
        flow_path = path_to_zip_files + "flow"
        files = [f for f in os.listdir(flow_path)]
        files.sort()
        for file in files:
            if file.endswith(".zip"):
                date = file.split(".")[0]
                if date_already_in_database(date):
                    continue

                get_frequent_olr_codes()
                parse_zip(flow_path + "/" + file, True)
                print(f"Finished analyzing {file}, starting insertion")
                insert_flow()
                print("finished insertion")

    if args.insert_incidents:
        incidents_path = path_to_zip_files + "incidents"
        create_incidents_table()
        files = [f for f in os.listdir(incidents_path)]
        files.sort()
        for file in files:
            if file.endswith(".zip"):
                reset_incident_data()
                parse_zip(incidents_path + "/" + file, False)
                print(f"Finished analyzing {file}")
                insert_incidents()
    delete_unresolved_olr_entries()

    print("done")
