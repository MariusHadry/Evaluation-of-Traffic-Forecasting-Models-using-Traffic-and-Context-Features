import json
import os
import zipfile
import psycopg2

"""
Extracts flow information of frequent edges and saves this data into a .csv-file.

This method is currently not advised, as everything is written into a database instead of .csv-files.
"""

flow_data = {}
path_to_zip_files = "data/flow/"
path_to_write_destination = "output/"

# Connect to an existing database
connection = psycopg2.connect(host="", port="",
                              user="", password="", database="lowerFranconia")


def get_frequent_olr_codes():
    # create cursor
    cursor = connection.cursor()

    # selection for frequent olr_codes
    # TODO: adjust frequency depending on data!
    query = """SELECT olr_code FROM found_olr_codes
                WHERE frequency >= 100"""

    cursor.execute(query)

    result = cursor.fetchall()
    for row in result:
        flow_data[row[0]] = []


def analyze_zip(input_zip):
    with zipfile.ZipFile(input_zip) as input_zip:
        for name in input_zip.namelist():
            if name.endswith(".txt"):
                with input_zip.open(name) as file:
                    data = file.read().decode("UTF-8")

                    # check file size and skip empty files
                    if len(data) == 0:
                        continue

                    # extract information from files
                    extract_information(data)


def extract_information(data):
    parsed = json.loads(data)
    time = parsed["sourceUpdated"]
    for entry in parsed["results"]:
        olr = entry["location"]["olr"]
        if olr in flow_data:
            to_add = {
                'time': time,
                'olr': olr,
                'speed': entry["currentFlow"]["speed"] if "speed" in entry["currentFlow"] else -1,
                'speedUncapped': entry["currentFlow"]["speedUncapped"] if "speedUncapped" in entry["currentFlow"] else -1,
                'freeFlow': entry["currentFlow"]["freeFlow"] if "freeFlow" in entry["currentFlow"] else -1,
                'jamFactor': entry["currentFlow"]["jamFactor"] if "jamFactor" in entry["currentFlow"] else -1,
                'confidence': entry["currentFlow"]["confidence"] if "confidence" in entry["currentFlow"] else -1,
                'traversability': entry["currentFlow"]["traversability"] if "traversability" in entry["currentFlow"] else "unknown"
            }
            flow_data[olr].append(to_add)


def get_as_csv(data):
    csv_string = "Time;OLR Code;Speed;Speed Uncapped;Free Flow;Jam Factor;Confidence;Traversability\n"

    for e in data:
        csv_string += e["time"] + ";"
        csv_string += e["olr"] + ";"
        csv_string += str(e["speed"]) + ";"
        csv_string += str(e["speedUncapped"]) + ";"
        csv_string += str(e["freeFlow"]) + ";"
        csv_string += str(e["jamFactor"]) + ";"
        csv_string += str(e["confidence"]) + ";"
        csv_string += str(e["traversability"]) + ";"
        csv_string += "\n"

    return csv_string


if __name__ == '__main__':
    get_frequent_olr_codes()

    files = [f for f in os.listdir(path_to_zip_files)]
    files.sort()
    for file in files:
        if file.endswith(".zip"):
            analyze_zip(path_to_zip_files + "/" + file)
            print(f"Finished analyzing {file}")

    counter = 0
    for olr in flow_data.keys():
        # content = json.dumps(frequent_olr_codes[olr])
        content = get_as_csv(flow_data[olr])
        counter += 1
        with open(path_to_write_destination + "/" + str(counter) + ".csv", "w") as text_file:
            text_file.write(content)

