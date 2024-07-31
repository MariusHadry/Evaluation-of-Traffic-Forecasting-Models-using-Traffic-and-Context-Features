import os
import json
import zipfile
import psycopg2
from tqdm import tqdm

olr_base: None | dict = None
file_count: int = 0
path_to_zip_files: str = "data/"
connection = psycopg2.connect(host="", port="",
                              user="", password="",
                              database="lowerFranconia")


def extract_olr(data: str) -> dict:
    parsed = json.loads(data)
    ret = {}
    for res in parsed["results"]:
        # every olr code should only appear once in a single file. Thus, the value is set to one.
        ret[res["location"]["olr"]] = 1

    return ret


def update_dict(to_check):
    global olr_base
    for b in to_check:
        if b in olr_base:
            olr_base[b] += 1
        else:
            olr_base[b] = 1


def analyze_zip(input_zip):
    global olr_base, file_count

    with zipfile.ZipFile(input_zip) as input_zip:
        for name in input_zip.namelist():
            if name.endswith(".txt"):
                with input_zip.open(name) as file:
                    data = file.read().decode("UTF-8")

                    # check file size and skip empty files
                    if len(data) == 0:
                        continue

                    file_count += 1

                    if olr_base is None:
                        olr_base = extract_olr(data)
                    else:
                        tmp_olr = extract_olr(data)
                        update_dict(tmp_olr)


def write_olr_base_to_db(data_source):
    global olr_base

    if olr_base is None:
        print("olr_base was empty, aborting!")
        return

    # create cursor
    cursor = connection.cursor()
    query = """CREATE TABLE IF NOT EXISTS public.olr_codes (
                  id serial not null,
                  olr_code varchar(1024) NOT NULL,
                  frequency integer NOT NULL,
                  datasource VARCHAR(10) NOT NULL,
                  geometry geometry(MultiLineString,4326),
                  PRIMARY KEY (id)
                )"""
    cursor.execute(query)
    connection.commit()

    query = """INSERT INTO public.olr_codes (
                olr_code, frequency, datasource)
                VALUES (%s, %s, %s);"""

    # insert dictionary entries into database
    for key in olr_base:
        cursor.execute(query, (key, olr_base[key], data_source))

    connection.commit()


def create_olr_eid_mapping_table():
    # This table is usually already created at this time as it is automatically created with the database-creation
    # script. In case the table was deleted it is recreated here.

    cursor = connection.cursor()
    query = """CREATE TABLE IF NOT EXISTS public.olr_eid_mapping (
                    olr_code character varying(2048) NOT NULL,
                    edge_id bigint NOT NULL,
                    starting_edge boolean,
                    ending_edge boolean,
                    index_in_path integer,
                    PRIMARY KEY (olr_code, edge_id)
                )"""
    cursor.execute(query)
    connection.commit()
    print("olr eid mapping table created")


if __name__ == '__main__':
    print("==========================")
    print("=== processing flow files ===")
    print("==========================\n")

    files = [f for f in os.listdir(path_to_zip_files + "flow")]
    for file in tqdm(files):
        if file.endswith(".zip"):
            analyze_zip(path_to_zip_files + "flow/" + file)
    write_olr_base_to_db("flow")

    print("==========================")
    print("=== processing incidents files ===")
    print("==========================\n")

    olr_base = None
    files = [f for f in os.listdir(path_to_zip_files + "incidents")]
    for file in tqdm(files):
        if file.endswith(".zip"):
            analyze_zip(path_to_zip_files + "incidents/" + file)
    write_olr_base_to_db("incidents")

    create_olr_eid_mapping_table()

    print(f"{file_count} Files were analyzed")
