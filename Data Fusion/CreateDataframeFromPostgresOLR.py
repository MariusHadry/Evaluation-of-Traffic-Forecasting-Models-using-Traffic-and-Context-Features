"""
Steps to use this script:

* ensure that olr_codes, resolved olr codes, weather data, and flow data is inserted into the database
* check if olr_codes have geometries assigned. If this is not the case execute 'SQL Scripts/add-geometry-olr-codes.sql'
* assign speed_percentages to flow_data table (or use new version of insertion script)

"""
import argparse

import pandas as pd
import datetime
import os

import psycopg2
from timeit import default_timer as timer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy connectable.*")

postgres_connection = psycopg2.connect(host="", port="",
                                       user="postgres", password="",
                                       database="lowerFranconia")

df_flow: None | pd.DataFrame = None
incident_state: None | dict = None
weather_state: None | dict = None


def load_flow_with_edge_data(date):
    global df_flow

    query = f"""SELECT flow_data.olr_code, "timestamp", MAX(fow) AS max_fow, MAX(frc) AS max_frc,
                    MIN(fow) AS min_fow, MIN(frc) AS min_frc, MAX(tags->'maxspeed') AS "maxspeed",
                    AVG(freeflow) AS free_flow, AVG(speed_percentage) AS speed_percentage
                FROM openlr_edges, planet_osm_line, flow_data, olr_eid_mapping
                WHERE "timestamp" BETWEEN '{date}' AND '{date + datetime.timedelta(days=1)}' AND
                        openlr_edges.osm_id = planet_osm_line.osm_id AND 
                        olr_eid_mapping.olr_code = flow_data.olr_code AND
                        olr_eid_mapping.edge_id = line_id
                        AND lanes is not null
                GROUP BY flow_data.olr_code, timestamp"""

    df_flow = pd.read_sql(query, postgres_connection)


def merge_incidents(date, prints=False):
    global df_flow

    df_flow['has_incident'] = False
    df_flow['inc_type'] = None
    df_flow['inc_criticality'] = None
    df_flow['inc_road_closed'] = False

    # retrieve incidents for the current day
    query = f"""WITH d as (
                SELECT edge_id, start_time, end_time, "type", criticality, road_closed, incident_id
                FROM incident_data i, olr_eid_mapping oem
                WHERE start_time < '{date + datetime.timedelta(days=1)}' 
                       AND end_time >= '{date}' AND i.olr_code = oem.olr_code)
                
            SELECT olr.olr_code, MAX(start_time), MAX(end_time), MAX("type"), MAX(criticality),
                    bool_or(road_closed), incident_id
            FROM d, olr_eid_mapping oem, olr_codes olr
            WHERE olr.olr_code =oem.olr_code AND olr.datasource = 'flow' AND olr.frequency >= 112457
                AND d.edge_id = oem.edge_id
            GROUP BY incident_id, olr.olr_code"""

    _cursor = postgres_connection.cursor()
    _cursor.execute(query)

    batch_size = 5000
    rows = _cursor.fetchmany(size=batch_size)
    counter = 0

    while rows:
        counter += len(rows)
        process_incident_rows(rows)
        rows = _cursor.fetchmany(size=batch_size)
        if prints:
            print(f"\tincident batch done: {counter}")

    # write last incident, as there is no change afterward
    write_incident_to_df()


def process_incident_rows(rows):
    global incident_state

    for row in rows:
        if incident_state is None:
            incident_state = {
                'id': row[6],
                'inc_type': row[3],
                'inc_criticality': row[4],
                'inc_road_closed': row[5],
                'start_time': row[1],
                'end_time': row[2],
                'olr_codes': []
            }

        elif incident_state['id'] != row[6]:
            write_incident_to_df()
            incident_state = {
                'id': row[6],
                'inc_type': row[3],
                'inc_criticality': row[4],
                'inc_road_closed': row[5],
                'start_time': row[1],
                'end_time': row[2],
                'olr_codes': []
            }

        # always append edge
        incident_state['olr_codes'].append(row[0])


def write_incident_to_df():
    filter_mask = (df_flow['olr_code'].isin(incident_state['olr_codes'])) & \
                  (df_flow['timestamp'] >= incident_state['start_time']) & \
                  (df_flow['timestamp'] <= incident_state['end_time'])

    df_flow.loc[filter_mask, 'has_incident'] = True
    df_flow.loc[filter_mask, 'inc_type'] = incident_state['inc_type']
    df_flow.loc[filter_mask, 'inc_criticality'] = incident_state['inc_criticality']
    df_flow.loc[filter_mask, 'inc_road_closed'] = incident_state['inc_road_closed']


def merge_weather(date, prints=False):
    global df_flow

    df_flow['w_type_0'] = 0
    df_flow['w_type_1'] = 0
    df_flow['w_type_2'] = 0
    df_flow['w_type_3'] = 0
    df_flow['w_type_4'] = 0
    df_flow['w_type_5'] = 0
    df_flow['w_type_6'] = 0

    query = f"""SELECT w.id, w.timestamp, olr_code, "type", "level"
                FROM weather_data w, olr_codes o
                WHERE w.timestamp >= '{date}' AND w.timestamp < '{date + datetime.timedelta(days=1)}' 
                        AND ST_Intersects(w.simple_geometry, o.geometry)
                ORDER BY w.id"""

    _cursor = postgres_connection.cursor()
    _cursor.execute(query)

    batch_size = 50000
    rows = _cursor.fetchmany(size=batch_size)
    counter = 0

    while rows:
        counter += len(rows)
        process_weather_rows(rows)
        rows = _cursor.fetchmany(size=batch_size)
        if prints:
            print(f"\tweather batch done: {counter}")

    # write last weather object, as there is no change afterward
    write_weather_to_df()


def process_weather_rows(rows):
    global weather_state

    for row in rows:
        if weather_state is None:
            weather_state = {
                'id': row[0],
                'time_start': row[1],
                'time_end': row[1] + datetime.timedelta(hours=1),
                'w_type': row[3],
                'w_level': row[4],
                'olr_codes': []
            }
        elif weather_state['id'] != row[0]:
            write_weather_to_df()
            weather_state = {
                'id': row[0],
                'time_start': row[1],
                'time_end': row[1] + datetime.timedelta(hours=1),
                'w_type': row[3],
                'w_level': row[4],
                'olr_codes': []
            }

        # always append edge
        weather_state['olr_codes'].append(row[2])


def write_weather_to_df():
    filter_mask = (df_flow['timestamp'] >= weather_state['time_start']) & \
                  (df_flow['timestamp'] < weather_state['time_end']) & \
                  (df_flow['olr_code'].isin(weather_state['olr_codes']))

    df_flow.loc[filter_mask, 'w_type_' + str(weather_state['w_type'])] = weather_state['w_level']


def str2date(date_str):
    date_format = '%d-%m-%Y'
    return datetime.datetime.strptime(date_str, date_format)


def consider_bavarian_holidays_2022(date):
    global df_flow

    date_format = '%d-%m-%Y'
    date_string = date.strftime(date_format)

    df_flow['holiday_or_vacation'] = False

    # single holidays
    if date_string in ['15-04-2022', '18-04-2022', '01-05-2022', '26-05-2022', '06-06-2022']:
        df_flow['holiday_or_vacation'] = True

    # school vacation
    start_end_pairs = [
        [str2date('28-02-2022'), str2date('04-03-2022')],  # Winter vacation
        [str2date('11-04-2022'), str2date('23-04-2022')],  # Easter vacation
        [str2date('07-06-2022'), str2date('18-06-2022')],  # Pentecost vacation
                       ]

    for p in start_end_pairs:
        if p[0] <= date <= p[1]:
            df_flow['holiday_or_vacation'] = True


def preprocess_dataframe(dataframe):
    numeric_only = dataframe.copy()

    if 'has_incident' in numeric_only.columns:
        numeric_only = dataframe.drop(['has_incident'], axis=1)

    if 'inc_road_closed' in numeric_only.columns:
        numeric_only['inc_road_closed'] = numeric_only['inc_road_closed'].astype(int)

    if 'inc_criticality' in numeric_only.columns:
        numeric_only['inc_criticality'] = numeric_only['inc_criticality'].map(
            {'low': 1, 'minor': 2, 'major': 3, 'critical': 4})

    if 'maxspeed' in numeric_only.columns:
        # maxspeed data
        numeric_only = numeric_only.replace({'maxspeed': 'DE:urban'}, '50')
        numeric_only = numeric_only.replace({'maxspeed': 'none'}, '300')

        # signals: most likely variable due to adaptive road signs, so we set it to 0 which equals unknown
        numeric_only = numeric_only.replace({'maxspeed': 'signals'}, '0')
        numeric_only = numeric_only.replace({'maxspeed': '70; 50'}, '70')

        numeric_only['maxspeed'] = pd.to_numeric(numeric_only['maxspeed'])

        #       remove N/As by replacing them with zeros
        numeric_only['maxspeed'] = numeric_only['maxspeed'].fillna(0)

    if 'free_flow' in numeric_only:
        numeric_only['free_flow'] = numeric_only['free_flow'] * 3.6     # m/s -> km/h
        numeric_only['free_flow'] = numeric_only['free_flow'].round(1)

    return numeric_only


def process_single_day(date, prints=False, use_incidents=False, use_weather=False, use_holidays=False):
    global df_flow

    if prints:
        start_time = timer()
        print("executing")

    load_flow_with_edge_data(date)

    if prints:
        print("\tflow with edges loaded")

    if use_incidents:
        merge_incidents(date)

        if prints:
            print("\tmerged incidents")
            end_time = timer()
            print(f"time for flow/incident processing: {end_time - start_time}")

    if use_weather:
        if prints:
            start_time = timer()
            print("merging weather..")

        merge_weather(date)

        if prints:
            end_time = timer()
            print(f"time for weather processing: {end_time - start_time}")

    if use_holidays:
        consider_bavarian_holidays_2022(date)

    df_flow = preprocess_dataframe(df_flow)

    os.makedirs("data/", exist_ok=True)
    df_flow.to_csv(f"data/{date.strftime('%Y-%m-%d')}.csv", sep=";", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load traffic data from database and save to *.csv files for '
                                                 'each day.')

    # expected as '%Y-%m-%d'
    parser.add_argument('--start_date', action='store', help="first date as '%Y-%m-%d'")
    parser.add_argument('--end_date', action='store', help="last date as '%Y-%m-%d'")
    args = parser.parse_args()

    cursor = postgres_connection.cursor()
    start_date, end_date = None, None

    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        cursor.execute("SELECT \"timestamp\" FROM public.flow_data ORDER BY timestamp ASC LIMIT 1")
        start_date = cursor.fetchone()[0]

    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        cursor.execute("SELECT \"timestamp\" FROM public.flow_data ORDER BY timestamp DESC LIMIT 1")
        end_date = cursor.fetchone()[0]

    current_date = start_date

    # while current_date <= end_date:
    for _ in tqdm(range((end_date - start_date).days + 1)):
        process_single_day(current_date)
        current_date = current_date + datetime.timedelta(days=1)
