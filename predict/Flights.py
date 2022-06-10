import requests
from datetime import datetime, timedelta
import pandas as pd


def get_API_url(date_string: str, arrival: str, cargo: str) -> str:
    """
    Given date, arrival, cargo, return the API URL endpoint
    """
    return f'https://www.hongkongairport.com/flightinfo-rest/rest/flights/past?date={date_string}&arrival={arrival}&lang=en&cargo={cargo}'


def request_flights_data(API_url: str) -> list:
    """
    Request from API_url, if unsuccessful response, return an empty list.
    """
    request = requests.get(API_url)

    if request.status_code == 200:
        data = request.json()
    else:
        data = []
        print(f'Invalid Response from API: {request.status_code}')

    return data


def normalize_flights_data(flights_data: list) -> pd.DataFrame:
    """
    Takes the nested list of JSON objects and flattens it into a dataframe.
    """
    normalized_df = pd.json_normalize(flights_data, "list",
                                      ["date", "arrival", "cargo"],
                                      errors='ignore', record_prefix='')

    return normalized_df


def extract_transform_flights(normalized_df: pd.DataFrame, arrival: str) -> pd.DataFrame:
    """
    Takes the needed columns from the normalized df and transforms them according to
    the type of flights being extracted: arrival or departure.

    If arrival flights:
    - Take the last flight before arriving to HKG as flight
    - Take the last origin before arriving to HKG as origin
    - Use HKG for destination 

    If departure flights:
    - Take the first flight after departing from HKG as flight
    - Take the first destination after departing from HKG as destination
    - Use HKG for origin 
    """

    df = pd.DataFrame()

    # Adjustments needed to be made depending on the query parameter arrival
    scheduled_field_name = 'scheduled_arrival' if arrival == 'true' else 'scheduled_departure'
    actual_field_name = 'actual_arrival' if arrival == 'true' else 'actual_departure'

    df[scheduled_field_name] = pd.to_datetime(normalized_df['date']+' '+normalized_df['time'],
                                              infer_datetime_format=True)

    # clean away all alphabetical characters in status, this is the actual timestamp
    df['date'] = normalized_df['date']
    df['status'] = normalized_df['status'].str.replace(
        r"[a-zA-Z]", '').str.strip()

    df['flight_num'] = normalized_df['flight'].apply(
        lambda x: x[-1].get("no")) if arrival == 'true' else normalized_df['flight'].apply(lambda x: x[0].get("no"))

    df['origin'] = normalized_df['origin'].apply(
        lambda x: x[-1]) if arrival == 'true' else 'HKG'
    df['destination'] = 'HKG' if arrival == 'true' else normalized_df['destination'].apply(
        lambda x: x[0])

    df['airline'] = normalized_df['flight'].apply(
        lambda x: x[-1].get("airline"))

    df['arrival'] = normalized_df['arrival'].astype(bool)
    df['cargo'] = normalized_df['cargo'].astype(bool)

    # apply the logic, if there is a date then use it, if not then use the date parameter in our query
    df[actual_field_name] = pd.to_datetime(
        df.apply(lambda x: datetime.strptime(x['status'].split(' ')[-1][1:-1], '%d/%m/%Y').strftime('%Y-%m-%d') + ' ' +
                 x['status'].split(' ')[0]
                 if len(x['status'].split(' ')) > 1
                 else x['date'] + ' ' + x['status'], axis=1), infer_datetime_format=True)

    return df[[scheduled_field_name, actual_field_name, 'flight_num',
               'origin', 'destination', 'airline', 'arrival', 'cargo']]


def get_flights(date_string: str, arrival: str, cargo: str) -> pd.DataFrame:
    """
    Return the parsed flights data into a dataframe, encapsulating all subfunctions into one.
    """

    API_url = get_API_url(date_string, arrival, cargo)
    flights_data = request_flights_data(API_url)
    normalized_df = normalize_flights_data(flights_data)
    df = extract_transform_flights(normalized_df, arrival)

    return df
