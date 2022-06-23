# Notebook setup
import time
from datetime import datetime, date, timedelta
from importlib import resources
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb

from package.flights.etl import get_flights
from package.flights.utils import get_previous_flight

from typing import Union


def get_all_flights() -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Returns all flights within the [t-7, t+7] from public API
    """

    # setting up required parameters
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    date_range = [(today + timedelta(days=t)).strftime('%Y-%m-%d')
                  for t in range(-7, 8)]  # list of date_str to use as query parameters

    departures = pd.DataFrame()
    arrivals = pd.DataFrame()

    for date_str in date_range:
        for cargo in ['true', 'false']:
            if date_str > today_str:
                departure = get_flights(
                    date_string=date_str, arrival='false', cargo=cargo)
                departures = pd.concat([departures, departure])

            arrival = get_flights(date_string=date_str,
                                  arrival='true', cargo=cargo)
            arrivals = pd.concat([arrivals, arrival])

            time.sleep(0.5)

    return departures, arrivals


def map_previous_flights(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """
    For each departure flight, map all previous arrival flights to a df and return it
    """
    previous_flights = departures.apply(
        lambda x: get_previous_flight(x, arrivals), axis=1)
    previous_flights.columns = ['scheduled_arrival',
                                'actual_arrival', 'arrival_flight_num']

    df = pd.concat([departures, previous_flights], axis=1)

    return df


def set_up_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Series of cleanup procedures done to fit the inference input format
    """

    # adding route_options
    df['scheduled_departure_date'] = df['scheduled_departure'].dt.date

    route_options = df.copy()
    route_flight_options = route_options[['scheduled_departure_date', 'destination', 'flight_num']].groupby(
        by=['scheduled_departure_date', 'destination']).nunique().reset_index()
    route_airline_options = route_options[['scheduled_departure_date', 'destination', 'airline']].groupby(
        by=['scheduled_departure_date', 'destination']).nunique().reset_index()

    route_flight_options.columns = [
        'scheduled_departure_date', 'destination', 'route_flight_options']
    route_airline_options.columns = [
        'scheduled_departure_date', 'destination', 'route_airline_options']

    df = df.merge(route_flight_options, how='left', on=[
                  'scheduled_departure_date', 'destination'])
    df = df.merge(route_airline_options, how='left', on=[
                  'scheduled_departure_date', 'destination'])

    # adding date features as columns
    df['prev_arr_hour'] = df['scheduled_arrival'].dt.hour
    df['prev_arr_weekday'] = df['scheduled_arrival'].dt.weekday + 1
    df['dep_hour'] = df['scheduled_departure'].dt.hour
    df['dep_weekday'] = df['scheduled_departure'].dt.weekday + 1

    # calculating turnaround and arrival_delay
    df['scheduled_turnaround'] = (
        df['scheduled_departure'] - df['scheduled_arrival']).dt.total_seconds()/60
    df['arrival_delay'] = (df['actual_arrival'] -
                           df['scheduled_arrival']).dt.total_seconds()/60

    # get Iata code from flight_num
    df['iata'] = df['flight_num'].str[:2]

    # getting fleet_size using iata
    with resources.path("package.dataset", "airlines.csv") as file_path:
        airlines = pd.read_csv(file_path)[['iata', 'aircrafts']]
        airlines.columns = ['iata', 'fleet_size']

    df = df.merge(airlines, how='left', on='iata')

    # convert boolean to binary representation
    df['cargo'] = df['cargo'].astype(int)

    return df


def add_prediction_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split those with actual arrival and those without into different prediction types and return df
    """

    # splitting the data into data with actual arrival and those without arrival_delays (due to no actual_arrival yet)
    actual_df = df.drop(df[df['scheduled_arrival'].notnull()
                           & df['actual_arrival'].isnull()].index)
    tentative_df = df[df['scheduled_arrival'].notnull() &
                      df['actual_arrival'].isnull()]

    # add in prediction types
    actual_df['prediction_type'] = 'Actual'
    tentative_df['prediction_type'] = 'Tentative'

    # for tentative df, we will use [0, 45, 90, 180] as possible arrival_delay values
    tentative_arrival_delays = pd.DataFrame(
        [0, 45, 90, 180], columns=['tentative_arrival_delay'])
    tentative_df = tentative_df.merge(tentative_arrival_delays, how='cross')
    tentative_df['arrival_delay'] = tentative_df['tentative_arrival_delay']
    tentative_df['actual_arrival'] = tentative_df['scheduled_arrival'] + \
        pd.to_timedelta(tentative_df['arrival_delay'], unit='m')

    # re-merge the actual and tentative dataframes to form df
    return pd.concat([actual_df, tentative_df])


def split_input_for_prediction(df: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Split df into the input for predictions and an info df for output
    """
    # reorder to follow the exact format of training data
    X = df[['prev_arr_hour', 'prev_arr_weekday', 'dep_hour', 'dep_weekday',
            'scheduled_turnaround', 'arrival_delay', 'destination', 'iata',
           'fleet_size', 'cargo', 'route_flight_options', 'route_airline_options']].reset_index().drop(columns='index')

    # for results
    X_info = df[['flight_num', 'scheduled_departure', 'arrival_flight_num', 'scheduled_arrival', 'actual_arrival', 'arrival_delay',
                 'origin', 'destination', 'iata', 'cargo', 'prediction_type']].reset_index().drop(columns='index')

    return X, X_info


def get_predictions(X: pd.DataFrame, X_info: pd.DataFrame, model_names: list) -> pd.DataFrame:
    """
    Takes input X and returns a dataframe with predictions, ready to be inserted into BigQuery
    """

    # Loading training assets to assist with training
    with resources.path("package.models", model_names[0]) as iata_path:
        with open(iata_path, "rb") as f:
            iata_encoder = pickle.load(f)

    with resources.path("package.models", model_names[1]) as dest_path:
        with open(dest_path, "rb") as f:
            dest_encoder = pickle.load(f)

    with resources.path("package.models", model_names[2]) as model_path:
        with open(model_path, "rb") as f:
            bst = pickle.load(f)

    threshold = float('0.' + str(model_names[2].split('_')[-1].split('.')[0]))

    # get encoded columns
    iata_OHE = pd.DataFrame(iata_encoder.transform(
        np.array(X['iata']).reshape(-1, 1)))
    iata_OHE.columns = [f'iata_{x}' for x in iata_OHE.columns]

    dest_OHE = pd.DataFrame(dest_encoder.transform(
        np.array(X['destination']).reshape(-1, 1)))
    dest_OHE.columns = [f'dest_{x}' for x in dest_OHE.columns]

    # X in the right format for inference
    X = pd.concat([X, iata_OHE, dest_OHE], axis=1
                  ).drop(columns=['iata', 'destination']).fillna(np.nan)

    d_matrix = xgb.DMatrix(X)

    preds = bst.predict(d_matrix)
    preds_label = [int(pred >= threshold) for pred in preds]

    X_info['prediction_prob'] = preds
    X_info['prediction'] = preds_label
    X_info['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    X_info['model_threshold'] = threshold

    return X_info


def end_to_end_pipeline(model_paths: list) -> pd.DataFrame:
    """
    Combines all functions for an end to end pipeline:
    1. Gets latest flights from API
    2. Clean the data into the required format
    3. Use the model from model_paths to run predictions
    4. Return an output df
    """

    print('Getting flights from API...', end='\r')
    departures, arrivals = get_all_flights()
    print('Getting flights from API... Done.')

    print('Mapping previous flights...', end='\r')
    df = map_previous_flights(departures, arrivals)
    print('Mapping previous flights... Done.')

    print('Setting up columns...', end='\r')
    df = set_up_columns(df)
    print('Setting up columns... Done.')

    print('Adding prediction types...', end='\r')
    df = add_prediction_types(df)
    print('Adding prediction types... Done.')

    print('Splitting data for prediction...', end='\r')
    X, X_info = split_input_for_prediction(df)
    print('Splitting data for prediction... Done.')

    print('Running Predictions...', end='\r')
    output = get_predictions(X, X_info, model_paths)
    print('Running Predictions... Done.')

    print('Complete!')

    return output
