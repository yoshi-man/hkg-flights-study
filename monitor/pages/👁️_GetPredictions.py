from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px

st.set_page_config(
    page_title="HKG Flight Predictions - Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials,
                         project=credentials.project_id,)


@st.experimental_memo(ttl=43200)
def get_departures() -> pd.DataFrame:

    query = f"""
    SELECT * FROM flights.departures
    WHERE DATE(scheduled_departure) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    ORDER BY scheduled_departure DESC
    """

    df = client.query(query).to_dataframe()

    df['flight_num'] = df['flight_num'].str.replace(' ', '')

    return df


@st.experimental_memo(ttl=43200)
def get_arrivals() -> pd.DataFrame:

    query = f"""
    SELECT * FROM flights.arrivals
    WHERE DATE(scheduled_arrival) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    ORDER BY scheduled_arrival DESC
    """

    df = client.query(query).to_dataframe()

    df['flight_num'] = df['flight_num'].str.replace(' ', '')

    return df


@st.experimental_memo(ttl=43200)
def get_predictions() -> pd.DataFrame:

    query = f"""
    SELECT * FROM flights.predictions
    WHERE DATE(scheduled_departure) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    ORDER BY scheduled_departure
    """

    df = client.query(query).to_dataframe()
    df['flight_num'] = df['flight_num'].str.replace(' ', '')
    df['arrival_flight_num'] = df['arrival_flight_num'].str.replace(' ', '')

    return df


departures = get_departures()
arrivals = get_arrivals()
predictions = get_predictions()

# Initialization
url_params = st.experimental_get_query_params()

latest_predictions = predictions[predictions['prediction_date']
                                 == predictions['prediction_date'].max()]

unique_airlines = list(
    sorted(latest_predictions['flight_num'].str.slice(0, 2).unique()))


if url_params.get("airline") is None:
    st.session_state['airline'] = 'CX'
    st.experimental_set_query_params(airline='CX')

elif url_params.get("airline")[0] not in unique_airlines:
    st.session_state['airline'] = 'CX'
    st.experimental_set_query_params(airline='CX')
else:
    st.session_state['airline'] = url_params.get("airline")[0]

airline_index = unique_airlines.index(st.session_state['airline'])

st.title('Flight Predictions per Airline')

selected_airline = st.selectbox(label='Select Airline (default=CX)',
                                options=unique_airlines, index=airline_index)

st.session_state['airline'] = selected_airline
st.experimental_set_query_params(airline=selected_airline)


@st.experimental_memo
def get_past_flights(departures: pd.DataFrame, num_weeks: int):
    past_weeks = (datetime.today() +
                  timedelta(days=-7*num_weeks)).strftime('%Y-%m-%d')

    return departures[departures['scheduled_departure'] >= past_weeks]


airline_predictions = predictions[predictions['flight_num'].str.slice(
    0, 2) == selected_airline]

airline_departures = departures[departures['flight_num'].str.slice(
    0, 2) == selected_airline]

airline_departures_past_week = get_past_flights(airline_departures, 1)

airline_arrivals = arrivals[arrivals['flight_num'].str.slice(
    0, 2) == selected_airline]


# Now that the airline has been selected, we can show the information accordingly
def get_past_delay_percentage(departures: pd.DataFrame) -> float:
    delay_series = (departures['actual_departure'] -
                    departures['scheduled_departure']).dt.total_seconds()/60
    delay_bools = delay_series.apply(lambda x: 1 if x > 15 else 0).to_numpy()

    return np.round(np.sum(delay_bools)/delay_bools.size*100, 0)


def get_past_average_delay(departures: pd.DataFrame) -> float:
    delay_series = ((departures['actual_departure'] -
                    departures['scheduled_departure']).dt.total_seconds()/60).to_numpy()

    return np.round(np.nanmean(delay_series), 0)


def get_predicted_delay_percentage(predictions: pd.DataFrame) -> float:
    max_pred_date = predictions['prediction_date'].max()

    latest_preds = predictions[predictions['prediction_date']
                               == max_pred_date]['prediction'].to_numpy()

    return np.round(np.sum(latest_preds)/latest_preds.size*100, 0)


st.header('This coming week, ')
col1, col2, col3 = st.columns(3)
with col1:
    latest_predictions = airline_predictions[airline_predictions['prediction_date']
                                             == airline_predictions['prediction_date'].max()]
    st.metric(
        label='Departures',
        value=f"{latest_predictions.groupby(['scheduled_departure', 'flight_num']).ngroups}",
        delta=f"{latest_predictions.groupby(['scheduled_departure', 'flight_num']).ngroups - airline_departures_past_week.shape[0]} flights"
    )


with col2:
    st.metric(
        label='Predicted Delay %',
        value=f"{get_predicted_delay_percentage(latest_predictions)}%",
        delta=f"{get_predicted_delay_percentage(latest_predictions) - get_past_delay_percentage(airline_departures_past_week)}%",
        delta_color="inverse"
    )

with col3:
    st.caption(
        'Predictions were only made based on a binary classification of delay or on-time.')


# show table of both past departures and arrivals
edited_departures = airline_departures.copy()
edited_departures['actual_delayed_minutes'] = (airline_departures['actual_departure'] -
                                               airline_departures['scheduled_departure']).dt.total_seconds()/60

edited_departures['actual_delayed'] = edited_departures['actual_delayed_minutes'].apply(
    lambda x: 1 if x > 15 else 0).astype(int)


merged_predictions = predictions.merge(edited_departures[['scheduled_departure', 'flight_num', 'actual_departure', 'actual_delayed']], how='left', left_on=[
                                       'scheduled_departure', 'flight_num'], right_on=['scheduled_departure', 'flight_num'])

prediction_performance = merged_predictions[merged_predictions['actual_delayed'].notnull(
)]

latest_prediction_date = prediction_performance['prediction_date'].max()
the_week_before = (latest_prediction_date -
                   timedelta(days=7)).strftime('%Y-%m-%d')


#####################

latest_predictions = airline_predictions[airline_predictions['prediction_date']
                                         == airline_predictions['prediction_date'].max()]

display_predictions = latest_predictions.copy()

display_predictions['scheduled_departure'] = display_predictions['scheduled_departure'].dt.strftime(
    '%Y-%m-%d %H:%M')

display_predictions['flight_type'] = display_predictions['cargo'].apply(
    lambda x: 'Freighter' if x else 'Passenger')

display_predictions = display_predictions[['flight_type',
                                           'flight_num', 'destination', 'scheduled_departure', 'prediction', 'prediction_prob']].sort_values(
    by=['scheduled_departure'], ascending=True, ignore_index=True
)


st.header(
    f'Latest Predictions ({latest_prediction_date.strftime("%Y-%m-%d")})')

col1, col2 = st.columns(2)

with col1:
    st.write(display_predictions)

with col2:
    predictions_chart = display_predictions.copy()

    predictions_chart = predictions_chart[['destination', 'prediction', 'flight_num']].groupby(
        by=['destination', 'prediction']).count().reset_index()

    predictions_chart['prediction'] = predictions_chart['prediction'].apply(
        lambda x: 'On-Time' if x == 0 else 'Delayed')

    predictions_chart = predictions_chart.pivot(index='destination',
                                                columns='prediction', values='flight_num').fillna(0).reset_index()

    if 'Delayed' in predictions_chart and 'On-Time' in predictions_chart:
        predictions_chart['total'] = predictions_chart['Delayed'] + \
            predictions_chart['On-Time']

    elif 'Delayed' in predictions_chart and 'On-Time' not in predictions_chart:
        predictions_chart['On-Time'] = 0
        predictions_chart['total'] = predictions_chart['Delayed']

    elif 'Delayed' not in predictions_chart and 'On-Time' in predictions_chart:
        predictions_chart['Delayed'] = 0
        predictions_chart['total'] = predictions_chart['On-Time']

    else:
        predictions_chart['On-Time'] = 0
        predictions_chart['Delayed'] = 0
        predictions_chart['total'] = 0

    predictions_chart = predictions_chart.sort_values(
        by='total', ascending=False)
    predictions_chart.drop(columns='total', inplace=True)

    fig = px.bar(predictions_chart, y=['On-Time', 'Delayed'],
                 x='destination', labels={'value': 'Number of Departures', 'variable': 'Prediction', 'destination': 'Dest'}, width=700).update_layout(xaxis=dict(showgrid=False),
                                                                                                                                                      yaxis=dict(
                     showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)

#######################

latest_predictions = prediction_performance[prediction_performance['prediction_date']
                                            == prediction_performance['prediction_date'].max()]

week_before_predictions = prediction_performance[prediction_performance['prediction_date'].dt.strftime('%Y-%m-%d')
                                                 == the_week_before]

display_predictions = week_before_predictions.copy()

display_predictions['scheduled_departure'] = display_predictions['scheduled_departure'].dt.strftime(
    '%Y-%m-%d %H:%M')

display_predictions['actual_departure'] = display_predictions['actual_departure'].dt.strftime(
    '%Y-%m-%d %H:%M')

display_predictions['prediction_date'] = display_predictions['prediction_date'].dt.strftime(
    '%Y-%m-%d')

display_predictions['actual_delayed'] = display_predictions['actual_delayed'].astype(
    int)

display_predictions['flight_type'] = display_predictions['cargo'].apply(
    lambda x: 'Freighter' if x else 'Passenger')

display_predictions = display_predictions[['flight_type',
                                           'flight_num', 'destination', 'scheduled_departure', 'actual_departure', 'actual_delayed', 'prediction']].sort_values(
    by=['scheduled_departure'], ascending=False, ignore_index=True
)


st.header(f'Predictions Made Last Week ({the_week_before})')

col1, col2 = st.columns(2)

correct_predictions = display_predictions[display_predictions['actual_delayed']
                                          == display_predictions['prediction']]
wrong_predictions = display_predictions[display_predictions['actual_delayed']
                                        != display_predictions['prediction']]

with col1:
    st.subheader('Correct Predictions')
    st.write(f"Correct Count: {correct_predictions.shape[0]}")
    st.write(correct_predictions, use_container_width=True)

with col2:
    st.subheader('Incorrect Predictions')
    st.write(f"Incorrect Count: {wrong_predictions.shape[0]}")
    st.write(wrong_predictions, use_container_width=True)
