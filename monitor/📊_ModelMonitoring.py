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
    page_title="HKG Flight Predictions - Streamlit",
    layout="wide",
    initial_sidebar_state="expanded",
)

# url_params = st.experimental_get_query_params()

# Initialization
# if 'airline' not in st.session_state:
#     if url_params.get("airline") is None:
#         st.session_state['airline'] = 'CX'
#     else:
#         st.session_state['airline'] = url_params.get("airline")[0]


# credentials = service_account.Credentials.from_service_account_file(
#     './bq_key.json', scopes=["https://www.googleapis.com/auth/cloud-platform"],
# )

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials,
                         project=credentials.project_id,)


@st.experimental_memo
def get_departures() -> pd.DataFrame:

    query = f"""
    SELECT * FROM flights.departures
    WHERE DATE(scheduled_departure) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    ORDER BY scheduled_departure DESC
    """

    df = client.query(query).to_dataframe()

    df['flight_num'] = df['flight_num'].str.replace(' ', '')

    return df


@st.experimental_memo
def get_arrivals() -> pd.DataFrame:

    query = f"""
    SELECT * FROM flights.arrivals
    WHERE DATE(scheduled_arrival) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    ORDER BY scheduled_arrival DESC
    """

    df = client.query(query).to_dataframe()

    df['flight_num'] = df['flight_num'].str.replace(' ', '')

    return df


@st.experimental_memo
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


@st.experimental_memo
def get_past_flights(departures: pd.DataFrame, num_weeks: int):
    past_weeks = (datetime.today() +
                  timedelta(days=-7*num_weeks)).strftime('%Y-%m-%d')

    return departures[departures['scheduled_departure'] >= past_weeks]


departures = get_departures()
arrivals = get_arrivals()
predictions = get_predictions()
departures_past_week = get_past_flights(departures, 1)
departures_past_2_weeks = get_past_flights(departures, 2)

departures_between_2_to_1_week = departures_past_2_weeks[departures_past_2_weeks['scheduled_departure'] < (
    datetime.today() + timedelta(days=-7)).strftime('%Y-%m-%d')]

today_str = (datetime.today()).strftime('%Y-%m-%d')
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################


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


st.title(f"Basic Information - {today_str}")
st.header('In the past week, ')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label='Departures',
        value=f"{departures_past_week.shape[0]}",
        delta=f"{departures_past_week.shape[0] - departures_between_2_to_1_week.shape[0]} flights"
    )

with col2:
    st.metric(
        label='Delay %',
        value=f"{get_past_delay_percentage(departures_past_week)}%",
        delta=f"{get_past_delay_percentage(departures_past_week) - get_past_delay_percentage(departures_between_2_to_1_week)}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label='Average Delay',
        value=f"{get_past_average_delay(departures_past_week)}",
        delta=f"{get_past_average_delay(departures_past_week) - get_past_average_delay(departures_between_2_to_1_week)} minutes",
        delta_color="inverse"
    )


st.header('This coming week, ')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label='Departures',
        value=f"{predictions.groupby(['scheduled_departure', 'flight_num']).ngroups}",
        delta=f"{predictions.groupby(['scheduled_departure', 'flight_num']).ngroups - departures_past_week.shape[0]} flights"
    )

with col2:
    st.metric(
        label='Predicted Delay %',
        value=f"{get_predicted_delay_percentage(predictions)}%",
        delta=f"{get_predicted_delay_percentage(predictions) - get_past_delay_percentage(departures_past_week)}%",
        delta_color="inverse"
    )

with col3:
    st.caption(
        'Predictions were only made based on a binary classification of delay or on-time.')

##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

st.header("Predictions Overview")

earliest_prediction_date = predictions['prediction_date'].min()
num_preds_past_60 = predictions.shape[0]
preds_per_day = predictions[['prediction', 'prediction_date']].groupby(
    by=['prediction_date']).count().reset_index()

max_pred_date = predictions['prediction_date'].max()

latest_preds = predictions[predictions['prediction_date'] == max_pred_date]

preds_per_airline = predictions[predictions['prediction_date'] == max_pred_date][['iata', 'prediction', 'flight_num']].groupby(
    by=['iata', 'prediction']).count().reset_index()

preds_per_airline['prediction'] = preds_per_airline['prediction'].apply(
    lambda x: 'On-Time' if x == 0 else 'Delayed')

preds_per_airline = preds_per_airline.pivot(index='iata',
                                            columns='prediction', values='flight_num').fillna(0).reset_index()


preds_per_airline['total'] = preds_per_airline['Delayed'] + \
    preds_per_airline['On-Time']
preds_per_airline = preds_per_airline.sort_values(by='total', ascending=False)
preds_per_airline.drop(columns='total', inplace=True)

st.write(f'Since {earliest_prediction_date},')


col1, col2 = st.columns(2)

with col1:

    st.write(
        'Everyday at 9am HKT, a set of predictions are made for flights in the upcoming 7 days')

    st.metric(
        label='Total Number of Predictions',
        value=f"{num_preds_past_60}"
    )

    st.metric(
        label='Total Number of Lates Predicted',
        value=f"{predictions[predictions['prediction']==1].shape[0]}"
    )

    st.metric(
        label='Total Number of On-Time Predicted',
        value=f"{predictions[predictions['prediction']==0].shape[0]}"
    )

    st.metric(
        label='Delay %',
        value=f"{np.round(predictions[predictions['prediction']==1].shape[0]/num_preds_past_60*100, 1)}%"
    )

    st.write('\n\n\n\n')
    st.write('\n\n\n\n')
    st.write('\n\n\n\n')
    st.write('\n\n\n\n')
    st.write('\n\n\n\n')
    # of the largest airlines, what's the ranking
    st.write('Out of the 10 airlines with the most flights coming this week, these are top 5 with the highest delay percentages.')

    top_10_airlines = preds_per_airline.head(10)
    top_10_airlines['delay_percentage'] = np.round(top_10_airlines['Delayed'] /
                                                   (top_10_airlines['On-Time'] + top_10_airlines['Delayed'])*100, 1)

    st.dataframe(top_10_airlines.sort_values(
        by='delay_percentage', ascending=False).reset_index().drop(columns=['index']).head(5))

with col2:
    fig = px.line(preds_per_day, x="prediction_date",
                  y="prediction", labels={'prediction': 'Number of Predictions', 'prediction_date': 'Date of Prediction'}, width=700).update_layout(xaxis=dict(showgrid=False),
                                                                                                                                                    yaxis=dict(
                                                                                                                                                        showgrid=False)
                                                                                                                                                    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(preds_per_airline, y=['On-Time', 'Delayed'],
                 x='iata', labels={'value': 'Number of Departures', 'variable': 'Prediction', 'iata': 'Airline'}, width=700).update_layout(xaxis=dict(showgrid=False),
                                                                                                                                           yaxis=dict(
                                                                                                                                               showgrid=False)
                                                                                                                                           )

    st.plotly_chart(fig, use_container_width=True)

st.title('Model Performance')


edited_departures = departures.copy()
edited_departures['actual_delayed_minutes'] = (departures['actual_departure'] -
                                               departures['scheduled_departure']).dt.total_seconds()/60

edited_departures['actual_delayed'] = edited_departures['actual_delayed_minutes'].apply(
    lambda x: 1 if x > 15 else 0).astype(int)


merged_predictions = predictions.merge(edited_departures[['scheduled_departure', 'flight_num', 'actual_delayed']], how='left', left_on=[
                                       'scheduled_departure', 'flight_num'], right_on=['scheduled_departure', 'flight_num'])

prediction_performance = merged_predictions[merged_predictions['actual_delayed'].notnull(
)]


y_actual = prediction_performance['actual_delayed'].astype(int).to_numpy()
y_pred = prediction_performance['prediction_prob'].astype(float).to_numpy()
y_pred_labels = prediction_performance['prediction'].astype(int).to_numpy()

st.write('An important point to note is that as predictions are made on a rolling basis on ', {
         "from": "t+1", "to": "t+7"})

st.write('So predictions made 8 days from today will have actual delays ready for us to evaluate our model. What this means is we will be evaluating model performance by the prediction date and NOT the departure date.')

st.subheader('Confusion Matrix and ROC Curve')

st.write(
    f'This displays all the predictions that were made from t+1 to t+7, everyday. This meant that for the same flight, depending on the given information and availability (i.e. the actual arrival of previous flight), we will have different numbers of predictions for each flight.')

confusion = confusion_matrix(y_actual, y_pred_labels)
TN = confusion[0][0]
FN = confusion[1][0]
FP = confusion[0][1]
TP = confusion[1][1]


def get_all_accuracies(TN, FN, FP, TP):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = 2*TP/(2*TP + FP + FN)

    return precision, recall, accuracy, f1


fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
roc_auc = auc(fpr, tpr)


st.write('If we consider all predictions made in the past 90 days, using the confusion matrix we can evaluate the overall performance.')
col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
with col1:
    st.metric(label="Threshold Used",
              value=f"{np.round(predictions['model_threshold'].mean(), 3)}")
    st.metric(label="True Positive", value=f"{TP}")
    st.metric(label="False Positive", value=f"{FP}")
    st.metric(label="True Negative", value=f"{TN}")
    st.metric(label="False Negative", value=f"{FN}")

with col2:
    precision, recall, accuracy, f1 = get_all_accuracies(TN, FN, FP, TP)
    st.metric(label="Precision", value=f"{np.round(precision, 2)}")
    st.metric(label="Recall", value=f"{np.round(recall, 2)}")
    st.metric(label="Accuracy", value=f"{np.round(accuracy*100, 1)}%")
    st.metric(label="F1-Score", value=f"{np.round(f1, 2)}")
    st.metric(label="AUC", value=f"{np.round(roc_auc,2)}")


with col3:
    fig = px.imshow(confusion,
                    text_auto=True,
                    labels=dict(x="Pred", y="Actual"),
                    x=['On-Time', 'Late'],
                    y=['On-Time', 'Late']).update_xaxes(
        side="top").update_coloraxes(showscale=False)

    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.area(
        x=fpr, y=tpr,
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    ).add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    ).update_layout(xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("AUC per Week - A More Useful View")

st.write(f"Even more importantly, we want to monitor if the model has shifted. Having a weekly view of the AUC metric will allow us to see if there is a downward trend. Once we see it go below a certain threshold, it's time to re-train our model and maybe do more feature engineering.")


list_of_prediction_dates = list(
    prediction_performance['prediction_date'].dt.strftime('%Y-%m-%d').unique())

aucs = []

for prediction_date in list_of_prediction_dates:
    preds_df = prediction_performance[prediction_performance['prediction_date'].dt.strftime(
        '%Y-%m-%d') == prediction_date]
    y_actual = preds_df['actual_delayed'].astype(int).to_numpy()
    y_pred = preds_df['prediction_prob'].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

auc_trend = pd.DataFrame(
    {'prediction_date': list_of_prediction_dates, 'auc': aucs})

fig = px.line(auc_trend, x="prediction_date",
              y="auc", labels={'auc': 'AUC', 'prediction_date': 'Date of Prediction'}, width=700).update_layout(yaxis_range=[0, 1]).update_layout(xaxis=dict(showgrid=False),
                                                                                                                                                  yaxis=dict(
                                                                                                                                                      showgrid=False))


st.plotly_chart(fig, use_container_width=True)
