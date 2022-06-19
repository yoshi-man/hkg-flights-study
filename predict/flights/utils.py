import pandas as pd


def get_previous_flight(departing_flight, arrivals):
    """
    Given a row of departing flight, return the previous arrival flight based on the listed rules.
    """
    try:
        past_arrivals = arrivals[((arrivals['origin'] == departing_flight['destination']) &
                                  (arrivals['airline'] == departing_flight['airline']) &
                                  (arrivals['cargo'] == departing_flight['cargo']))].sort_values(by='scheduled_arrival', ascending=False, ignore_index=True)

        past_arrivals['scheduled_departure'] = departing_flight['scheduled_departure']
        past_arrivals['actual_departure'] = departing_flight['actual_departure']

        past_arrivals['scheduled_turnaround'] = (
            past_arrivals['scheduled_departure'] - past_arrivals['scheduled_arrival']).dt.total_seconds()/60
        past_arrivals['actual_turnaround'] = (
            past_arrivals['scheduled_departure'] - past_arrivals['actual_arrival']).dt.total_seconds()/60

        latest_arrival = past_arrivals[(past_arrivals['scheduled_turnaround'] >= 15) & (past_arrivals['scheduled_turnaround'] <= 10800) &
                                       (((past_arrivals['actual_turnaround'] >= 15) & (past_arrivals['actual_turnaround'] <= 10800))
                                        | past_arrivals['actual_turnaround'].isnull())
                                       ].iloc[0][['scheduled_arrival', 'actual_arrival', 'flight_num']]

        return latest_arrival

    except Exception as e:
        return pd.Series({'scheduled_arrival': None, 'actual_arrival': None, 'flight_num': None})
