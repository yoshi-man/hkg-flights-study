from package.predict import end_to_end_pipeline
import pandas as pd
from google.cloud import bigquery


def run_predictions(request):
    MODEL_NAMES = ["20220618_iata_encoder.pkl",
                   "20220618_dest_encoder.pkl",
                   "20220618_t_0_13112187385559082.pkl"
                   ]

    output = end_to_end_pipeline(MODEL_NAMES)

    client = bigquery.Client()
    predictions_table = client.get_table(
        "project-eagle-347210.flights.predictions")

    errors = client.insert_rows_from_dataframe(predictions_table, output)

    if errors == []:
        return 'Completed without Errors.'

    return 'With errors'
