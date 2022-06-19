# Prediction - Predicting Departure Delays for HKG Flights

The aim of this part is:
> To build a lightweight end to end prediction pipeline that can continuously predict the departure delays for upcoming flights in HKG, based on a continuously updated database of past flights record ([See Explore](/explore)).

This means that we need the below 3 main components to help us get to a contiuous prediction pipeline:

1. **Model Building**: Using our past flight records stored in BigQuery, train a _satisfactory_ model using XGBoost to predict the probability of delay at departure
2. **Model Deployment**: A way for us to get this trained model to do batch inference in a regular interval (not realtime due to data refresh frequency, it's unecessary)
3. **Model Monitoring**: A frontend app for us to monitor how well our model is doing, and signal if retraining is needed

![Draft - Frame 2 (1)](https://user-images.githubusercontent.com/38344465/174471116-d8a1b067-04c4-4ef5-a8bc-8020b069a51c.jpg)


## Getting Started

Feel free to check out the notebooks here for more information on each steps, more specifically the model building and deployment part. Clone the notebook and fire up Jupyter to run through the steps taken to get to a model via XGBoost.
