# Exploring What's in the Data

This part of the study consists of:
1. Getting the dataset via the API consistently via Python
2. Basic exploration of a static dataset via R
3. Building a lightweight data pipeline to start collecting flights data everyday
4. Data visualisation in monitoring performance and metrics via Tableau

The pipeline uses simple components from GCP.
![Data Pipeline](https://user-images.githubusercontent.com/38344465/173010837-c2dc2cac-60d3-4781-9bad-d86e1a368a36.png)


## Getting Started
To see how the dataset is being extracted and transformed, [```extract_transform_load.ipynb```](explore/extract_transform_load.ipynb) will be the file to check out. If you'd like to run it yourself, simply clone this repo and run a jupyter server to run through each cell. There are not secret keys required.

To see the basic exploration I did via R, download the file [```flights_eda_notebook.nb.html```](explore/flights_eda_notebook.nb.html) and open via a browser. If you'd like to run it yourself, there's an R notebook in [```r_notebooks/flights_eda_notebook.Rmd```](explore/r_notebooks/flights_eda_notebook.Rmd).

For the dashboard, this is a static dashboard living in [Tableau Public](https://public.tableau.com/app/profile/yoshi.man1207/viz/FlightsThroughHongKongInternationalAirport/Overview). The local file is not included here as there consists of a private key linked to GCP Bigquery. Get in touch if you'd like to see an updated view!


