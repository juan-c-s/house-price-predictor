import dash
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from house_price_predictor.dash_app.read_s3_file import read_from_s3
from house_price_predictor.dash_app.constants import HOUSING_S3_BUCKET, HOUSING_S3_KEY_REFINED

# Create sample data (you can replace this with your actual data)
np.random.seed(42)

df = read_from_s3(HOUSING_S3_BUCKET, HOUSING_S3_KEY_REFINED, file_type='parquet')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('House Price Prediction Metrics'),
    
    # Metrics Table
    html.H2('Metrics Table'),
    dash_table.DataTable(
        id='metrics-table',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.round(2).to_dict('records'),
        style_table={'overflowX': 'auto'},
        page_size=10
    ),
    
    # Error Plot
    html.H2('Error Distribution'),
    dcc.Graph(
        id='error-plot',
        figure=px.histogram(
            df,
            x='error',
            title='Distribution of Prediction Errors',
            labels={'error': 'Prediction Error ($)'},
            marginal='box'
        )
    )
])

if __name__ == '__main__':
    app.run(debug=True)
