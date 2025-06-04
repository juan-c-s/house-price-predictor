import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import duckdb
import os
from house_price_predictor.dash_app.constants import S3_PATH

# Create a global DuckDB connection
def create_duckdb_connection():
    # Create a persistent connection
    conn = duckdb.connect(database=':memory:', read_only=False)
    
    # Install and load extensions
    conn.install_extension("httpfs")
    conn.load_extension("httpfs")
    
    # Configure S3 credentials
    conn.execute("""
        SET s3_region='us-east-1';
        SET s3_access_key_id='{aws_access_key_id}';
        SET s3_secret_access_key='{aws_secret_access_key}';
    """.format(
        aws_access_key_id=os.getenv('aws_access_key_id'),
        aws_secret_access_key=os.getenv('aws_secret_access_key')
    ))
    
    if os.getenv('aws_session_token'):
        conn.execute(f"""
            SET s3_session_token='{os.getenv('aws_session_token')}';
        """)
    
    return conn

# Initialize connection
db_conn = create_duckdb_connection()

PAGE_SIZE = 100
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Explorador de datos en S3 con DuckDB"),
    dcc.Input(
        id='page-number',
        type='number',
        min=0,
        value=0,
        step=1,
        required=True
    ),
    html.Button("Cargar página", id='load-button', n_clicks=0),
    dash_table.DataTable(id='table', page_size=PAGE_SIZE)
])

@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    Input('load-button', 'n_clicks'),
    Input('page-number', 'value'),
    prevent_initial_call=True
)
def update_table(n_clicks, page):
    if page is None:
        page = 0
    
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0
        
    offset = page * PAGE_SIZE
    
    try:
        # Use the global connection
        query = f"""
            SELECT *
            FROM read_parquet('{S3_PATH}')
            LIMIT {PAGE_SIZE} OFFSET {offset}
        """
        # Execute query using the connection
        result = db_conn.execute(query)
        df = result.df()
        
        # Test if we got data
        print(f"Retrieved {len(df)} rows")
        
        columns = [{"name": i, "id": i} for i in df.columns]
        return df.to_dict('records'), columns
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        # Add more detailed error information
        import traceback
        print(traceback.format_exc())
        return [], []

if __name__ == '__main__':
    # Test the connection before starting the server
    try:
        test_query = "SELECT 1"
        result = db_conn.execute(test_query).fetchone()
        print("DuckDB connection test successful")
        
        # Test S3 access
        test_s3_query = f"""
            SELECT COUNT(*) 
            FROM read_parquet('{S3_PATH}')
        """
        count = db_conn.execute(test_s3_query).fetchone()[0]
        print(f"Successfully connected to S3. Found {count} rows in the parquet file")
        
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    app.run(debug=True)

def cleanup():
    # Close the connection when the app stops
    if 'db_conn' in globals():
        db_conn.close()

import atexit
atexit.register(cleanup)