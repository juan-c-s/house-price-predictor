import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

from house_price_predictor.dash_app.constants import S3_PATH

def create_duckdb_connection():
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.install_extension("httpfs")
    conn.load_extension("httpfs")
    
    if os.getenv('AWS_SESSION_TOKEN'):
        conn.execute(f"""
            SET s3_session_token='{os.getenv('AWS_SESSION_TOKEN')}';
        """)
    conn.execute("""
        SET s3_region='us-east-1';
        SET s3_access_key_id='{AWS_ACCESS_KEY_ID}';
        SET s3_secret_access_key='{AWS_SECRET_ACCESS_KEY}';
    """.format(
        AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID'),
        AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
    ))
    
    return conn

db_conn = create_duckdb_connection()

PAGE_SIZE = 10
app = dash.Dash(__name__)

# Simplified layout using built-in pagination
app.layout = html.Div([
    html.H2(
        "House Price Predictions Explorer",
        style={
            'textAlign': 'center',
            'marginBottom': '24px',
            'color': '#1c3b5a',
            'fontWeight': '600'
        }
    ),
    html.Div(
        id='table-container',
        className='table-container'
    ),
    html.Div(
        id='pagination-container',
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'marginTop': '20px'
        }
    ),
    dash_table.DataTable(
        page_size=PAGE_SIZE,
        page_current=0,  # Start at first page
        page_action='custom',  # Enable custom pagination
        css=[
            {
                'selector': '.dash-table',
                'rule': 'margin: 20px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'
            },
            {
                'selector': '.dash-table-header',
                'rule': 'background-color: #f8f9fa; color: #333; font-weight: 600;'
            },
            {
                'selector': '.dash-table-cell',
                'rule': 'padding: 12px; border-bottom: 1px solid #e9ecef;'
            }
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'minWidth': '100px', 'maxWidth': '300px', 'whiteSpace': 'normal', 'padding': '10px'},
        style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'}
    )
])

@app.callback(
    Output('table-container', 'children'),
    Output('pagination-container', 'children'),
    [Input('pagination-container', 'n_clicks')],
    [State('pagination-container', 'children')]
)
def update_table(n_clicks, current_children):
    try:
        # Get total count for pagination
        count_query = f"""
            SELECT COUNT(*) 
            FROM read_parquet('{S3_PATH}')
        """
        total_count = db_conn.execute(count_query).fetchone()[0]
        total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
        
        # Get current page from button clicks
        page = 0
        if current_children:
            for child in current_children:
                if isinstance(child, dict) and 'props' in child:
                    props = child['props']
                    if 'className' in props and 'current-page' in props.get('className', ''):
                        page = int(props.get('children', '0'))
                        break
        
        offset = page * PAGE_SIZE
        
        # Query data for current page
        query = f"""
            SELECT *
            FROM read_parquet('{S3_PATH}')
            LIMIT {PAGE_SIZE} OFFSET {offset}
        """
        result = db_conn.execute(query)
        df = result.df()
        
        # Create table
        table_children = [
            html.Thead([
                html.Tr([
                    html.Th(
                        col,
                        style={
                            'textAlign': 'center' if col in ['prediction', 'actual', 'error', 'error_percentage'] else 'left',
                            'padding': '12px',
                            'backgroundColor': '#1c3b5a',
                            'color': 'white',
                            'fontWeight': '600'
                        }
                    ) for col in df.columns
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(
                        str(cell),
                        style={
                            'textAlign': 'right' if column in ['prediction', 'actual', 'error', 'error_percentage'] else 'left',
                            'padding': '12px',
                            'borderBottom': '1px solid #e2e8f0',
                            'color': '#1e293b'
                        }
                    ) for cell, column in zip(row, df.columns)
                ],
                style={
                    'transition': 'background-color 0.2s ease',
                    'cursor': 'pointer'
                },
                className='table-row'
                ) for row in df.itertuples(index=False)
            ])
        ]
        
        # Create pagination buttons
        pagination_buttons = []
        for i in range(total_pages):
            button = html.Button(
                str(i + 1),
                className='pagination-button' + (' current-page' if i == page else ''),
                style={
                    'margin': '0 4px',
                    'padding': '8px 16px',
                    'border': '1px solid #e2e8f0',
                    'backgroundColor': 'white' if i != page else '#1c3b5a',
                    'color': '#1c3b5a' if i != page else 'white',
                    'cursor': 'pointer',
                    'borderRadius': '4px'
                }
            )
            pagination_buttons.append(button)
        
        return table_children, pagination_buttons
    except Exception as e:
        print(f"Error: {str(e)}")
        return [html.Div("Error loading data", style={
            'textAlign': 'center',
            'color': '#1c3b5a',
            'marginTop': '20px'
        })], []

if __name__ == '__main__':
    try:
        # Get total count for pagination
        count_query = f"""
            SELECT COUNT(*) 
            FROM read_parquet('{S3_PATH}')
        """
        count = db_conn.execute(count_query).fetchone()[0]
        print(f"Successfully connected to S3. Found {count} rows in the parquet file")
        
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    app.run(debug=True, port="3030")

def cleanup():
    if 'db_conn' in globals():
        db_conn.close()

import atexit
atexit.register(cleanup)
