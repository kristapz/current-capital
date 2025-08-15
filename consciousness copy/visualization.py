import os
import logging
import json
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
from google.cloud import bigquery

# -------------------------------
# Configuration
# -------------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "test1-427219"
DATASET_ID = "consciousness"
TABLE_ID = "papers"
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=PROJECT_ID)


def fetch_papers_with_embeddings():
    """
    Fetches articles, facts, links, and fields from the BigQuery table.
    """
    query = f"""
        SELECT
          article,
          facts,
          link,
          field
        FROM `{FULL_TABLE_ID}`
    """
    try:
        query_job = client_bq.query(query)
        rows = list(query_job)
        logging.info(f"Fetched {len(rows)} rows from {FULL_TABLE_ID}.")
        return rows
    except Exception as e:
        logging.error(f"Failed to fetch data from BigQuery: {e}")
        return []


def parse_embeddings(rows):
    """
    Parses the fetched rows to extract embeddings and associated metadata.
    """
    results = []
    for row in rows:
        article_text = row.get('article', '')
        facts_dict = row.get('facts', {})
        link_val = row.get('link', '')
        field_val = row.get('field', '')  # Include the field value

        if not isinstance(facts_dict, dict):
            continue

        for fact_key, fact_value in facts_dict.items():
            if not isinstance(fact_value, dict):
                continue

            embedding_json = fact_value.get('embedding_model_2', None)
            if not embedding_json:
                continue

            try:
                embedding = json.loads(embedding_json)  # list of floats
            except json.JSONDecodeError:
                embedding = []

            statement = fact_value.get('Statement', '')
            evidence = fact_value.get('Evidence', '')
            content_snippet = f"Statement: {statement} | Evidence: {evidence}"

            if embedding:
                results.append({
                    'article': article_text,
                    'fact_key': fact_key,
                    'embedding': embedding,
                    'content_snippet': content_snippet.strip(),
                    'link': link_val,
                    'field': field_val  # Add the field to each result
                })
    return results


def chunk_string(s, chunk_size=80):
    """
    Splits a string into chunks separated by HTML line breaks.
    """
    if not s:
        return ""
    chunks = [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]
    return "<br>".join(chunks)


def create_visualization():
    """
    Creates the Dash visualization app.
    """
    global df_umap  # Declare df_umap as global so it's accessible in the callback
    rows = fetch_papers_with_embeddings()
    if not rows:
        logging.warning("No data retrieved or table is empty.")
        return None

    parsed = parse_embeddings(rows)
    if not parsed:
        logging.warning("No valid embeddings found in the table.")
        return None

    df = pd.DataFrame(parsed)
    embeddings = np.array(df['embedding'].tolist())

    logging.info("Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)
    logging.info("UMAP dimensionality reduction completed.")

    df_umap = pd.DataFrame(embedding_3d, columns=['x', 'y', 'z'])
    df_umap['article'] = df['article']
    df_umap['fact_key'] = df['fact_key']
    df_umap['link'] = df['link']
    df_umap['field'] = df['field']  # Include the field column

    # Build multiline hover text
    df_umap['hover_text'] = (
        df_umap['fact_key'] + ": " + df['content_snippet'] +
        "<br><br>Link: " + df_umap['link'].fillna('') +
        "<br><br>Field: " + df_umap['field'].fillna('')
    )
    df_umap['hover_text'] = df_umap['hover_text'].apply(lambda txt: chunk_string(txt, 60))

    # We'll store link in customdata, so we can open it from clientside
    df_umap['customdata'] = df_umap.apply(
        lambda row: {"link": row['link']},
        axis=1
    )

    # Fetch unique fields
    unique_fields = df_umap['field'].dropna().unique().tolist()
    unique_fields = sorted(unique_fields)

    # Initialize color groups
    default_groups = [{'Group': f'Group {i+1}', 'Color': px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]}
                     for i in range(5)]  # Initialize with 5 default groups

    # Layout of the app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H3("3D Visualization of 'papers' (Click a point to open link)"),

        # Section for Defining Color Groups
        html.Div([
            html.H4("Define Color Groups"),
            dash_table.DataTable(
                id='color-groups-table',
                columns=[
                    {'name': 'Group Name', 'id': 'Group', 'type': 'text'},
                    {'name': 'Color', 'id': 'Color', 'type': 'text'}
                ],
                data=default_groups,
                editable=True,
                row_deletable=True,
                # Initial color picker column
                style_cell={
                    'minWidth': '150px', 'width': '200px', 'maxWidth': '200px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Color'},
                        'backgroundColor': 'white'
                    }
                ],
                # Adding color picker via a Dropdown with color names or hex codes
                # For better user experience, integrating a color picker might be necessary
            ),
            html.Button('Add Group', id='add-group-button', n_clicks=0),
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin': '10px'}),

        # Section for Assigning Fields to Groups
        html.Div([
            html.H4("Assign Fields to Groups"),
            dash_table.DataTable(
                id='field-assignments-table',
                columns=[
                    {'name': 'Field', 'id': 'Field'},
                    {'name': 'Group', 'id': 'Group', 'presentation': 'dropdown'}
                ],
                data=[{'Field': field, 'Group': ''} for field in unique_fields],
                editable=True,
                dropdown={
                    'Group': {
                        'options': [
                            {'label': group['Group'], 'value': group['Group']}
                            for group in default_groups
                        ]
                    }
                },
                style_cell={
                    'minWidth': '200px', 'width': '250px', 'maxWidth': '250px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
            )
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin': '10px'}),

        # Section for Visualization
        html.Div([
            dcc.Dropdown(
                id='color-dropdown',
                options=[
                    {'label': 'Color by Article', 'value': 'article'},
                    {'label': 'Color by Field', 'value': 'field'}
                ],
                value='field',  # default value set to 'field'
                clearable=False,
                style={'width': '50%', 'margin-bottom': '20px'}
            ),
            dcc.Graph(
                id='3d-scatter',
                config={'displayModeBar': False},
                clickData=None
            ),
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin': '10px'}),

        # Hidden Stores for Data
        dcc.Store(id='group-color-store'),
        dcc.Store(id='field-group-store'),

        # Hidden Div for Client-Side Callback
        html.Div(id='client-side-dummy', style={'display': 'none'}),
    ], style={'width': '100%', 'display': 'inline-block'})

    # Callback to add a new group row
    @app.callback(
        Output('color-groups-table', 'data'),
        [Input('add-group-button', 'n_clicks')],
        [State('color-groups-table', 'data'),
         State('color-groups-table', 'columns')]
    )
    def add_group(n_clicks, rows, columns):
        if n_clicks > 0:
            new_group = {'Group': f'Group {len(rows)+1}', 'Color': 'blue'}
            rows.append(new_group)
        return rows

    # Callback to update group-color store when color groups table changes
    @app.callback(
        Output('group-color-store', 'data'),
        [Input('color-groups-table', 'data')]
    )
    def update_group_color_store(groups_data):
        group_color = {}
        for group in groups_data:
            group_name = group.get('Group')
            color = group.get('Color')
            if group_name and color:
                group_color[group_name] = color
        return group_color

    # Callback to update field-group assignments dropdown options based on color groups
    @app.callback(
        Output('field-assignments-table', 'dropdown'),
        [Input('color-groups-table', 'data')]
    )
    def update_field_dropdown_options(groups_data):
        group_options = [{'label': group['Group'], 'value': group['Group']}
                        for group in groups_data]
        return {
            'Group': {
                'options': group_options
            }
        }

    # Callback to update field-group store when field assignments table changes
    @app.callback(
        Output('field-group-store', 'data'),
        [Input('field-assignments-table', 'data')]
    )
    def update_field_group_store(assignments_data):
        field_group = {}
        for assignment in assignments_data:
            field = assignment.get('Field')
            group = assignment.get('Group')
            if field and group:
                field_group[field] = group
        return field_group

    # Callback to update the figure based on group-color and field-group mappings
    @app.callback(
        Output('3d-scatter', 'figure'),
        [Input('group-color-store', 'data'),
         Input('field-group-store', 'data'),
         Input('color-dropdown', 'value')]
    )
    def update_figure(group_color, field_group, color_dropdown):
        if not group_color:
            group_color = {}
        if not field_group:
            field_group = {}

        if color_dropdown == 'field':
            # Map fields to colors based on group assignments
            df_umap['assigned_group'] = df_umap['field'].map(field_group)
            df_umap['assigned_color'] = df_umap['assigned_group'].map(group_color)

            # Check for fields without group assignment
            unassigned = df_umap[df_umap['assigned_color'].isna()]

            # Assign a default color to unassigned points
            default_color = 'lightgrey'
            df_unassigned = df_umap[df_umap['assigned_color'].isna()].copy()
            if not df_unassigned.empty:
                df_unassigned['assigned_color'] = default_color

            # Combine assigned and unassigned
            df_plot = pd.concat([df_umap[df_umap['assigned_color'].notna()], df_unassigned], ignore_index=True)

            # Create a list of unique groups
            unique_groups = df_plot['assigned_group'].dropna().unique().tolist()

            # Create traces for each group
            traces = []
            for group in unique_groups:
                subdf = df_plot[df_plot['assigned_group'] == group]
                color = group_color.get(group, default_color)
                trace = go.Scatter3d(
                    x=subdf['x'],
                    y=subdf['y'],
                    z=subdf['z'],
                    text=subdf['hover_text'],
                    customdata=subdf['customdata'],
                    mode='markers',
                    name=group,
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.7
                    ),
                    hovertemplate='%{text}<extra></extra>'
                )
                traces.append(trace)

            # Add unassigned points if any
            if not df_unassigned.empty:
                trace = go.Scatter3d(
                    x=df_unassigned['x'],
                    y=df_unassigned['y'],
                    z=df_unassigned['z'],
                    text=df_unassigned['hover_text'],
                    customdata=df_unassigned['customdata'],
                    mode='markers',
                    name='Unassigned',
                    marker=dict(
                        size=5,
                        color=default_color,
                        opacity=0.7
                    ),
                    hovertemplate='%{text}<extra></extra>'
                )
                traces.append(trace)

            fig = go.Figure(data=traces)
            fig.update_layout(
                height=800,
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title='')
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Arial',
                    align='left',
                    namelength=-1
                ),
                showlegend=True  # Show legend when coloring by field
            )

        elif color_dropdown == 'article':
            # Assign a unique color to each article using a hash-based RGB generator
            def generate_color(article):
                """
                Generates a unique RGB color string for each article based on its hash.
                """
                if not isinstance(article, str):
                    return 'blue'  # Default color for non-string articles
                hash_code = abs(hash(article)) % (256**3)
                r = (hash_code >> 16) & 255
                g = (hash_code >> 8) & 255
                b = hash_code & 255
                return f'rgb({r}, {g}, {b})'

            # Apply the color generation function to each article
            df_umap['assigned_color'] = df_umap['article'].apply(generate_color)

            fig = go.Figure(data=go.Scatter3d(
                x=df_umap['x'],
                y=df_umap['y'],
                z=df_umap['z'],
                text=df_umap['hover_text'],
                customdata=df_umap['customdata'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_umap['assigned_color'],
                    opacity=0.7,
                    showscale=False  # Hide color scale
                ),
                hovertemplate='%{text}<extra></extra>'
            ))

            fig.update_layout(
                height=800,
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title='')
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Arial',
                    align='left',
                    namelength=-1
                ),
                showlegend=False  # Hide legend when coloring by article
            )

        else:
            # Default to no coloring
            fig = go.Figure()

        return fig

    # Client-Side Callback: open link in a new tab on click
    app.clientside_callback(
        """
        function(clickData) {
            if (clickData && clickData.points && clickData.points.length > 0) {
                var pointData = clickData.points[0].customdata;
                if (pointData && pointData.link) {
                    var linkUrl = pointData.link;
                    // Open link in a new tab
                    window.open(linkUrl, "_blank");
                }
            }
            // Return null or no_update because we don't need
            // to update anything in the layout
            return null;
        }
        """,
        Output('client-side-dummy', 'children'),
        Input('3d-scatter', 'clickData')
    )

    return app


def main():
    """
    Main entry point to run the Dash app.
    """
    app = create_visualization()
    if app:
        app.run_server(debug=True)
    else:
        logging.error("Could not launch the app because no data was found.")


if __name__ == "__main__":
    main()
