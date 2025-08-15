import os
import logging
import json
import numpy as np
import pandas as pd

import umap
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from dash.dcc import send_data_frame

# -------------------------------
# Configuration
# -------------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "test1-427219"
DATASET_ID = "consciousness"
TABLE_ID = "papers"
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=PROJECT_ID)

# -------------------------------
# Data Fetching
# -------------------------------
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
    Parses each row to extract embeddings (from 'embedding_model_2') and metadata:
      - article
      - field
      - link
      - statement + evidence
      - fact_key
    Returns a list of dicts.
    """
    results = []
    for row in rows:
        article_text = row.get('article', '')
        facts_dict = row.get('facts', {})
        link_val = row.get('link', '')
        field_val = row.get('field', '')

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

            if embedding:
                results.append({
                    'article': article_text.strip(),
                    'field': field_val.strip(),
                    'link': link_val.strip(),
                    'fact_key': fact_key.strip(),
                    'statement': statement.strip(),
                    'evidence': evidence.strip(),
                    'embedding': np.array(embedding, dtype=float),
                })
    logging.info(f"parse_embeddings: Found {len(results)} total items with valid embeddings.")
    return results


# -------------------------------
# Similarity Computation
# -------------------------------
def compute_top_similarities(items_a, items_b, top_k=10):
    """
    Given two sets of items, compute pairwise cosine similarity
    and return the top K pairs in descending order without reusing any article.

    Each article from Group A and Group B is used only once.
    """
    if not items_a or not items_b:
        logging.debug("compute_top_similarities: One of the groups is empty.")
        return []

    emb_a = np.vstack([x['embedding'] for x in items_a])
    emb_b = np.vstack([x['embedding'] for x in items_b])

    sim_matrix = cosine_similarity(emb_a, emb_b)
    sims = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            sims.append((sim_matrix[i, j], i, j))

    sims.sort(key=lambda x: x[0], reverse=True)
    logging.debug(f"compute_top_similarities: Matrix shape={sim_matrix.shape}, sorted {len(sims)} similarities.")

    results = []
    used_articles_a = set()
    used_articles_b = set()

    for sim_val, i, j in sims:
        a = items_a[i]
        b = items_b[j]
        article_a = a["article"]
        article_b = b["article"]

        # Skip if either article has already been used
        if article_a in used_articles_a or article_b in used_articles_b:
            continue

        # Add the pair
        results.append({
            "similarity": round(float(sim_val), 4),
            "statement_a": a["statement"],
            "evidence_a": a["evidence"],
            "field_a": a["field"],
            "link_a": a["link"],
            "fact_key_a": a["fact_key"],
            "statement_b": b["statement"],
            "evidence_b": b["evidence"],
            "field_b": b["field"],
            "link_b": b["link"],
            "fact_key_b": b["fact_key"],
        })

        # Mark articles as used
        used_articles_a.add(article_a)
        used_articles_b.add(article_b)

        if len(results) >= top_k:
            break

    logging.debug(f"compute_top_similarities: Selected {len(results)} unique pairs.")
    return results


def compute_multi_group_average(list_of_item_lists, top_k=10):
    """
    For 'intersection-average' among multiple groups (2..7).
    We pick exactly one item from each group,
    compute all pairwise cos sims among those chosen items,
    average them, and rank the top K combos by that average.

    list_of_item_lists: e.g. [groupA_items, groupB_items, groupC_items, ...]

    Return up to top_k combos:
      {
        "avg_similarity": float,
        "group1_statement": ...,
        "group1_evidence": ...,
        ...
        "groupN_statement": ...,
        ...
      }
    """
    # Filter out empty groups
    actual_groups = [g for g in list_of_item_lists if g]
    n = len(actual_groups)
    if n < 2:
        logging.debug("compute_multi_group_average: <2 non-empty groups, no intersection-average.")
        return []

    # Log how many items in each group
    for idx, grp in enumerate(actual_groups):
        logging.debug(f"Group {idx+1} size = {len(grp)}")

    # We'll do a Cartesian product picking one item from each group
    combos = list(product(*actual_groups))
    logging.debug(f"compute_multi_group_average: total combos = {len(combos)}")

    results = []
    for combo_idx, combo in enumerate(combos):
        # combo is a tuple of length n, one item per group
        # pairwise cos sims among these n items
        embeddings = [c['embedding'] for c in combo]
        sim_vals = []
        for i in range(n):
            for j in range(i+1, n):
                v = float(cosine_similarity(embeddings[i].reshape(1, -1),
                                            embeddings[j].reshape(1, -1))[0, 0])
                sim_vals.append(v)
        if sim_vals:
            avg_sim = sum(sim_vals)/len(sim_vals)
        else:
            avg_sim = 0.0

        row = {
            "avg_similarity": round(avg_sim, 4),
        }
        # store each group's data
        for idx, item in enumerate(combo):
            gidx = idx+1
            row[f"group{gidx}_statement"] = item["statement"]
            row[f"group{gidx}_evidence"] = item["evidence"]
            row[f"group{gidx}_field"] = item["field"]
            row[f"group{gidx}_link"] = item["link"]
            row[f"group{gidx}_factkey"] = item["fact_key"]

        results.append(row)

        # optional debug for first few combos
        if combo_idx < 5:
            logging.debug(f"Combo idx={combo_idx}, avg_sim={avg_sim}, items={[x['field'] for x in combo]}")

    results.sort(key=lambda x: x["avg_similarity"], reverse=True)
    logging.debug(f"compute_multi_group_average: returning top_k={top_k} from {len(results)} combos.")
    return results[:top_k]


def chunk_string(s, chunk_size=80):
    """
    Splits a string into chunks separated by HTML line breaks.
    """
    if not s:
        return ""
    chunks = [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]
    return "<br>".join(chunks)


def create_app():
    rows = fetch_papers_with_embeddings()
    if not rows:
        logging.warning("No data retrieved or table is empty.")
        return None

    parsed = parse_embeddings(rows)
    if not parsed:
        logging.warning("No valid embeddings found in the table.")
        return None

    df_main = pd.DataFrame(parsed)

    # UMAP 3D
    embeddings = np.vstack(df_main['embedding'].values)
    logging.info("Reducing dimensionality with UMAP (3D)...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)
    logging.info("UMAP dimensionality reduction completed.")

    df_umap = pd.DataFrame(embedding_3d, columns=['x', 'y', 'z'])
    df_umap['article'] = df_main['article']
    df_umap['field'] = df_main['field']
    df_umap['fact_key'] = df_main['fact_key']
    df_umap['statement'] = df_main['statement']
    df_umap['evidence'] = df_main['evidence']
    df_umap['link'] = df_main['link']

    # Hover text
    df_umap['hover_text'] = (
        "Fact: " + df_umap['fact_key'].fillna('') +
        "<br>Statement: " + df_umap['statement'].fillna('') +
        "<br>Evidence: " + df_umap['evidence'].fillna('') +
        "<br>Field: " + df_umap['field'].fillna('') +
        "<br>Link: " + df_umap['link'].fillna('')
    )
    df_umap['hover_text'] = df_umap['hover_text'].apply(lambda x: chunk_string(x, 60))
    df_umap['customdata'] = df_umap.apply(lambda row: {"link": row['link']}, axis=1)

    default_groups = [{'Group': f'Group {i+1}', 'Color': px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]}
                      for i in range(5)]
    unique_fields = sorted(df_umap['field'].dropna().unique().tolist())

    app = dash.Dash(__name__)
    serialized_data = df_main.to_dict('records')

    # Intersection-average columns
    intersection_columns = [{"name": "Avg Similarity", "id": "avg_similarity"}]
    for gidx in range(1, 8):
        intersection_columns.extend([
            {"name": f"Group{gidx} Statement", "id": f"group{gidx}_statement"},
            {"name": f"Group{gidx} Evidence", "id": f"group{gidx}_evidence"},
            {"name": f"Group{gidx} Field", "id": f"group{gidx}_field"},
            {"name": f"Group{gidx} Link", "id": f"group{gidx}_link"},
        ])

    app.layout = html.Div([
        html.H2("3D Visualization & Similarity with Intersection-Average Logging"),
        dcc.Store(id='main-data-store', data=serialized_data),
        dcc.Store(id='intersection-columns', data=intersection_columns),

        # Visualization
        html.Div([
            html.Div([
                html.H5("Define Color Groups (for Field-based coloring)"),
                dash_table.DataTable(
                    id='color-groups-table',
                    columns=[
                        {'name': 'Group', 'id': 'Group', 'type': 'text'},
                        {'name': 'Color', 'id': 'Color', 'type': 'text'}
                    ],
                    data=default_groups,
                    editable=True,
                    row_deletable=True,
                    style_cell={'minWidth': '80px','maxWidth': '180px','fontSize': '12px','padding': '4px','lineHeight': '14px'},
                ),
                html.Button('Add Color Group', id='add-group-button', n_clicks=0),
                html.Hr(),
                html.H5("Assign Fields to Color Groups"),
                dash_table.DataTable(
                    id='field-assignments-table',
                    columns=[
                        {'name': 'Field', 'id': 'Field'},
                        {'name': 'Group', 'id': 'Group', 'presentation': 'dropdown'}
                    ],
                    data=[{'Field': f, 'Group': ''} for f in unique_fields],
                    editable=True,
                    dropdown={
                        'Group': {
                            'options': [
                                {'label': g['Group'], 'value': g['Group']}
                                for g in default_groups
                            ]
                        }
                    },
                    style_cell={'minWidth': '100px','maxWidth': '220px','fontSize': '12px','padding': '4px','lineHeight': '14px'},
                ),
            ], style={'width': '30%', 'display': 'inline-block','verticalAlign': 'top','padding':'10px'}),

            html.Div([
                html.Label("Color By:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[
                        {'label': 'Color by Article', 'value': 'article'},
                        {'label': 'Color by Field', 'value': 'field'}
                    ],
                    value='field',
                    clearable=False,
                    style={'width': '60%', 'marginBottom': '8px'}
                ),
                dcc.Graph(
                    id='3d-scatter',
                    config={'displayModeBar': False},
                    style={'height':'600px'}
                ),
            ], style={'width': '65%','display':'inline-block','verticalAlign':'top','padding':'10px'}),
        ]),

        html.Hr(),
        # Similarities
        html.H3("Up to 7 Groups of Fields - Pairwise or Intersection-Average"),
        html.Div([
            html.P("Enter fields for each group (comma-separated)."
                   "Then pick a mode: Pairwise (A vs B) or Intersection-Average across all non-empty groups. "
                   "We log debug info for the intersection-average approach."),
            html.Div([
                html.Label("Group A Fields:"),
                dcc.Input(id='groupA', type='text', style={'width':'100%'}, value=""),
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group B Fields:"),
                dcc.Input(id='groupB', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group C Fields:"),
                dcc.Input(id='groupC', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group D Fields:"),
                dcc.Input(id='groupD', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group E Fields:"),
                dcc.Input(id='groupE', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group F Fields:"),
                dcc.Input(id='groupF', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),
            html.Div([
                html.Label("Group G Fields:"),
                dcc.Input(id='groupG', type='text', style={'width':'100%'}, value="")
            ], style={'margin':'4px'}),

            html.Label("Top K:", style={'marginTop':'10px'}),
            dcc.Input(id='topK', type='number', value=10, style={'width':'100px'}),
            html.Br(),
            dcc.RadioItems(
                id='similarity-mode',
                options=[
                    {'label': 'Compare only Group A vs Group B (pairwise)', 'value': 'pairwise'},
                    {'label': 'Intersection-Average (all non-empty groups)', 'value': 'intersection-average'}
                ],
                value='pairwise',
                style={'margin':'8px'}
            ),
            html.Button("Compute Similarities", id='similarity-btn', n_clicks=0, style={'margin':'8px'}),
            html.Button("Download CSV", id='download-button', n_clicks=0, style={'margin':'8px'}),
            dcc.Download(id='download-similarities'),
        ], style={'border':'1px solid #ccc','padding':'10px','margin':'10px','width':'95%','display':'inline-block'}),

        dash_table.DataTable(
            id='similarity-table',
            data=[],
            columns=[],
            page_size=10,
            style_table={
                'overflowX': 'auto',
                'margin': '8px auto',
                'width': '95%'
            },
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
                'fontSize': '12px',
                'padding': '4px',
                'lineHeight': '14px'
            },
            style_header={
                'fontWeight': 'bold',
            },
        ),

        # Hidden
        dcc.Store(id='group-color-store'),
        dcc.Store(id='field-group-store'),
        html.Div(id='client-side-dummy', style={'display':'none'})
    ])

    # -------------
    # Color table & assignment
    # -------------
    @app.callback(
        Output('color-groups-table', 'data'),
        Input('add-group-button', 'n_clicks'),
        State('color-groups-table', 'data')
    )
    def add_color_group(n_clicks, rows):
        if n_clicks and n_clicks > 0:
            rows.append({'Group': f'Group {len(rows)+1}', 'Color': 'blue'})
        return rows

    @app.callback(
        Output('group-color-store', 'data'),
        Input('color-groups-table', 'data')
    )
    def store_color_groups(tbl_data):
        group_color = {}
        for row in tbl_data:
            gname = row.get('Group')
            col = row.get('Color')
            if gname and col:
                group_color[gname] = col
        return group_color

    @app.callback(
        Output('field-assignments-table', 'dropdown'),
        Input('color-groups-table', 'data')
    )
    def update_field_dropdown(tbl_data):
        opts = [{'label': r['Group'], 'value': r['Group']} for r in tbl_data]
        return {'Group': {'options': opts}}

    @app.callback(
        Output('field-group-store', 'data'),
        Input('field-assignments-table', 'data')
    )
    def store_field_groups(tbl_data):
        field_group = {}
        for row in tbl_data:
            f = row.get('Field')
            g = row.get('Group')
            if f and g:
                field_group[f] = g
        return field_group

    # -------------
    # 3D Scatter
    # -------------
    df_umap_local = df_umap.copy()

    @app.callback(
        Output('3d-scatter', 'figure'),
        Input('group-color-store', 'data'),
        Input('field-group-store', 'data'),
        Input('color-dropdown', 'value')
    )
    def update_3d_scatter(group_color, field_group, color_by):
        if not group_color:
            group_color = {}
        if not field_group:
            field_group = {}
        dfc = df_umap_local.copy()

        if color_by == 'field':
            dfc['assigned_group'] = dfc['field'].map(field_group)
            dfc['assigned_color'] = dfc['assigned_group'].map(group_color)
            dfc['assigned_color'] = dfc['assigned_color'].fillna('lightgrey')

            color_list = dfc['assigned_color'].unique().tolist()
            traces = []
            for c in color_list:
                subdf = dfc[dfc['assigned_color'] == c]
                groups_in_subdf = subdf['assigned_group'].dropna().unique().tolist()
                gname = groups_in_subdf[0] if groups_in_subdf else "Unassigned"
                traces.append(go.Scatter3d(
                    x=subdf['x'], y=subdf['y'], z=subdf['z'],
                    text=subdf['hover_text'],
                    customdata=subdf['customdata'],
                    mode='markers',
                    name=gname,
                    marker=dict(size=5, color=c, opacity=0.7),
                    hovertemplate='%{text}<extra></extra>'
                ))
            fig = go.Figure(data=traces)
            fig.update_layout(
                height=600,
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title='')
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                hovermode='closest',
                showlegend=True
            )
            return fig

        elif color_by == 'article':
            def gen_color(a):
                if not isinstance(a, str):
                    return 'blue'
                h = abs(hash(a)) % (256**3)
                r = (h >> 16) & 255
                g = (h >> 8) & 255
                b = h & 255
                return f'rgb({r},{g},{b})'

            dfc['assigned_color'] = dfc['article'].apply(gen_color)
            trace = go.Scatter3d(
                x=dfc['x'], y=dfc['y'], z=dfc['z'],
                text=dfc['hover_text'],
                customdata=dfc['customdata'],
                mode='markers',
                marker=dict(size=5, color=dfc['assigned_color'], opacity=0.7),
                hovertemplate='%{text}<extra></extra>'
            )
            fig = go.Figure(data=[trace])
            fig.update_layout(
                height=600,
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title='')
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                hovermode='closest',
                showlegend=False
            )
            return fig

        return go.Figure()

    # Link opening
    app.clientside_callback(
        """
        function(clickData) {
            if (clickData && clickData.points && clickData.points.length > 0) {
                var cd = clickData.points[0].customdata;
                if (cd && cd.link) {
                    window.open(cd.link, "_blank");
                }
            }
            return null;
        }
        """,
        Output('client-side-dummy','children'),
        Input('3d-scatter','clickData')
    )

    # -------------
    # Similarities
    # -------------
    @app.callback(
        Output('similarity-table', 'data'),
        Output('similarity-table', 'columns'),
        Input('similarity-btn', 'n_clicks'),
        State('main-data-store', 'data'),
        State('groupA', 'value'),
        State('groupB', 'value'),
        State('groupC', 'value'),
        State('groupD', 'value'),
        State('groupE', 'value'),
        State('groupF', 'value'),
        State('groupG', 'value'),
        State('topK', 'value'),
        State('similarity-mode', 'value'),
        State('intersection-columns', 'data')
    )
    def compute_similarity(n_clicks,
                           data_store,
                           ga, gb, gc, gd, ge, gf, gg,
                           top_k, mode,
                           intersection_cols):
        if n_clicks < 1 or not data_store:
            return [], []

        df_all = pd.DataFrame(data_store)
        if df_all.empty:
            return [], []

        def parse_fields(s):
            return [f.strip() for f in s.split(',') if f.strip()]

        groupA = parse_fields(ga)
        groupB = parse_fields(gb)
        groupC = parse_fields(gc)
        groupD = parse_fields(gd)
        groupE = parse_fields(ge)
        groupF = parse_fields(gf)
        groupG = parse_fields(gg)

        if not top_k or top_k < 1:
            top_k = 10

        if mode == 'pairwise':
            # Only compare group A vs B
            items_a = df_all[df_all['field'].isin(groupA)].to_dict('records')
            items_b = df_all[df_all['field'].isin(groupB)].to_dict('records')
            for x in items_a:
                x['embedding'] = np.array(x['embedding'], dtype=float)
            for x in items_b:
                x['embedding'] = np.array(x['embedding'], dtype=float)

            results = compute_top_similarities(items_a, items_b, top_k=top_k)
            # Pairwise columns
            columns = [
                {"name": "Similarity", "id": "similarity"},
                {"name": "Statement (A)", "id": "statement_a"},
                {"name": "Evidence (A)", "id": "evidence_a"},
                {"name": "Field (A)", "id": "field_a"},
                {"name": "Link (A)", "id": "link_a"},
                {"name": "Statement (B)", "id": "statement_b"},
                {"name": "Evidence (B)", "id": "evidence_b"},
                {"name": "Field (B)", "id": "field_b"},
                {"name": "Link (B)", "id": "link_b"},
            ]
            logging.debug(f"Pairwise mode: group A size={len(items_a)}, group B size={len(items_b)}, results={len(results)}.")
            return results, columns

        else:
            # intersection-average among all non-empty groups
            # build up to 7 item lists
            def fetch_items(fields):
                subset = df_all[df_all['field'].isin(fields)].to_dict('records')
                for r in subset:
                    r['embedding'] = np.array(r['embedding'], dtype=float)
                return subset

            group_lists = []
            all_labels = ['A','B','C','D','E','F','G']
            field_arrays = [groupA, groupB, groupC, groupD, groupE, groupF, groupG]
            for idx, arr in enumerate(field_arrays):
                if arr:
                    items = fetch_items(arr)
                    logging.debug(f"Group {all_labels[idx]} has {len(items)} items. fields={arr}")
                    group_lists.append(items)

            if len(group_lists) < 2:
                logging.debug("Intersection-average: <2 non-empty groups => no results.")
                return [], intersection_cols

            # Now do compute_multi_group_average
            multi_res = compute_multi_group_average(group_lists, top_k=top_k)
            # intersection columns
            data = []
            for row in multi_res:
                data.append(row)

            logging.debug(f"intersection-average: returned {len(data)} rows.")
            return data, intersection_cols

    # -------------
    # Download CSV
    # -------------
    @app.callback(
        Output('download-similarities', 'data'),
        Input('download-button', 'n_clicks'),
        State('similarity-table', 'data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, table_data):
        if not table_data:
            return None
        df = pd.DataFrame(table_data)
        return send_data_frame(df.to_csv, "similarities.csv")

    return app


def main():
    app = create_app()
    if app:
        app.run_server(debug=True, host='0.0.0.0', port=8080)
    else:
        logging.error("No data found to visualize or compute. Exiting.")


if __name__ == "__main__":
    main()
