# Import libraries
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load Data, Models, and Pre-computed Assets

# Load the main dataset to populate dropdowns
try:
    df = pd.read_csv('Backend\\Output Files\\stage_6_property_data.csv')
except FileNotFoundError:
    print("Error: 'stage_6_property_data.csv' not found. Make sure it's in the same directory.")
    exit()

# Load the trained price prediction model
try:
    price_model = joblib.load('Backend\\Output Files\\Price Prediction model files\\conformal_predict_price_model_pipeline_catBoost.joblib')
    q_hat = joblib.load('Backend\\Output Files\\Price Prediction model files\\q_hat.joblib')
except FileNotFoundError:
    print("Error: 'conformal_predict_price_model_pipeline_catBoost.joblib' not found.")
    exit()

# Load the assets for the recommendation engine
try:
    society_profiles = joblib.load('Backend\\Output Files\\Society_Recommendation_System_Files\\society_profiles.joblib')
    society_profiles_scaled = joblib.load('Backend\\Output Files\\Society_Recommendation_System_Files\\society_profiles_scaled.joblib')
    society_location_map = pd.read_csv('Backend\\Output Files\\Society_Recommendation_System_Files\\society_location_map.csv')
    society_scaler = joblib.load('Backend\\Output Files\\Society_Recommendation_System_Files\\standardScaler.joblib')
except FileNotFoundError as e:
    print(f"Error loading recommender assets: {e}. Please ensure all files are present.")
    exit()

# Define Training Schema and Global Statistics
# Exact training columns that the pipeline expects
X_train = pd.read_csv('Backend\\Output Files\\Price Prediction model files\\X_train_for_CatBoost.csv')
EXPECTED_COLUMNS = X_train.columns.to_list()

# Distance columns for location-aware imputation
DIST_COLS = [c for c in EXPECTED_COLUMNS if c.startswith('dist_to_')]

# Global statistics for imputation
GLOBAL_NUM_MEAN = df.select_dtypes(include=['number']).mean(numeric_only=True)
GLOBAL_CAT_MODE = df.select_dtypes(exclude=['number']).mode().iloc[0] if not df.select_dtypes(exclude=['number']).empty else pd.Series(dtype=object)

# Amenity/utility numeric columns (exclude core features and distances)
AMENITY_NUM_COLS = [c for c in EXPECTED_COLUMNS
                    if c not in ['City','Area','Type of Property','Transaction Type','Property Lifespan',
                                 'Commercial','Covered Area','Bedrooms','Bathrooms','Balconies',
                                 'House Help Room','Store Room','Puja Room'] 
                    and c not in DIST_COLS
                    and (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))]

# Object/categorical expected columns
CAT_OBJECT_COLS = [c for c in EXPECTED_COLUMNS if c in df.columns and df[c].dtype == object]

# Pre-compute dropdown options
CITIES = sorted(df['City'].unique())
AREAS = sorted(df['Area'].unique())
PROPERTY_TYPES = sorted(df['Type of Property'].unique())
TRANSACTION_TYPES = sorted(df['Transaction Type'].unique())
PROPERTY_LIFESPANS = sorted(df['Property Lifespan'].unique())
BEDROOMS = sorted(df['Bedrooms'].unique())
BATHROOMS = sorted(df['Bathrooms'].unique())
BALCONIES = sorted(df['Balconies'].unique())

# Define the full feature list for the recommender vector
RECOMMENDER_FEATURES = society_profiles_scaled.columns.tolist()

print("--- Application assets loaded successfully. ---")

# App Initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Application Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("ML driven Real Estate Analytics"), width=12, className="text-center my-4")
    ]),

    dbc.Row([
        # LEFT COLUMN: USER INPUTS
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Configure Your Property")),
                dbc.CardBody([
                    # City Dropdown
                    dbc.Label("City", html_for="city-dropdown"),
                    dcc.Dropdown(id='city-dropdown', options=[{'label': i, 'value': i} for i in CITIES], value=CITIES[0]),
                    html.Br(),

                    # Area Dropdown
                    dbc.Label("Area / Locality", html_for="area-dropdown"),
                    dcc.Dropdown(id='area-dropdown', options=[], value=None),
                    html.Br(),

                    # Property Type Dropdown
                    dbc.Label("Type of Property", html_for="prop-type-dropdown"),
                    dcc.Dropdown(id='prop-type-dropdown', options=[{'label': i, 'value': i} for i in PROPERTY_TYPES], value=PROPERTY_TYPES[0]),
                    html.Br(),

                    # Transaction Type Dropdown
                    dbc.Label("Transaction Type", html_for="trans-type-dropdown"),
                    dcc.Dropdown(id='trans-type-dropdown', options=[{'label': i, 'value': i} for i in TRANSACTION_TYPES], value=TRANSACTION_TYPES[0]),
                    html.Br(),

                    # Property Lifespan Dropdown
                    dbc.Label("Property Age", html_for="lifespan-dropdown"),
                    dcc.Dropdown(id='lifespan-dropdown', options=[{'label': i, 'value': i} for i in PROPERTY_LIFESPANS], value=PROPERTY_LIFESPANS[0]),
                    html.Br(),

                    # Covered Area Input
                    dbc.Label("Covered Area (in sq. ft.)", html_for="area-input"),
                    dbc.Input(id='area-input', type='number', value=1000, min=100, max=10000),
                    html.Br(),

                    # Bedrooms Dropdown
                    dbc.Label("Bedrooms", html_for="bedrooms-dropdown"),
                    dcc.Dropdown(id='bedrooms-dropdown', options=[{'label': i, 'value': i} for i in BEDROOMS], value=BEDROOMS[0]),
                    html.Br(),

                    # Bathrooms Dropdown
                    dbc.Label("Bathrooms", html_for="bathrooms-dropdown"),
                    dcc.Dropdown(id='bathrooms-dropdown', options=[{'label': i, 'value': i} for i in BATHROOMS], value=BATHROOMS[0]),
                    html.Br(),

                    # Balconies Dropdown
                    dbc.Label("Balconies", html_for="balconies-dropdown"),
                    dcc.Dropdown(id='balconies-dropdown', options=[{'label': i, 'value': i} for i in BALCONIES], value=BALCONIES[0]),
                    html.Br(),

                    # Binary Radio Items
                    dbc.Label("Is it a Commercial Property?"),
                    dbc.RadioItems(id='commercial-radio', options=[{'label': 'Yes', 'value': 'Y'}, {'label': 'No', 'value': 'N'}], value='N', inline=True),
                    html.Br(),

                    dbc.Label("House Help Room Available?"),
                    dbc.RadioItems(id='house-help-radio', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0, inline=True),
                    html.Br(),

                    dbc.Label("Store Room Available?"),
                    dbc.RadioItems(id='store-room-radio', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0, inline=True),
                    html.Br(),

                    # Submit Button
                    dbc.Button("Predict Price & Recommend Societies", id='submit-button', color="primary", className="mt-3", n_clicks=0)
                ])
            ]),
            width=4
        ),

        # RIGHT COLUMN: OUTPUTS
        dbc.Col(
            dbc.Spinner(
                [
                    # Price Prediction Output
                    dbc.Card([
                        dbc.CardHeader(html.H4("Price Analysis")),
                        dbc.CardBody(id='price-output', children="Please configure your property and click the button to see the results.")
                    ]),
                    html.Br(),
                    # Society Recommendation Output
                    dbc.Card([
                        dbc.CardHeader(html.H4("Top 5 Society Recommendations")),
                        dbc.CardBody(id='recommendation-output')
                    ])
                ]
            ),
            width=8
        )
    ])
], fluid=True)

@app.callback(
    Output('area-dropdown', 'options'),
    Output('area-dropdown', 'value'),
    Input('city-dropdown', 'value'),
)
def update_area_options(selected_city):
    if not selected_city:
        return [], None
    areas = sorted(df.loc[df['City'] == selected_city, 'Area'].dropna().unique())
    options = [{'label': a, 'value': a} for a in areas]
    value = areas[0] if areas else None
    return options, value

# Backend Callback Logic
@app.callback(
    Output('price-output', 'children'),
    Output('recommendation-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [
        State('city-dropdown', 'value'),
        State('area-dropdown', 'value'),
        State('prop-type-dropdown', 'value'),
        State('trans-type-dropdown', 'value'),
        State('lifespan-dropdown', 'value'),
        State('commercial-radio', 'value'),
        State('area-input', 'value'),
        State('bedrooms-dropdown', 'value'),
        State('bathrooms-dropdown', 'value'),
        State('balconies-dropdown', 'value'),
        State('house-help-radio', 'value'),
        State('store-room-radio', 'value')
    ]
)
def update_outputs(n_clicks, city, area, prop_type, trans_type, lifespan, commercial, covered_area, bedrooms, bathrooms, balconies, house_help, store_room):
    # Prevent firing on initial page load
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    # HARD FILTER FOR LOCATION
    candidate_map = society_location_map[
        (society_location_map['City'] == city) &
        (society_location_map['Area'] == area)
    ]
    candidate_list = candidate_map['Society'].tolist()

    # BUILD FULL FEATURE VECTOR FOR PRICE PREDICTION
    # Build a full feature row matching training schema
    row = {col: np.nan for col in EXPECTED_COLUMNS}

    # Mandatory UI fields
    row['City'] = city
    row['Area'] = area
    row['Type of Property'] = prop_type
    row['Transaction Type'] = trans_type
    row['Property Lifespan'] = lifespan
    row['Commercial'] = commercial
    row['Covered Area'] = covered_area
    row['Bedrooms'] = bedrooms
    row['Bathrooms'] = bathrooms
    row['Balconies'] = balconies
    row['House Help Room'] = house_help
    row['Store Room'] = store_room

    # Location-aware distances using local means within candidate_list
    if candidate_list:
        local_profiles = society_profiles.loc[[s for s in candidate_list if s in society_profiles.index]]
        for col in DIST_COLS:
            if col in local_profiles.columns:
                val = local_profiles[col].mean()
                if pd.notna(val):
                    row[col] = float(val)

    # Numeric amenity/utilities imputation with global means
    for col in AMENITY_NUM_COLS:
        if pd.isna(row.get(col, np.nan)):
            mean_val = GLOBAL_NUM_MEAN.get(col, np.nan)
            if pd.notna(mean_val):
                row[col] = float(mean_val)

    # Fill remaining categorical/object expected fields with global mode
    for col in CAT_OBJECT_COLS:
        if pd.isna(row.get(col, np.nan)):
            mode_val = GLOBAL_CAT_MODE.get(col, np.nan)
            if pd.notna(mode_val):
                row[col] = mode_val

    # Handle Puja Room / Study specifically
    for col in ['Puja Room','Study']:
        if col in EXPECTED_COLUMNS and pd.isna(row.get(col, np.nan)):
            # If numeric in training, use mean; else use mode
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mean_val = GLOBAL_NUM_MEAN.get(col, np.nan)
                if pd.notna(mean_val):
                    row[col] = float(mean_val)
            else:
                mode_val = GLOBAL_CAT_MODE.get(col, np.nan)
                if pd.notna(mode_val):
                    row[col] = mode_val

    # Create input DataFrame with exact columns in the right order
    input_full = pd.DataFrame([row], columns=EXPECTED_COLUMNS)
    
    # PRICE PREDICTION WITH CONFIDENCE INTERVALS
    try:
        pred = price_model.predict(input_full)
        point_pred = float(np.exp(pred[0]))
        lower_bound = float(np.exp(pred[0] - q_hat))
        upper_bound = float(np.exp(pred[0] + q_hat))

        price_output_component = html.Div([
            html.H5(f"Predicted Price: ₹ {point_pred:.2f} Crores", className="text-success"),
            html.P(f"95% Confidence Range: ₹ {lower_bound:.2f} Cr - ₹ {upper_bound:.2f} Cr")
        ])
        print(price_output_component)
    except Exception as e:
        price_output_component = html.Div([
            html.H5("Error in Price Prediction", className="text-danger"),
            html.P(f"Details: {str(e)}")
        ])

    # SOCIETY RECOMMENDATION LOGIC
    if not candidate_list:
        recommendation_output_component = html.P("No societies found in the database for selected area to make a recommendation.")
        return price_output_component, recommendation_output_component

    try:
        # Construct the User's "Ideal Property" Vector for Recommendation
        # Isolate local profiles to get location-specific averages
        local_profiles = society_profiles.loc[[s for s in candidate_list if s in society_profiles.index]]

        # Create the user vector template
        user_vector = pd.Series(index=RECOMMENDER_FEATURES, dtype=float)
    
        # Fill with user's direct inputs
        user_vector['Covered Area'] = covered_area
        user_vector['Bedrooms'] = bedrooms
        user_vector['Bathrooms'] = bathrooms
        user_vector['House Help Room'] = house_help
        user_vector['Store Room'] = store_room
        user_vector['Commercial'] = 1 if commercial == 'Y' else 0

        # Fill with location-aware averages for distance features
        distance_cols = [col for col in RECOMMENDER_FEATURES if 'dist_to_' in col]
        for col in distance_cols:
            if col in local_profiles.columns:
                user_vector[col] = local_profiles[col].mean()

        # Fill remaining NaNs with global averages from society profiles
        global_means = society_profiles.mean()
        user_vector.fillna(global_means, inplace=True)

        # Convert user_vector series to a dataframe
        user_vector = user_vector.to_frame().T
        
        # Synchronize the columns exactly in same order as society_profiles
        user_vector = user_vector[RECOMMENDER_FEATURES]
        
        # Scale the user vector
        user_vector_scaled = society_scaler.transform(user_vector) 

        # Calculate Similarity and Rank
        candidate_profiles_scaled = society_profiles_scaled.loc[[s for s in candidate_list if s in society_profiles_scaled.index]]
        similarity_scores = cosine_similarity(user_vector_scaled, candidate_profiles_scaled)

        # Create a series of scores with society names
        scores_series = pd.Series(similarity_scores[0], index=candidate_profiles_scaled.index)
        top_5_societies = scores_series.nlargest(5)

        # Format the output
        recommendation_output_component = dbc.ListGroup(
            [dbc.ListGroupItem(f"{i+1}. {society_name}") 
             for i, (society_name, score) in enumerate(top_5_societies.items())]
        )

    except Exception as e:
        recommendation_output_component = html.Div([
            html.H5("Error in Society Recommendation", className="text-danger"),
            html.P(f"Details: {str(e)}")
        ])

    return price_output_component, recommendation_output_component

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)