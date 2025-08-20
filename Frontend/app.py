# Import libraries
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

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

# --- Helper Functions for Visualizations ---
def create_price_kde(city, area, predicted_price):
    area_data = df[(df['City'] == city) & (df['Area'] == area)]
    prices = area_data['Price (Crores)'].dropna().values
    prices_median = area_data['Price (Crores)'].dropna().median()
    if len(prices) < 2:
        return html.P("Not enough data for KDE plot in selected area.")

    # KDE calculation
    kde = gaussian_kde(prices)
    x_range = np.linspace(prices.min(), prices.max(), 200)
    y_kde = kde(x_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines', name='Price Density'))
    fig.add_vline(x=predicted_price, line_dash="dash", line_color="red",
                  annotation_text=f"Prediction: ₹{predicted_price:.2f} Cr")
    fig.add_vline(x=prices_median, line_dash="dash", line_color="orange",
                  annotation_text=f"Median: ₹{prices_median:.2f} Cr", annotation_position="left")

    fig.update_layout(
        title=f'Price Distribution of Properties in {area}, {city}',
        xaxis_title='Price (Crores)',
        yaxis_title='Density',
        height=400
    )
    return dcc.Graph(figure=fig)

def create_feature_comparison(city, area, user_features):
    """Create bar chart comparing user features to area averages"""
    area_data = df[(df['City'] == city) & (df['Area'] == area)]
    area_data['Balconies'].replace({
        '1.0': 1,
        '2.0': 2,
        '3.0': 3,
        '3+': 4

    }, inplace=True 
    )
    if area_data.empty:
        return html.P("No data available for selected area.")
    
    # Calculate area averages
    area_averages = {
        'Covered Area': area_data['Covered Area'].mean()
    }

    # Create comparison data
    features = list(area_averages.keys())
    user_values = [user_features[f] for f in features]
    area_values = [area_averages[f] for f in features]
    
    fig = go.Figure(data=[
        go.Bar(name='Your Property', y=features, x=user_values, orientation='h', text=[f"{v:.2f}" for v in user_values], textposition='auto'),
        go.Bar(name='Area Average', y=features, x=area_values, orientation='h', text=[f"{v:.2f}" for v in area_values], textposition='auto')
    ])
    
    fig.update_layout(barmode='group', title=f'Your Property vs Area Average in {area}',
                     height=400)
    return dcc.Graph(figure=fig)

def create_amenity_wordcloud(top_societies):
    """Create wordcloud of amenities in top recommended societies"""
    if not top_societies:
        return html.P("No societies available for wordcloud.")
    
    # Get amenity columns (binary features)
    amenity_cols = [col for col in society_profiles.columns 
                   if col not in ['Commercial', 'Price (Crores)', 'Covered Area', 'Carpet Area', 'Bedrooms', 'Bathrooms', 'Balconies', 
                                 'House Help Room', 'Store Room', 'Commercial', 'latitude', 'longitude', 'listing_count'] 
                   and not col.startswith('dist_to_')]
    
    # Calculate amenity frequencies in top societies
    top_society_data = society_profiles.loc[top_societies, amenity_cols]
    amenity_freq = top_society_data.sum().sort_values(ascending=False)
    
    if amenity_freq.empty:
        return html.P("No amenity data available.")
    
    # Create wordcloud
    wordcloud_dict = dict(zip(amenity_freq.index, amenity_freq.values))
    
    try:
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(wordcloud_dict)
        
        # Convert to base64 for display
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        encoded_img = base64.b64encode(img.read()).decode()
        
        return html.Img(src=f"data:image/png;base64,{encoded_img}", 
                       style={'width': '100%', 'height': 'auto'})
    except:
        return html.P("Error generating wordcloud.")

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

                    # Commercial Radio Button
                    dbc.Label("Looking for a Commercial Property?"),
                    dbc.RadioItems(id='commercial-radio', options=[{'label': 'Yes', 'value': 'Y'}, {'label': 'No', 'value': 'N'}], value='N', inline=True),
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
                    dcc.Dropdown(id='lifespan-dropdown', options=[
                        {'label': 'New Construction', 'value': 'New construction'},
                        {'label': 'Less than 5 years', 'value': 'Less than 5 years'},
                        {'label': '5 to 10 years', 'value': '5 to 10 years'},
                        {'label': '10 to 15 years', 'value': '10 to 15 years'},
                        {'label': '15 to 20 years', 'value': '15 to 20 years'},
                        {'label': 'Above 20 years', 'value': 'Above 20 years'}
                    ], value='New construction'),
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
                    dcc.Dropdown(id='bathrooms-dropdown', options=[
                        {'label': 1, 'value': 1},
                        {'label': 2, 'value': 2},
                        {'label': 3, 'value': 3},
                        {'label': 4, 'value': 4},
                        {'label': 5, 'value': 5},
                        {'label': 6, 'value': 6},
                        {'label': 7, 'value': 7},
                        {'label': '7+', 'value': 10}
                    ], value=BATHROOMS[0]),
                    html.Br(),

                    # Balconies Dropdown
                    dbc.Label("Balconies", html_for="balconies-dropdown"),
                    dcc.Dropdown(id='balconies-dropdown', options=[
                        {'label': '1.0', 'value': '1.0'},
                        {'label': '2.0', 'value': '2.0'},
                        {'label': '3.0', 'value': '3.0'},
                        {'label': '3+', 'value': '3+'}
                    ], value=BALCONIES[0]),
                    html.Br(),

                    # Binary Radio Items

                    dbc.Label("Need House Help Room?"),
                    dbc.RadioItems(id='house-help-radio', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0, inline=True),
                    html.Br(),

                    dbc.Label("Need Store Room?"),
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
                    ]),
                    # Analytics Visualizations Card
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("📊 Analytics Dashboard", className="mb-0",
                                style={'color': '#ffa502', 'fontWeight': '600'})
                        ]),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab(label="Price Comparison", tab_id="price-tab"),
                                dbc.Tab(label="Covered Area Comparison", tab_id="features-tab"),
                                dbc.Tab(label="Common Amenities Cloud", tab_id="amenities-tab")
                            ], id="analytics-tabs", active_tab="price-tab"),
                            html.Div(id="analytics-content", style={'marginTop': '20px'})
                        ])
                    ], style={'marginTop': '20px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.3)'})
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

@app.callback(
    Output('analytics-content', 'children'),
    [Input('analytics-tabs', 'active_tab'),
     Input('submit-button', 'n_clicks')],
    [State('city-dropdown', 'value'),
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
     State('store-room-radio', 'value')]
)
def update_analytics_content(active_tab, n_clicks, city, area, prop_type, trans_type, lifespan, commercial, covered_area, bedrooms, bathrooms, balconies, house_help, store_room):
    if n_clicks == 0 or not city or not area:
        return html.P("Please run the 'PRICE ANALYSIS' model first to see analytics of the selected region.")
    
    # Get predicted price (simplified - you might want to store this globally)
    try:
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
        pred = price_model.predict(input_full)
        predicted_price = float(np.exp(pred[0]))
        
        # Get all societies in selected area for wordcloud
        candidate_map = society_location_map[
            (society_location_map['City'] == city) &
            (society_location_map['Area'] == area)
        ]
        candidate_list = candidate_map['Society'].tolist()
        
    except:
        predicted_price = 1.0
        candidate_list = []
    
    if active_tab == "price-tab":
        return create_price_kde(city, area, predicted_price)
    elif active_tab == "features-tab":
        user_features = {
            'Covered Area': covered_area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Balconies': balconies
        }
        return create_feature_comparison(city, area, user_features)
    elif active_tab == "amenities-tab":
        return create_amenity_wordcloud(candidate_list)
    
    return html.P("Select a tab to view analytics.")

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
            html.H3(f"Predicted Price: ₹ {point_pred:.2f} Crores", className="text-success"),
            html.P(f"95% Confidence Range: ₹ {lower_bound:.2f} Cr - ₹ {upper_bound:.2f} Cr")
        ])
        
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