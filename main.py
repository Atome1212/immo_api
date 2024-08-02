import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import base64
import xgboost
import logging

logging.basicConfig(level=logging.INFO)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("./css/style.css")

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style> 
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("./data/bg/background.png")

def load_file(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return False

def handle_categorical_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def find_latest_model_path(model_dir):
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        return False
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def test_model(input_data, model_path, scaler, columns):
    model = joblib.load(model_path)
    X_test = prepare_data(input_data, scaler, columns)
    y_pred = model.predict(X_test)
    return y_pred

def prepare_data(input_data, scaler, columns):
    df = pd.DataFrame([input_data])
    df = handle_categorical_data(df)
    df = df.reindex(columns=columns, fill_value=0)
    df = scaler.transform(df)
    return df

cleaned_data_path = "./data/cleaned_data.csv"
df = load_file(cleaned_data_path)
if df is False:
    st.error("Failed to load cleaned data file.")
    st.stop()

X = df.drop(columns=['Price'])
X = handle_categorical_data(X)
columns = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

model_path = find_latest_model_path("./data/")
if not model_path:
    st.error("No model found.")
    st.stop()

st.markdown('<div class="custom-container">', unsafe_allow_html=True)
st.markdown('<div class="custom-title" style="font-size: 36px;">🏡 Real Estate Price Prediction 🏡</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle" style="font-size: 24px;">Enter the property characteristics to get an estimated price:</div>', unsafe_allow_html=True)

initial_states = {
    'postal_code': 0,
    'bathroom_count': 0,
    'bedroom_count': 0,
    'construction_year': 2024,
    'number_of_facades': 0,
    'peb': 'B',
    'surface_of_plot': 0,
    'living_area': 0,
    'garden_area': 0,
    'state_of_building': 'AS_NEW',
    'swimming_pool': False,
    'terrace': False,
    'Fireplace': False,
    'Furnished': False,
    'toilet_count': 0,
    'room_count': 0,
    #'subtype_of_property': 'house'
}

for key, value in initial_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

state_of_building_mapping = {
    'Good': 'GOOD',
    'To be done up': 'TO_BE_DONE_UP',
    'As new': 'AS_NEW',
    'To renovate': 'TO_RENOVATE',
    'To restore': 'TO_RESTORE',
    'Just renovated': 'JUST_RENOVATED'
}

with st.form("prediction_form"):
    with st.expander("Basic Information"):
        postal_code = st.number_input("Postal Code", min_value=0, step=1, value=st.session_state['postal_code'])
        construction_year = st.selectbox("Year of Construction", list(range(1800, 2035)), index=st.session_state['construction_year'] - 1800)
        peb = st.selectbox("Energy Efficiency (PEB)", ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'], index=['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'].index(st.session_state['peb']))
        #subtype_of_property = st.selectbox("Subtype Of Property", ['Apartment', 'Apartment Block', 'Bungalow', 'Castle', 'Chalet', 'Country Cottage', 'Duplex', 'Exceptional Property', 'Farmhouse', 'Flat Studio', 'Ground Floor', 'House', 'Kot', 'Loft', 'Manor House', 'Mansion', 'Mixed Use Building', 'Other Property', 'Penthouse', 'Pavilion', 'Service Flat', 'Show House', 'Town House', 'Triplex', 'Villa'], index=['apartment',  'apartment_block', 'bungalow', 'castle', 'chalet', 'country_cottage', 'duplex', 'exceptional_property', 'farmhouse', 'flat_studio', 'ground_floor', 'house', 'kot', 'loft', 'manor_house', 'mansion', 'mixed_use_building', 'other_property', 'penthouse', 'pavilion', 'service_flat', 'show_house', 'town_house', 'triplex', 'villa'].index(st.session_state['subtype_of_property']))   
        state_of_building = st.selectbox("State of Building", list(state_of_building_mapping.keys()), index=list(state_of_building_mapping.keys()).index(next(key for key, value in state_of_building_mapping.items() if value == st.session_state['state_of_building'])))
       
    with st.expander("Property Size"):
        surface_of_plot = st.number_input("Plot Area (m²)", min_value=0, step=1, value=st.session_state['surface_of_plot'])
        living_area = st.number_input("Living Area (m²)", min_value=0, step=1, value=st.session_state['living_area'])
        garden_area = st.number_input("Garden Area (m²)", min_value=0, step=1, value=st.session_state['garden_area'])

    with st.expander("Rooms and Facilities"):
        bathroom_count = st.slider("Number of Bathrooms", min_value=0, max_value=10, value=st.session_state['bathroom_count'])
        bedroom_count = st.slider("Number of Bedrooms", min_value=0, max_value=10, value=st.session_state['bedroom_count'])
        number_of_facades = st.slider("Number of Facades", min_value=0, max_value=4, value=st.session_state['number_of_facades'])
        toilet_count = st.slider("Number of Toilets", min_value=0, max_value=10, value=st.session_state['toilet_count'])
        room_count = st.slider("Number of Rooms", min_value=0, max_value=30, value=st.session_state['room_count'])
        swimming_pool = st.checkbox("Swimming Pool", value=st.session_state['swimming_pool'])
        terrace = st.checkbox("Terrace", value=st.session_state['terrace'])
        Fireplace = st.checkbox("Fireplace", value=st.session_state['Fireplace'])
        Furnished = st.checkbox("Furnished", value=st.session_state['Furnished'])
        
    submit_button = st.form_submit_button(label="Predict Price 💰")

st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    st.session_state['postal_code'] = postal_code
    st.session_state['bathroom_count'] = bathroom_count
    st.session_state['bedroom_count'] = bedroom_count
    st.session_state['construction_year'] = construction_year
    st.session_state['number_of_facades'] = number_of_facades
    st.session_state['peb'] = peb
    #st.session_state['subtype_of_property'] = subtype_of_property
    st.session_state['surface_of_plot'] = surface_of_plot
    st.session_state['living_area'] = living_area
    st.session_state['garden_area'] = garden_area
    st.session_state['state_of_building'] = state_of_building_mapping[state_of_building]
    st.session_state['swimming_pool'] = swimming_pool
    st.session_state['terrace'] = terrace
    st.session_state['Fireplace'] = Fireplace
    st.session_state['Furnished'] = Furnished
    st.session_state['toilet_count'] = toilet_count
    st.session_state['room_count'] = room_count

    input_data = {
        "PostalCode": postal_code,
        "BathroomCount": bathroom_count,
        "BedroomCount": bedroom_count,
        "ConstructionYear": construction_year,
        "NumberOfFacades": number_of_facades,
        "SurfaceOfPlot": surface_of_plot,
        "LivingArea": living_area,
        "GardenArea": garden_area,
        "SwimmingPool": int(swimming_pool),
        "Terrace": int(terrace),
        "Fireplace": int(Fireplace),
        "Furnished": int(Furnished),
        "ToiletCount": toilet_count,
        "RoomCount": room_count,
        "PEB": f"PEB_{peb}",
        #"subtype_of_property": f"subtype_of_property_{subtype_of_property}",
        "StateOfBuilding": f"StateOfBuilding_{state_of_building_mapping[state_of_building]}"
    }

    y_pred = test_model(input_data, model_path, scaler, columns)

    st.markdown(f'<div class="custom-success" style="font-size: 24px;">The predicted price is: {y_pred[0]:.2f} € 🏷️</div>', unsafe_allow_html=True)
