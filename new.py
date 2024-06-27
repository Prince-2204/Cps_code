import pandas as pd
import time
import pickle
import numpy as np
import sklearn
import streamlit as st
from sklearn.decomposition import PCA
import joblib
loaded_pipeline = joblib.load('preprocessing_pipeline.pkl')
model = joblib.load('random_forest_model_with_spca.pkl')

st.markdown("<h2>VOC Based Ripening Stage Of Banana</h2>", unsafe_allow_html=True)

prediction_box = st.empty()
class_box = st.empty()
arr_box = st.empty()
shape_box = st.empty()

import ast


def reorder_columns(df):
    try:
        a.drop('timestamp', axis='columns', inplace=True)
        a.drop('Unnamed: 0', axis='columns', inplace=True)

    except Exception as e:
        print('...')
    new_column_order = ['MQ2', 'MQ3', 'MQ4', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135']
    assert set(new_column_order) == set(df.columns), "Invalid column order"
    reordered_df = df[new_column_order]
    print(reordered_df)

    return reordered_df


def extract1(dict_string):
    try:
        # Safely evaluate the string as a Python literal (dictionary)
        dictionary = ast.literal_eval(dict_string)
        # Extract the value associated with the key 'N'
        value = dictionary.get('N', None)
        # Convert the value to an integer (if it's a valid integer)
        if value is not None:
            try:
                numeric_value = int(value)
                return numeric_value
            except ValueError:
                return None
        else:
            return None
    except (ValueError, SyntaxError):
        # Handle errors in case the string is not a valid dictionary
        return None


st.markdown(
    """
    <style>
    body {
        background-color: #222;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#
# Set the members of the team
members = ["Rahul Kumar Under The Guidance Of Dr. N.S. Rajput Sir (Associate Professor)"]

# Display the members in the bottom rightmost corner
st.sidebar.title("Experiment Done By")
for member in members:
    st.sidebar.write(member)


# Function to read the CSV file from the S3 bucket URL
def read_csv_from_s3(url):
    df = pd.read_csv(url)
    return df


# Set the URL of the CSV file in the S3 bucket
csv_url = "https://cloudbucketnsr.s3.eu-north-1.amazonaws.com/dynamodb_data.csv"
lr = 0

while True:
    # Wait for 30 seconds
    time.sleep(5)
    # Read the CSV file from the S3 bucket URL
    try:
        # Attempt to read the CSV file from the provided URL
        dff = pd.read_csv(csv_url)
        dff = dff.sort_values(by='timestamp')
        a = dff
        for column in a.columns:
            a[column] = a[column].apply(extract1)
        a = reorder_columns(a)
        # Clear the previous data and display the updated data

        arr = np.array(a.iloc[[-1]])
        df = pd.DataFrame(arr, columns=['Sensor1', 'Sensor2', 'Sensor3','Sensor4','Sensor5','Sensor6','Sensor7','Sensor8'])
        pred_pca = loaded_pipeline.transform(df)
        prediction = model.predict(pred_pca)
        strr = ""
        if prediction[0] == 0:
            strr = "Overripen"
        if prediction[0] == 1:
            strr = 'Ripen'
        if prediction[0] == 2:
            strr = 'Underripen'

        class_box.markdown(f"""
        <div style="border: 1px solid #000; padding: 10px; width: 100%; height: 100%; margin: 0 auto;">
        <font size="800"> {strr}</font>
        </div>
        """, unsafe_allow_html=True)

        rr = a.shape[0]
        if lr != rr:
            lr = a.shape[0]

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")