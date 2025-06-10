# This Project is created By Phanendra Jain
# this is is final year Project
# this project is created to predict the Air Quality Index in india
# this project will predict the AQI of 14 locations based on 3 states in Bharat 
# The dataset are from 2 sources
# here are the links of the dataset 
# CPCB : https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/aqi-repository
# Kaggle : https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023
# the data is starting from 2021 to 2024

from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
import os

app = Flask(__name__)
CORS(app)

def get_aqi_level_info(aqi):
    try:
        aqi = float(aqi)
    except (ValueError, TypeError):
        return "N/A", "Invalid AQI value."

    if aqi <= 50:
        return "Good", "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."

def get_aqi_data(location):
    if location == 'Herbal Park':
        filepath = 'data/Herbal_Park_Moradabad_2024.csv'
    elif location == 'Borivali':
        filepath = 'data/Borivali_East_Mumbai_MPCB_2024.csv'
    elif location == 'ITO':
        filepath = 'data/ITO_Delhi_CPCB_2024.csv'
    elif location == 'Buddhi Vihar':
        filepath = 'data/Buddhi_Vihar_Moradabad_2024.csv'
    elif location == 'Andheri':
        filepath = 'data/Chakala-Andheri_East_Mumbai_2024.csv'
    elif location == 'Knowledge Park-III':
        filepath = 'data/Knowledge_Park_III_Greater_Noida_2024.csv'
    elif location == 'Knowledge Park-V':
        filepath = 'data/Knowledge_Park_V_Greater_Noida_2022.csv'
    elif location == 'Lalbagh':
        filepath = 'data/Lalbagh_Lucknow_2024.csv'
    elif location == 'Loni':
        filepath = 'data/Loni_Ghaziabad_2024.csv'
    elif location == 'Rohini':
        filepath = 'data/Rohini_Delhi_2024.csv'
    elif location == 'Sanjay Nagar':
        filepath = 'data/Sanjay_Nagar_Ghaziabad_2024.csv'
    elif location == 'Shahjahan Garden':
        filepath = 'data/Shahjahan_Garden_Agra_2024.csv'
    elif location == 'Talkatora District Industries Center':
        filepath = 'data/Talkatora_District_Industries_Center_Lucknow_2024.csv'
    elif location == 'Karve Road':
        filepath = 'data/Karve_Road.csv'
    else:
        return None

    if not os.path.exists(filepath):
        return None

    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.json.get('location')
        selected_date_str = request.json.get('date')

        data = get_aqi_data(location)
        if data is None or data.empty:
            return jsonify({'status': 'error', 'message': 'Location data not found or empty'})

        # Ensure Day is properly named
        data.rename(columns={data.columns[0]: 'Day'}, inplace=True)
        data.replace("NA", pd.NA, inplace=True)

        for col in data.columns[1:]:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].mean())

        data_long = pd.melt(data, id_vars=['Day'], var_name='Month_Year', value_name='AQI')
        data_long[['Month', 'Year']] = data_long['Month_Year'].str.extract(r'([A-Za-z]+)_(\d{4})')

        # Ensure Day column is clean
        data_long['Day'] = pd.to_numeric(data_long['Day'], errors='coerce')
        data_long.dropna(subset=['Day', 'Month', 'Year'], inplace=True)

        month_map = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        data_long['Month_Num'] = data_long['Month'].map(month_map)

        data_long['Date'] = pd.to_datetime(
            data_long['Year'] + '-' + data_long['Month_Num'] + '-' + data_long['Day'].astype(int).astype(str).str.zfill(2),
            errors='coerce'
        )

        final_data = data_long[['Date', 'AQI']].dropna().sort_values('Date').reset_index(drop=True)
        final_data = final_data.rename(columns={'Date': 'ds', 'AQI': 'y'})

        if len(final_data) < 2:
            return jsonify({'status': 'error', 'message': 'Not enough data to train the model after cleaning.'})

        m = Prophet()
        m.fit(final_data)

        selected_date = pd.to_datetime(selected_date_str, errors='coerce')
        if pd.isna(selected_date):
            return jsonify({'status': 'error', 'message': 'Invalid selected date'})

        start_date = selected_date - timedelta(days=3)
        end_date = selected_date + timedelta(days=4)

        last_date = final_data['ds'].max()
        days_after = max((end_date - last_date).days, 0)

        future = m.make_future_dataframe(periods=days_after)
        forecast = m.predict(future)

        week_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)].copy()

        if week_forecast.empty:
            return jsonify({'status': 'error', 'message': 'No forecast available for the selected date range.'})

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set dark background colors
        fig.patch.set_facecolor('#121212')  # Dark grey/black background for figure
        ax.set_facecolor('#121212')         # Dark background for axes

        # Create bar plot
        ax.bar(week_forecast['ds'], week_forecast['yhat'], color='red')

        # Customize title and labels with white color
        ax.set_title(f'AQI Forecast for {location}\n{start_date.date()} to {end_date.date()}', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('AQI', color='white')

        # Customize tick colors to white for readability on dark background
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')

        plt.tight_layout()
        fig.savefig('static/forecast.png', facecolor=fig.get_facecolor())  # Save with bg color
        plt.close(fig)

        # Predicted AQI for selected date
        selected_day_prediction = week_forecast[week_forecast['ds'] == selected_date]
        if selected_day_prediction.empty:
            predicted_aqi = None
        else:
            predicted_aqi = round(selected_day_prediction['yhat'].values[0], 2)

        aqi_category, aqi_precaution = get_aqi_level_info(predicted_aqi) if predicted_aqi is not None else ("N/A", "N/A")

        return jsonify({
            'status': 'success',
            'prediction': week_forecast[['ds', 'yhat']].to_dict(orient='records'),
            'graph_url': '/static/forecast.png',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'selected_date': selected_date.strftime('%Y-%m-%d'),
            'predicted_aqi': predicted_aqi,
            'aqi_category': aqi_category,
            'aqi_precaution': aqi_precaution
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
