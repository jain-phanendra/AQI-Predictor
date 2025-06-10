import pandas as pd
from datetime import datetime

# Step 1: Load dataset
df = pd.read_csv("Karve Road.csv")
#df = pd.read_csv("Eco herbal park Moradabad 2023.csv")
#df = pd.read_csv("ITO, Delhi.csv")
#df = pd.read_csv("Knowledge Park   III, Greater Noida.csv")
#df = pd.read_csv("Knowledge Park   V, Greater Noida.csv")
#df = pd.read_csv("Lalbagh, Lucknow.csv")
#df = pd.read_csv("Loni, Ghaziabad.csv")
#df = pd.read_csv("Sanjay Nagar, Ghaziabad.csv")
#df = pd.read_csv("Shahjahan Garden, Agra.csv")
#df = pd.read_csv("Talkatora District Industries Center, Lucknow.csv")

# Step 2: Convert to datetime
df['From Date'] = pd.to_datetime(df['From Date'], errors='coerce')
df['Date'] = df['From Date'].dt.date

# Step 3: Filter for 2021 onwards
df = df[df['Date'] >= datetime(2021, 1, 1).date()]

# Step 4: Calculate daily average of pollutants
pollutants = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']
df_daily = df.groupby('Date')[pollutants].mean().reset_index()

# Step 5: AQI sub-index functions
def calculate_aqi_pm25(x):
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 49 / 30
    elif x <= 90: return 100 + (x - 60) * 99 / 30
    elif x <= 120: return 200 + (x - 90) * 99 / 30
    elif x <= 250: return 300 + (x - 120) * 99 / 130
    else: return 400 + (x - 250) * 100 / 100

def calculate_aqi_pm10(x):
    if x <= 50: return x
    elif x <= 100: return 50 + (x - 50) * 49 / 50
    elif x <= 250: return 100 + (x - 100) * 99 / 150
    elif x <= 350: return 200 + (x - 250) * 99 / 100
    elif x <= 430: return 300 + (x - 350) * 99 / 80
    else: return 400 + (x - 430) * 99 / 170

def calculate_aqi_no2(x):
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 49 / 40
    elif x <= 180: return 100 + (x - 80) * 99 / 100
    elif x <= 280: return 200 + (x - 180) * 99 / 100
    elif x <= 400: return 300 + (x - 280) * 99 / 120
    else: return 400 + (x - 400) * 99 / 125

def calculate_aqi_so2(x):
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 49 / 40
    elif x <= 380: return 100 + (x - 80) * 99 / 300
    elif x <= 800: return 200 + (x - 380) * 99 / 420
    elif x <= 1600: return 300 + (x - 800) * 99 / 800
    else: return 400 + (x - 1600) * 99 / 400

def calculate_aqi_co(x):
    if x <= 1: return x * 50
    elif x <= 2: return 50 + (x - 1) * 49
    elif x <= 10: return 100 + (x - 2) * 99 / 8
    elif x <= 17: return 200 + (x - 10) * 99 / 7
    else: return 400 + (x - 17) * 99 / 5

def calculate_aqi_o3(x):
    if x <= 50: return x
    elif x <= 100: return 50 + (x - 50) * 49 / 50
    elif x <= 168: return 100 + (x - 100) * 99 / 68
    elif x <= 208: return 200 + (x - 168) * 99 / 40
    elif x <= 748: return 300 + (x - 208) * 99 / 540
    else: return 400 + (x - 748) * 99 / 252

# Step 6: Apply AQI calculation using average
df_daily['AQI_PM2.5'] = df_daily['PM2.5 (ug/m3)'].apply(lambda x: calculate_aqi_pm25(x) if pd.notna(x) else None)
df_daily['AQI_PM10'] = df_daily['PM10 (ug/m3)'].apply(lambda x: calculate_aqi_pm10(x) if pd.notna(x) else None)
df_daily['AQI_NO2'] = df_daily['NO2 (ug/m3)'].apply(lambda x: calculate_aqi_no2(x) if pd.notna(x) else None)
df_daily['AQI_SO2'] = df_daily['SO2 (ug/m3)'].apply(lambda x: calculate_aqi_so2(x) if pd.notna(x) else None)
df_daily['AQI_CO'] = df_daily['CO (mg/m3)'].apply(lambda x: calculate_aqi_co(x) if pd.notna(x) else None)
df_daily['AQI_O3'] = df_daily['Ozone (ug/m3)'].apply(lambda x: calculate_aqi_o3(x) if pd.notna(x) else None)

# Final AQI using average of sub-indices
df_daily['AQI'] = df_daily[['AQI_PM2.5', 'AQI_PM10', 'AQI_NO2', 'AQI_SO2', 'AQI_CO', 'AQI_O3']].mean(axis=1)

# Step 7: Prepare for pivot
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily['Day'] = df_daily['Date'].dt.day
df_daily['Month_Year'] = df_daily['Date'].dt.strftime('%B %Y')

# Step 8: Create pivot table
pivot_df = df_daily.pivot(index='Day', columns='Month_Year', values='AQI')

# Sort columns chronologically
pivot_df = pivot_df.reindex(sorted(pivot_df.columns, key=lambda x: datetime.strptime(x, '%B %Y')), axis=1)

# Reset index to keep 'Day' as a column
pivot_df.reset_index(inplace=True)

# Step 9: Preview output
print(pivot_df.head())

#pivot_df.to_csv("Buddhi vihar 2023.csv", index=False)
#pivot_df.to_csv("herbal park .csv", index=False)
#pivot_df.to_csv("ITO.csv", index=False)
#pivot_df.to_csv("Knowledge Park   III.csv", index=False)
#pivot_df.to_csv("Knowledge Park   V.csv", index=False)
#pivot_df.to_csv("Lalbagh.csv", index=False)
pivot_df.to_csv("Karve_Road.csv", index=False)
#pivot_df.to_csv("Sanjay Nagar.csv", index=False)
#pivot_df.to_csv("Shahjahan Garden.csv", index=False)
#pivot_df.to_csv("Talkatora District Industries Center.csv", index=False)

print("Done")