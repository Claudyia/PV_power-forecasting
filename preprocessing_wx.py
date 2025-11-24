#attività di preproccesing per wx_dataset.xlsx
import pandas as pd
# Carica il dataset
df = pd.read_excel('wx_dataset.xlsx')


# Converti la colonna 'date' dal formato ISO al formato  mm/gg/aa hh:mm
def  to_european_datetime(value):
    timestamp = pd.to_datetime(value, errors='coerce')
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.strftime('%m/%d/%y %H:%M')

df['dt_iso'] = df['dt_iso'].apply(to_european_datetime)

#cambia il nome della colonna dt_iso in date
df = df.rename(columns={'dt_iso': 'date'})

#normalizzare  tutte le colonne numeriche tranne lat e lon
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist() 
for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val)

# categorizzare la colonna weather_description in base a certe parole chiave
def categorize_weather(description):
    description = description.lower()
    if 'broken' in description or 'overcast' in description:
        return 'cloudy'
    elif 'scattered' in description or 'few' in description:
        return 'partly cloudy'
    elif 'light rain' in description or 'moderate rain' in description:
        return 'rain'
    elif 'clear sky' in description:
        return 'clear'
    else:
        return 'other'
        
df['weather_category'] = df['weather_description'].apply(categorize_weather)

# One-hot encoding della colonna 'weather_category'
weather_dummies = pd.get_dummies(df['weather_category'], dtype=int, prefix='weather')
df = pd.concat([df, weather_dummies], axis=1)

# drop delle colonne non necessarie ( lat, lon, dew_point, pressure)
df = df.drop(columns=['lat', 'lon', 'dew_point', 'pressure', 'weather_description', 'weather_category'])

print(df.head())

#riempire valorino nullii con zeri
df = df.fillna(0)


# Salva il dataset preprocessato in un nuovo file Excel
df.to_excel('wx_dataset_preprocessed.xlsx', index=False)