#attività di preproccesing per pv_dataset.xlsx
import pandas as pd

# Carica il dataset
df = pd.read_excel('pv_dataset.xlsx')

# Converti la colonna 'Max kWp' in 'date' e la colonna 82.41 in 'Production kWh'
df = df.rename(columns={'Max kWp': 'date', 82.41: 'Production kWh'})

 #controllo quanti zeri e a che ora e corrispettivo print
def count_zeros_at_hour(dataframe, hour):
    count = 0
    for index, row in dataframe.iterrows():
        timestamp = pd.to_datetime(row['date'], errors='coerce')
        if pd.isna(timestamp):
            continue
        if timestamp.hour == hour and row['Production kWh'] == 0:
            count += 1
    return count
zeros_at_each_hour = {hour: count_zeros_at_hour(df, hour) for hour in range(24)}
for hour, count in zeros_at_each_hour.items():
    print(f"Hour {hour}: {count} zeros")


#normalizzare  i valori della colonna 'Production kWh'
min_val = df['Production kWh'].min()
max_val = df['Production kWh'].max()
df['Production kWh'] = (df['Production kWh'] - min_val) / (max_val - min_val)

""""per outlier removal, si potrebbe considerare di rimuovere i valori che sono al di fuori di 3 deviazioni standard dalla media.
IDEA DI CHAT NON NE SIAMO SICURE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
mean_val = df['Production kWh'].mean()
std_dev = df['Production kWh'].std()
lower_bound = mean_val - 3 * std_dev
upper_bound = mean_val + 3 * std_dev
df = df[(df['Production kWh'] >= lower_bound) & (df['Production kWh'] <= upper_bound)]
#print dei valori statistici dopo outlier removal
print("Dopo outlier removal:")
print(df['Production kWh'].describe())

#nuovo dataset preprocessato
df.to_excel('pv_dataset_preprocessed.xlsx', index=False)
print(df.head())

