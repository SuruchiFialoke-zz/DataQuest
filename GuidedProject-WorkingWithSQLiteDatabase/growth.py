import sqlite3
import pandas as pd
import math
conn = sqlite3.connect('factbook.db')

query = 'SELECT * FROM facts ORDER BY area_land ASC;'

df = pd.read_sql_query(query, conn)

print(df.columns)
print(df.shape)
print(df.info())

# Notice there are only 236 rows that have non-null population_growth and 242 in population
# Drop rows with not null population_growth and popolation

df = df.dropna(subset= ['population_growth', 'population'])
print(df.shape)

def predict_population(row):
    A = float(row['population'])
    x = float(row['population_growth'])
    N =  A * math.e** (x*0.35)
    return (round(N,2))
    
# Create a column in the dataframe
# With 2050 population
df['population_2050'] = df.apply(predict_population, axis=1)

# Sort values by population_2050
df.sort_values('population_2050', ascending = False, inplace=True)

print(df['name'].head(10))

