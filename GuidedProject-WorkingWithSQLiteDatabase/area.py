import sqlite3
import pandas as pd
conn = sqlite3.connect('factbook.db')

query = 'SELECT SUM(area_land) FROM facts WHERE area_land != "";'
query2 = 'SELECT SUM(area_water) FROM facts WHERE area_land != "";'

area_land = conn.execute(query).fetchall()

area_water = conn.execute(query2).fetchall()

print(area_land)
print(area_water)

ratio = float(area_land[0][0])/float(area_water[0][0])

print("Ratio: ", ratio)