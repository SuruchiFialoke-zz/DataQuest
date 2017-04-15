import sqlite3
conn = sqlite3.connect('factbook.db')

c = conn.cursor()
query = 'SELECT * FROM facts ORDER BY area_land ASC LIMIT 10;'
c.execute(query)

print(c.fetchall())