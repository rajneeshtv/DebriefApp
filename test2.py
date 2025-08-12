import sqlite3

conn = sqlite3.connect("debrief_app.db")
c = conn.cursor()
c.execute("ALTER TABLE tasks ADD COLUMN status TEXT;")
conn.commit()
conn.close()