import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sqlite3
import os

# Make Database
database_name = 'Portfolio.db'
con = sqlite3.connect(os.path.join('Data', database_name))
cur = con.cursor()

res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
table_names = res.fetchall()

if ~np.isin('selected_contracts', table_names):
    print("CREATE NEW DATABASE TABLE")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS selected_contracts(
                date TEXT NOT NULL,
                contract TEXT NOT NULL,
                strategy TEXT NOT NULL,
                allocated INTEGER NOT NULL,
                action INTEGER NOT NULL,
                PRIMARY KEY (date, contract, strategy))""")
    con.commit()
else:
    print("DATABASE TABLE ALREADY EXISTS")
    con.commit()

if ~np.isin('trades_info', table_names):
    print("CREATE NEW DATABASE TABLE")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS trades_info(
                date TEXT NOT NULL,
                contract TEXT NOT NULL,
                strategy TEXT NOT NULL,
                allocated REAL NOT NULL,
                ave_buy_p REAL,
                buy_time TEXT,
                ave_sell_p REAL,
                sell_time TEXT,
                position INTEGER,
                return REAL,
                profit REAL,
                PRIMARY KEY (date, contract, strategy))""")
    con.commit()
else:
    print("DATABASE TABLE ALREADY EXISTS")
    con.commit()

cur.close()
con.close()

# Make Database
database_name = 'Intraday.db'
con = sqlite3.connect(os.path.join('Data', database_name))
cur = con.cursor()

res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
table_names = res.fetchall()

if ~np.isin('trades_info', table_names):
    print("CREATE NEW DATABASE TABLE")
    
    cur.execute("""CREATE TABLE IF NOT EXISTS trades_info(
                date TEXT NOT NULL,
                contract TEXT NOT NULL,
                strategy TEXT NOT NULL,
                allocated REAL NOT NULL,
                ave_buy_p REAL,
                buy_time TEXT,
                ave_sell_p REAL,
                sell_time TEXT,
                position INTEGER,
                return REAL,
                profit REAL,
                PRIMARY KEY (date, contract, strategy))""")
    con.commit()
else:
    print("DATABASE TABLE ALREADY EXISTS")
    con.commit()

cur.close()
con.close()