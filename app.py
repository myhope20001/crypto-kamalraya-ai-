import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import time
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Crypto Trader", layout="wide")

DB = "ai_trade.db"

conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()

# ------------------------
# DB
# ------------------------

cur.execute("""
CREATE TABLE IF NOT EXISTS trades(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 time TEXT,
 ticker TEXT,
 price REAL,
 qty REAL,
 side TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS learning(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 rsi REAL,
 ma_gap REAL,
 volume REAL,
 momentum REAL,
 target INTEGER
)
""")

conn.commit()

# ------------------------
# wallet
# ------------------------

if "wallet" not in st.session_state:

    st.session_state.wallet = {
        "krw": 10000000.0,
        "positions": {}
    }

wallet = st.session_state.wallet

# ------------------------
# indicators
# ------------------------

def indicators(df):

    df["ma5"] = df.close.rolling(5).mean()
    df["ma20"] = df.close.rolling(20).mean()

    delta = df.close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    rs = up.rolling(14).mean() / down.rolling(14).mean()

    df["rsi"] = 100 - (100/(1+rs))

    df["ma_gap"] = (df.ma5 - df.ma20) / df.ma20

    df["momentum"] = df.close.pct_change(3)

    return df

# ------------------------
# tradable coins
# ------------------------

def tradable_coins():

    url = "https://api.upbit.com/v1/market/all?isDetails=true"

    try:
        res = requests.get(url)
        markets = res.json()
    except:
        return []

    coins = []

    for m in markets:

        ticker = m["market"]

        if not ticker.startswith("KRW-"):
            continue

        if m["market_warning"] == "CAUTION":
            continue

        try:

            df = pyupbit.get_ohlcv(ticker, interval="day", count=10)

            if df is None:
                continue

            if len(df) < 7:
                continue

        except:
            continue

        coins.append(ticker)

    return coins

# ------------------------
# Camarilla
# ------------------------

def camarilla_levels(ticker):

    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)

    if df is None or len(df) < 2:
        return None

    prev = df.iloc[-2]

    high = prev.high
    low = prev.low
    close = prev.close

    diff = high - low

    r3 = close + diff * 1.1 / 4
    r4 = close + diff * 1.1 / 2
    r5 = close + diff * 1.1

    s4 = close - diff * 1.1 / 2

    return r3,r4,r5,s4

# ------------------------
# learning data
# ------------------------

def build_learning_data(ticker):

    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=500)

    if df is None:
        return

    df = indicators(df)

    df["target"] = (df.close.shift(-3) > df.close).astype(int)

    df = df.dropna()

    for _,r in df.iterrows():

        cur.execute("INSERT INTO learning VALUES(NULL,?,?,?,?,?)",
                    (r.rsi,r.ma_gap,r.volume,r.momentum,r.target))

    conn.commit()

# ------------------------
# train model
# ------------------------

def train_model():

    df = pd.read_sql("SELECT * FROM learning", conn)

    if len(df) < 200:
        return None

    X = df[["rsi","ma_gap","volume","momentum"]]
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train,y_train)

    return model

# ------------------------
# top30
# ------------------------

def top30():

    tickers = tradable_coins()

    data = []

    for t in tickers[:60]:

        try:

            df = pyupbit.get_ohlcv(t, interval="minute1", count=30)

            if df is None:
                continue

            value = (df.close * df.volume).sum()

            if value < 1000000000:
                continue

            data.append((t,value))

        except:
            continue

    data = sorted(data,key=lambda x:x[1],reverse=True)

    return [d[0] for d in data[:30]]

# ------------------------
# features
# ------------------------

def features(ticker):

    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=50)

    if df is None:
        return None

    df = indicators(df)

    r = df.iloc[-1]

    vals = [r.rsi,r.ma_gap,r.volume,r.momentum]

    if any(pd.isna(vals)):
        return None

    return vals

# ------------------------
# AI rank
# ------------------------

def ai_rank(model):

    coins = top30()

    picks = []

    for t in coins:

        f = features(t)

        if f is None:
            continue

        X = pd.DataFrame([f],columns=["rsi","ma_gap","volume","momentum"])

        prob = model.predict_proba(X)[0][1]

        picks.append((t,prob))

    picks = sorted(picks,key=lambda x:x[1],reverse=True)

    return picks

# ------------------------
# training button
# ------------------------

if st.button("AI 학습 데이터 업데이트"):

    coins = top30()

    for c in coins:
        build_learning_data(c)

    st.success("학습 완료")

# ------------------------
# model
# ------------------------

model = train_model()

# ------------------------
# trading
# ------------------------

if model:

    ranked = ai_rank(model)

    # BUY
    for coin,prob in ranked:

        if prob < 0.65:
            continue

        if coin in wallet["positions"]:
            continue

        if wallet["krw"] < 5000:
            break

        price = pyupbit.get_current_price(coin)

        if price is None:
            continue

        buy_amt = min(wallet["krw"],1000000)

        qty = buy_amt / price

        wallet["krw"] -= buy_amt

        wallet["positions"][coin] = {
            "qty":qty,
            "buy_price":price,
            "sold_r3":False,
            "sold_r4":False,
            "sold_r5":False
        }

        cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
                    (datetime.now(),coin,price,qty,"BUY"))

        conn.commit()

    # SELL
    for coin in list(wallet["positions"].keys()):

        pos = wallet["positions"][coin]

        price = pyupbit.get_current_price(coin)

        if price is None:
            continue

        levels = camarilla_levels(coin)

        if levels is None:
            continue

        r3,r4,r5,s4 = levels

        f = features(coin)

        ai_prob = None

        if f is not None:

            X = pd.DataFrame([f],columns=["rsi","ma_gap","volume","momentum"])

            ai_prob = model.predict_proba(X)[0][1]

        # stop loss
        if price <= pos["buy_price"]*0.97 or price <= s4 or (ai_prob and ai_prob < 0.4):

            sell_qty = pos["qty"]

            wallet["krw"] += sell_qty * price

            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
                        (datetime.now(),coin,price,sell_qty,"STOP_LOSS"))

            del wallet["positions"][coin]

            conn.commit()

            continue

        # R3
        if price >= r3 and not pos["sold_r3"]:

            sell_qty = pos["qty"]*0.4

            wallet["krw"] += sell_qty * price

            pos["qty"] -= sell_qty

            pos["sold_r3"] = True

            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
                        (datetime.now(),coin,price,sell_qty,"SELL_R3"))

            conn.commit()

        # R4
        if price >= r4 and not pos["sold_r4"]:

            sell_qty = pos["qty"]*0.3

            wallet["krw"] += sell_qty * price

            pos["qty"] -= sell_qty

            pos["sold_r4"] = True

            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
                        (datetime.now(),coin,price,sell_qty,"SELL_R4"))

            conn.commit()

        # R5
        if price >= r5 and not pos["sold_r5"]:

            sell_qty = pos["qty"]

            wallet["krw"] += sell_qty * price

            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
                        (datetime.now(),coin,price,sell_qty,"SELL_R5"))

            del wallet["positions"][coin]

            conn.commit()

# ------------------------
# dashboard
# ------------------------

positions = wallet["positions"]

coin_value = 0

rows = []

for coin,pos in positions.items():

    price = pyupbit.get_current_price(coin)

    if price is None:
        continue

    value = pos["qty"] * price

    coin_value += value

    profit = (price-pos["buy_price"])/pos["buy_price"]*100

    rows.append({
        "ticker":coin,
        "qty":pos["qty"],
        "buy_price":pos["buy_price"],
        "current_price":price,
        "profit%":profit,
        "value":value
    })

asset = wallet["krw"] + coin_value

st.title("AI Crypto Multi Trader")

c1,c2,c3 = st.columns(3)

c1.metric("총 자산", f"{asset:,.0f} 원")
c2.metric("현금", f"{wallet['krw']:,.0f} 원")
c3.metric("코인 평가", f"{coin_value:,.0f} 원")

st.subheader("보유 코인")

if rows:
    st.dataframe(pd.DataFrame(rows))
else:
    st.write("보유 코인 없음")

# ------------------------
# 최근 거래
# ------------------------

st.subheader("최근 거래")

hist2 = pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 20", conn)

st.dataframe(hist2)

# ------------------------
# auto refresh
# ------------------------

time.sleep(300)

st.rerun()
