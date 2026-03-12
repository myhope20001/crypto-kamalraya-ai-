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

DB="trade_ai.db"

conn=sqlite3.connect(DB,check_same_thread=False)
cur=conn.cursor()

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

# -----------------------
# wallet
# -----------------------

if "wallet" not in st.session_state:

    st.session_state.wallet={
        "krw":10000000.0,
        "positions":{}
    }

wallet=st.session_state.wallet

MAX_INVEST=5000000

# -----------------------
# indicator
# -----------------------

def indicators(df):

    df["ma5"]=df.close.rolling(5).mean()
    df["ma20"]=df.close.rolling(20).mean()

    delta=df.close.diff()

    up=delta.clip(lower=0)
    down=-delta.clip(upper=0)

    rs=up.rolling(14).mean()/down.rolling(14).mean()

    df["rsi"]=100-(100/(1+rs))

    df["ma_gap"]=(df.ma5-df.ma20)/df.ma20

    df["momentum"]=df.close.pct_change(3)

    return df

# -----------------------
# tradable coins
# -----------------------

def tradable_coins():

    url="https://api.upbit.com/v1/market/all?isDetails=true"

    try:
        res=requests.get(url)
        markets=res.json()
    except:
        return []

    coins=[]

    for m in markets:

        ticker=m.get("market")

        if ticker is None:
            continue

        if not ticker.startswith("KRW-"):
            continue

        if m.get("market_warning")=="CAUTION":
            continue

        coins.append(ticker)

    return coins

# -----------------------
# TOP 거래대금
# -----------------------

def top_coins():

    tickers=tradable_coins()

    data=[]

    for t in tickers[:60]:

        try:

            df=pyupbit.get_ohlcv(t,"minute1",count=30)

            if df is None:
                continue

            value=(df.close*df.volume).sum()

            data.append((t,value))

        except:
            continue

    data=sorted(data,key=lambda x:x[1],reverse=True)

    return [d[0] for d in data[:30]]

# -----------------------
# feature
# -----------------------

def features(ticker):

    df=pyupbit.get_ohlcv(ticker,"minute1",count=50)

    if df is None:
        return None

    df=indicators(df)

    r=df.iloc[-1]

    vals=[r.rsi,r.ma_gap,r.volume,r.momentum]

    if any(pd.isna(vals)):
        return None

    return vals

# -----------------------
# learning data
# -----------------------

def build_learning(ticker):

    df=pyupbit.get_ohlcv(ticker,"minute1",count=400)

    if df is None:
        return

    df=indicators(df)

    df["target"]=(df.close.shift(-3)>df.close).astype(int)

    df=df.dropna()

    for _,r in df.iterrows():

        cur.execute(
        "INSERT INTO learning VALUES(NULL,?,?,?,?,?)",
        (r.rsi,r.ma_gap,r.volume,r.momentum,r.target)
        )

    conn.commit()

# -----------------------
# train model
# -----------------------

def train():

    df=pd.read_sql("SELECT * FROM learning",conn)

    if len(df)<300:
        return None

    X=df[["rsi","ma_gap","volume","momentum"]]
    y=df["target"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    model=RandomForestClassifier(n_estimators=200)

    model.fit(X_train,y_train)

    return model

# -----------------------
# AI rank
# -----------------------

def ai_rank(model):

    coins=top_coins()

    picks=[]

    for t in coins:

        f=features(t)

        if f is None:
            continue

        X=pd.DataFrame([f],columns=["rsi","ma_gap","volume","momentum"])

        prob=model.predict_proba(X)[0][1]

        picks.append((t,prob))

    picks=sorted(picks,key=lambda x:x[1],reverse=True)

    return picks

# -----------------------
# AI 매수 비율
# -----------------------

def buy_ratio(prob):

    if prob>0.85:
        return 1.0
    elif prob>0.75:
        return 0.5
    elif prob>0.65:
        return 0.3
    elif prob>0.55:
        return 0.2
    else:
        return 0

# -----------------------
# AI 매도 비율
# -----------------------

def sell_ratio(prob,profit):

    if prob<0.40:
        return 1

    if profit>5:
        return 0.5

    if profit>10:
        return 1

    return 0

# -----------------------
# 학습 버튼
# -----------------------

if st.button("AI 학습 데이터 생성"):

    coins=top_coins()

    for c in coins:
        build_learning(c)

    st.success("학습 데이터 완료")

# -----------------------
# model
# -----------------------

model=train()

# -----------------------
# trading
# -----------------------

if model:

    ranked=ai_rank(model)

    # 매수
    for coin,prob in ranked:

        ratio=buy_ratio(prob)

        if ratio==0:
            continue

        price=pyupbit.get_current_price(coin)

        if price is None:
            continue

        pos=wallet["positions"].get(coin)

        invest=MAX_INVEST*ratio

        if wallet["krw"]<invest:
            continue

        qty=invest/price

        wallet["krw"]-=invest

        if pos:

            pos["qty"]+=qty
            pos["buy_price"]=(pos["buy_price"]+price)/2

        else:

            wallet["positions"][coin]={
            "qty":qty,
            "buy_price":price
            }

        cur.execute(
        "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
        (datetime.now(),coin,price,qty,"BUY")
        )

        conn.commit()

    # 매도
    for coin in list(wallet["positions"].keys()):

        pos=wallet["positions"][coin]

        price=pyupbit.get_current_price(coin)

        if price is None:
            continue

        profit=(price-pos["buy_price"])/pos["buy_price"]*100

        f=features(coin)

        if f is None:
            continue

        X=pd.DataFrame([f],columns=["rsi","ma_gap","volume","momentum"])

        prob=model.predict_proba(X)[0][1]

        ratio=sell_ratio(prob,profit)

        if ratio==0:
            continue

        sell_qty=pos["qty"]*ratio

        wallet["krw"]+=sell_qty*price

        pos["qty"]-=sell_qty

        cur.execute(
        "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
        (datetime.now(),coin,price,sell_qty,"SELL")
        )

        conn.commit()

        if pos["qty"]<=0:

            del wallet["positions"][coin]

# -----------------------
# dashboard
# -----------------------

coin_value=0

rows=[]

for coin,pos in wallet["positions"].items():

    price=pyupbit.get_current_price(coin)

    if price is None:
        continue

    value=pos["qty"]*price

    coin_value+=value

    profit=(price-pos["buy_price"])/pos["buy_price"]*100

    rows.append({
    "ticker":coin,
    "qty":pos["qty"],
    "buy_price":pos["buy_price"],
    "price":price,
    "profit%":profit,
    "value":value
    })

asset=wallet["krw"]+coin_value

st.title("AI Crypto Trader")

c1,c2,c3=st.columns(3)

c1.metric("총 자산",f"{asset:,.0f}")
c2.metric("현금",f"{wallet['krw']:,.0f}")
c3.metric("코인 평가",f"{coin_value:,.0f}")

st.subheader("보유 코인")

if rows:
    st.dataframe(pd.DataFrame(rows))
else:
    st.write("보유 코인 없음")

st.subheader("최근 거래")

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 20",conn)

st.dataframe(hist)

time.sleep(300)

st.rerun()
