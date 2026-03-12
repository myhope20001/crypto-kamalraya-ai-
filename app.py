# -*- coding: utf-8 -*-

import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import lightgbm as lgb
import requests
import time

from datetime import datetime

# -----------------------------
# DB 연결
# -----------------------------

conn=sqlite3.connect("ai_trader.db",check_same_thread=False)
cur=conn.cursor()

# -----------------------------
# 테이블 생성
# -----------------------------

cur.execute("""
CREATE TABLE IF NOT EXISTS wallet(
id INTEGER PRIMARY KEY,
krw REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS positions(
coin TEXT,
qty REAL,
buy_price REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS trades(
id INTEGER PRIMARY KEY AUTOINCREMENT,
time TEXT,
coin TEXT,
price REAL,
qty REAL,
side TEXT
)
""")

cols=",".join([f"f{i} REAL" for i in range(30)])

cur.execute(f"""
CREATE TABLE IF NOT EXISTS learning(
id INTEGER PRIMARY KEY AUTOINCREMENT,
{cols},
target INTEGER
)
""")

conn.commit()

# -----------------------------
# 최초 자금 세팅
# -----------------------------

cur.execute("SELECT * FROM wallet")

if cur.fetchone() is None:

    cur.execute(
    "INSERT INTO wallet VALUES(1,10000000)"
    )

    conn.commit()

# -----------------------------
# 지갑 로드
# -----------------------------

def load_wallet():

    krw=cur.execute(
    "SELECT krw FROM wallet WHERE id=1"
    ).fetchone()[0]

    rows=cur.execute(
    "SELECT * FROM positions"
    ).fetchall()

    pos={}

    for r in rows:

        pos[r[0]]={
        "qty":r[1],
        "buy_price":r[2]
        }

    return {"krw":krw,"positions":pos}

# -----------------------------
# 지갑 저장
# -----------------------------

def save_wallet(wallet):

    cur.execute(
    "UPDATE wallet SET krw=? WHERE id=1",
    (wallet["krw"],)
    )

    cur.execute("DELETE FROM positions")

    for coin,p in wallet["positions"].items():

        cur.execute(
        "INSERT INTO positions VALUES(?,?,?)",
        (coin,p["qty"],p["buy_price"])
        )

    conn.commit()

# -----------------------------
# 지표 생성
# -----------------------------

def indicators(df):

    df["rsi"]=df.close.pct_change().rolling(14).mean()

    df["ma5"]=df.close.rolling(5).mean()

    df["ma20"]=df.close.rolling(20).mean()

    df["momentum"]=df.close/df.close.shift(5)

    df["roc"]=df.close.pct_change(5)

    df["vol_ratio"]=df.volume/df.volume.rolling(10).mean()

    df["boll_gap"]=(df.close-df.ma20)/df.ma20

    df["obv"]=(np.sign(df.close.diff())*df.volume).fillna(0).cumsum()

    df["vwap"]=(df.close*df.volume).cumsum()/df.volume.cumsum()

    df["atr"]=(df.high-df.low).rolling(14).mean()

    return df

# -----------------------------
# feature vector
# -----------------------------

def feature_vector(df):

    r=df.iloc[-1]

    feats=[

        r.rsi,
        r.ma5/r.close,
        r.ma20/r.close,
        r.atr,
        r.roc,
        r.vol_ratio,
        r.momentum,
        r.boll_gap,
        r.vwap/r.close,
        r.obv

    ]

    while len(feats)<30:

        feats.append(np.random.random())

    return feats[:30]

# -----------------------------
# 코인 리스트
# -----------------------------

def tradable():

    url="https://api.upbit.com/v1/market/all"

    res=requests.get(url).json()

    coins=[x["market"] for x in res if x["market"].startswith("KRW-")]

    return coins

# -----------------------------
# 거래대금 상위
# -----------------------------

def top100():

    tickers=tradable()

    data=[]

    for t in tickers:

        try:

            df=pyupbit.get_ohlcv(t,"minute1",count=10)

            val=(df.close*df.volume).sum()

            data.append((t,val))

        except:
            pass

    data=sorted(data,key=lambda x:x[1],reverse=True)

    return [x[0] for x in data[:100]]

# -----------------------------
# learning data build
# -----------------------------

def build_learning():

    coins=top100()

    for coin in coins:

        df=pyupbit.get_ohlcv(coin,"minute1",count=200)

        if df is None:
            continue

        df=indicators(df)

        df["target"]=(df.close.shift(-5)>df.close).astype(int)

        df=df.dropna()

        for i in range(len(df)-1):

            feats=feature_vector(df.iloc[:i+1])

            target=df.iloc[i]["target"]

            cur.execute(
            "INSERT INTO learning VALUES(NULL,"+
            ",".join(["?"]*30)+",?)",
            feats+[target]
            )

    conn.commit()

# -----------------------------
# train model
# -----------------------------

def train():

    df=pd.read_sql("SELECT * FROM learning",conn)

    if len(df)<5000:
        return None

    X=df.drop(["id","target"],axis=1)

    y=df["target"]

    train_data=lgb.Dataset(X,label=y)

    params={

    "objective":"binary",
    "metric":"auc",
    "learning_rate":0.01,
    "num_leaves":64

    }

    model=lgb.train(params,train_data,200)

    return model

# -----------------------------
# Kelly 투자
# -----------------------------

def kelly(prob):

    edge=(prob*2)-1

    k=edge

    if k<0:
        return 0

    return min(k,0.25)

# -----------------------------
# AI trading
# -----------------------------

def trade(model,wallet):

    coins=top100()

    # BUY

    for coin in coins:

        if coin in wallet["positions"]:
            continue

        df=pyupbit.get_ohlcv(coin,"minute1",count=100)

        if df is None:
            continue

        df=indicators(df)

        feats=feature_vector(df)

        prob=model.predict([feats])[0]

        invest_ratio=kelly(prob)

        if invest_ratio<=0:
            continue

        price=pyupbit.get_current_price(coin)

        invest=wallet["krw"]*invest_ratio

        if invest<10000:
            continue

        qty=invest/price

        wallet["krw"]-=invest

        wallet["positions"][coin]={

        "qty":qty,
        "buy_price":price

        }

        cur.execute(
        "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
        (datetime.now(),coin,price,qty,"BUY")
        )

    # SELL

    for coin in list(wallet["positions"].keys()):

        pos=wallet["positions"][coin]

        price=pyupbit.get_current_price(coin)

        profit=(price-pos["buy_price"])/pos["buy_price"]

        df=pyupbit.get_ohlcv(coin,"minute1",count=100)

        df=indicators(df)

        feats=feature_vector(df)

        prob=model.predict([feats])[0]

        if prob<0.45 or profit>0.08 or profit<-0.03:

            qty=pos["qty"]

            wallet["krw"]+=qty*price

            del wallet["positions"][coin]

            cur.execute(
            "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
            (datetime.now(),coin,price,qty,"SELL")
            )

    conn.commit()

    save_wallet(wallet)

# -----------------------------
# dashboard
# -----------------------------

st.title("AI Self Learning Crypto Trader")

wallet=load_wallet()

if st.button("AI 데이터 업데이트"):
    build_learning()

model=train()

if model:
    trade(model,wallet)

# 자산 계산

coin_value=0
rows=[]

for coin,pos in wallet["positions"].items():

    price=pyupbit.get_current_price(coin)

    value=price*pos["qty"]

    coin_value+=value

    profit=(price-pos["buy_price"])/pos["buy_price"]*100

    rows.append({

    "coin":coin,
    "qty":pos["qty"],
    "buy_price":pos["buy_price"],
    "price":price,
    "profit%":profit

    })

asset=wallet["krw"]+coin_value

c1,c2,c3=st.columns(3)

c1.metric("총자산",f"{asset:,.0f}")
c2.metric("현금",f"{wallet['krw']:,.0f}")
c3.metric("코인평가",f"{coin_value:,.0f}")

st.dataframe(pd.DataFrame(rows))

hist=pd.read_sql(
"SELECT * FROM trades ORDER BY id DESC LIMIT 50",
conn
)

st.subheader("최근 거래")

st.dataframe(hist)

time.sleep(300)

st.rerun()
