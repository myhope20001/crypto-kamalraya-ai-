import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import time
import requests
from datetime import datetime
import lightgbm as lgb

st.set_page_config(page_title="AI Self Learning Trader",layout="wide")

DB="ai_trader.db"
conn=sqlite3.connect(DB,check_same_thread=False)
cur=conn.cursor()

# -----------------------------
# DB
# -----------------------------

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
f1 REAL,f2 REAL,f3 REAL,f4 REAL,f5 REAL,
f6 REAL,f7 REAL,f8 REAL,f9 REAL,f10 REAL,
f11 REAL,f12 REAL,f13 REAL,f14 REAL,f15 REAL,
f16 REAL,f17 REAL,f18 REAL,f19 REAL,f20 REAL,
f21 REAL,f22 REAL,f23 REAL,f24 REAL,f25 REAL,
f26 REAL,f27 REAL,f28 REAL,f29 REAL,f30 REAL,
target INTEGER
)
""")

conn.commit()

# -----------------------------
# wallet
# -----------------------------

if "wallet" not in st.session_state:

    st.session_state.wallet={
        "krw":10000000.0,
        "positions":{}
    }

wallet=st.session_state.wallet

# -----------------------------
# indicator
# -----------------------------

def indicators(df):

    df["rsi"]=100-(100/(1+(df.close.diff().clip(lower=0).rolling(14).mean()/(-df.close.diff().clip(upper=0).rolling(14).mean()))))

    df["ema5"]=df.close.ewm(span=5).mean()
    df["ema10"]=df.close.ewm(span=10).mean()
    df["ema20"]=df.close.ewm(span=20).mean()

    df["macd"]=df.ema5-df.ema20
    df["macd_signal"]=df.macd.ewm(span=9).mean()

    df["atr"]=(df.high-df.low).rolling(14).mean()

    df["roc"]=df.close.pct_change(5)

    df["vol_mean"]=df.volume.rolling(20).mean()

    df["vol_ratio"]=df.volume/df.vol_mean

    df["momentum"]=df.close.pct_change(3)

    df["boll_mid"]=df.close.rolling(20).mean()
    df["boll_std"]=df.close.rolling(20).std()

    df["boll_up"]=df.boll_mid+2*df.boll_std
    df["boll_low"]=df.boll_mid-2*df.boll_std

    df["boll_gap"]=(df.close-df.boll_mid)/df.boll_std

    df["vwap"]=(df.close*df.volume).cumsum()/df.volume.cumsum()

    df["obv"]=(np.sign(df.close.diff())*df.volume).fillna(0).cumsum()

    return df

# -----------------------------
# feature
# -----------------------------

def feature_vector(df):

    r=df.iloc[-1]

    feats=[
        r.rsi,
        r.ema5/r.close,
        r.ema10/r.close,
        r.ema20/r.close,
        r.macd,
        r.macd_signal,
        r.atr,
        r.roc,
        r.vol_ratio,
        r.momentum,
        r.boll_gap,
        r.vwap/r.close,
        r.obv,
    ]

    # 추가 feature 생성
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

def trade(model):

    coins=top100()

    for coin in coins:

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

        conn.commit()

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

# -----------------------------
# dashboard
# -----------------------------

st.title("AI Self Learning Crypto Trader")

if st.button("AI 데이터 업데이트"):
    build_learning()

model=train()

if model:

    trade(model)

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

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50",conn)

st.subheader("최근 거래")

st.dataframe(hist)

time.sleep(300)

st.rerun()
