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
import threading

# ---------------------------
# DB 연결
# ---------------------------
conn = sqlite3.connect("ai_trader.db", check_same_thread=False)
cur = conn.cursor()

# ---------------------------
# 테이블 생성
# ---------------------------
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

cur.execute("""
CREATE TABLE IF NOT EXISTS learning_meta(
id INTEGER PRIMARY KEY,
last_time TEXT
)
""")

conn.commit()

# ---------------------------
# 초기 데이터 설정
# ---------------------------
if cur.execute("SELECT * FROM wallet").fetchone() is None:
    cur.execute("INSERT INTO wallet VALUES(1,10000000)")
    conn.commit()

if cur.execute("SELECT * FROM learning_meta").fetchone() is None:
    cur.execute("INSERT INTO learning_meta VALUES(1,'2000-01-01')")
    conn.commit()

# ---------------------------
# 지갑 로드 / 저장
# ---------------------------
def load_wallet():
    krw = cur.execute("SELECT krw FROM wallet WHERE id=1").fetchone()[0]
    rows = cur.execute("SELECT * FROM positions").fetchall()
    pos = {r[0]: {"qty":r[1], "buy_price":r[2]} for r in rows}
    return {"krw": krw, "positions": pos}

def save_wallet(wallet):
    cur.execute("UPDATE wallet SET krw=? WHERE id=1", (wallet["krw"],))
    cur.execute("DELETE FROM positions")
    for coin,p in wallet["positions"].items():
        cur.execute("INSERT INTO positions VALUES(?,?,?)", (coin, p["qty"], p["buy_price"]))
    conn.commit()

# ---------------------------
# 마지막 학습 시간
# ---------------------------
def last_learning_time():
    t = cur.execute("SELECT last_time FROM learning_meta WHERE id=1").fetchone()[0]
    return pd.to_datetime(t)

def update_learning_time():
    now = datetime.now()
    cur.execute("UPDATE learning_meta SET last_time=? WHERE id=1", (now,))
    conn.commit()

# ---------------------------
# 지표 / feature
# ---------------------------
def indicators(df):
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

def feature_vector(df):
    r = df.iloc[-1]
    feats = [
        r.ma5/r.close, r.ma20/r.close, r.atr, r.roc, r.vol_ratio,
        r.momentum, r.boll_gap, r.vwap/r.close, r.obv
    ]
    while len(feats)<30:
        feats.append(np.random.random())
    return feats[:30]

# ---------------------------
# 코인 리스트 / top100
# ---------------------------
def tradable():
    url="https://api.upbit.com/v1/market/all"
    res=requests.get(url).json()
    coins=[x["market"] for x in res if x["market"].startswith("KRW-")]
    return coins

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

# ---------------------------
# 학습 / AI 모델
# ---------------------------
def auto_learning():
    last_time = last_learning_time()
    coins=top100()
    for coin in coins:
        df = pyupbit.get_ohlcv(coin,"minute1",count=200)
        if df is None: continue
        df = indicators(df)
        df["target"] = (df.close.shift(-5)>df.close).astype(int)
        df = df.dropna()
        for i in range(len(df)-1):
            row_time = df.index[i]
            if row_time <= last_time: continue
            feats = feature_vector(df.iloc[:i+1])
            target = df.iloc[i]["target"]
            cur.execute("INSERT INTO learning VALUES(NULL," + ",".join(["?"]*30) + ",?)", feats+[target])
    conn.commit()
    update_learning_time()

def train():
    df = pd.read_sql("SELECT * FROM learning", conn)
    if len(df)<5000: return None
    X=df.drop(["id","target"],axis=1)
    y=df["target"]
    train_data=lgb.Dataset(X,label=y)
    params={"objective":"binary","metric":"auc","learning_rate":0.01,"num_leaves":64}
    model=lgb.train(params,train_data,200)
    return model

def kelly(prob):
    edge=(prob*2)-1
    if edge<0: return 0
    return min(edge,0.25)

def trade(model, wallet):
    coins = top100()
    for coin in coins:
        if coin in wallet["positions"]: continue
        df = pyupbit.get_ohlcv(coin,"minute1",count=100)
        if df is None: continue
        df = indicators(df)
        feats = feature_vector(df)
        prob = model.predict([feats])[0]
        invest_ratio = kelly(prob)
        if invest_ratio <=0: continue
        price = pyupbit.get_current_price(coin)
        invest = wallet["krw"]*invest_ratio
        if invest<10000: continue
        qty = invest/price
        wallet["krw"]-=invest
        wallet["positions"][coin] = {"qty":qty,"buy_price":price}
        cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",(datetime.now(),coin,price,qty,"BUY"))

    for coin in list(wallet["positions"].keys()):
        pos=wallet["positions"][coin]
        price=pyupbit.get_current_price(coin)
        profit=(price-pos["buy_price"])/pos["buy_price"]
        df = pyupbit.get_ohlcv(coin,"minute1",count=100)
        df = indicators(df)
        feats = feature_vector(df)
        prob = model.predict([feats])[0]
        if prob<0.45 or profit>0.08 or profit<-0.03:
            qty=pos["qty"]
            wallet["krw"] += qty*price
            del wallet["positions"][coin]
            cur.execute("INSERT INTO trades VALUES(NULL,?,?,?,?,?)",(datetime.now(),coin,price,qty,"SELL"))
    conn.commit()
    save_wallet(wallet)

# ---------------------------
# 백그라운드 엔진
# ---------------------------
def ai_engine():
    wallet = load_wallet()
    while True:
        auto_learning()
        model = train()
        if model: trade(model, wallet)
        time.sleep(1800)  # 30분마다 반복

# ---------------------------
# Streamlit Dashboard
# ---------------------------
st.title("AI Self Learning Crypto Trader (24h)")

wallet = load_wallet()

# 백그라운드 스레드 실행
if "engine_started" not in st.session_state:
    t = threading.Thread(target=ai_engine, daemon=True)
    t.start()
    st.session_state.engine_started = True

# 대시보드 표시
coin_value=0
rows=[]
for coin,pos in wallet["positions"].items():
    price=pyupbit.get_current_price(coin)
    value=price*pos["qty"]
    coin_value+=value
    profit=(price-pos["buy_price"])/pos["buy_price"]*100
    rows.append({"coin":coin,"qty":pos["qty"],"buy_price":pos["buy_price"],"price":price,"profit%":profit})

asset=wallet["krw"]+coin_value
c1,c2,c3=st.columns(3)
c1.metric("총자산",f"{asset:,.0f}")
c2.metric("현금",f"{wallet['krw']:,.0f}")
c3.metric("코인평가",f"{coin_value:,.0f}")

st.subheader("보유 코인")
st.dataframe(pd.DataFrame(rows))

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50",conn)
st.subheader("최근 거래")
st.dataframe(hist)

st.write("⚠️ Streamlit은 UI 역할만 합니다. 매매/학습 엔진은 백그라운드에서 24시간 동작합니다.")
