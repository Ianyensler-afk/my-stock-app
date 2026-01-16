import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import feedparser
import urllib.parse
import time
import requests

# --- 1. 設定頁面 ---
st.set_page_config(page_title="台股戰情室 V21.0 (抗崩潰版)", layout="wide")
st.title("📈 台股 AI 投資決策系統 (安全氣囊版)")

# --- 2. 側邊欄 ---
st.sidebar.header("🔍 查詢參數")
ticker_input = st.sidebar.text_input("股票代號:", "2330")
date_range = st.sidebar.select_slider("資料區間", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")
initial_capital = st.sidebar.number_input("回測初始資金 (元)", value=100000)

def format_ticker(symbol):
    symbol = symbol.strip()
    if symbol.isdigit(): return f"{symbol}.TW"
    return symbol.upper()

stock_id = format_ticker(ticker_input)
clean_id = stock_id.split('.')[0]

# --- 3. Goodinfo 爬蟲 ---
@st.cache_data(ttl=3600)
def get_goodinfo_data(stock_id_num):
    url = f"https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={stock_id_num}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36", "Referer": url}
    try:
        res = requests.get(url, headers=headers, timeout=5) # timeout 改短一點，避免卡死
        res.encoding = "utf-8"
        dfs = pd.read_html(res.text)
        target_df = None
        for df in dfs:
            flat_cols = [''.join(str(c) for c in col) for col in df.columns] if isinstance(df.columns, pd.MultiIndex) else df.columns
            if any("EPS" in str(c) for c in flat_cols): target_df = df; target_df.columns = flat_cols; break
        
        if target_df is None: return None
        cols = target_df.columns
        eps_c = next((c for c in cols if "EPS" in c and "元" in c), None)
        roe_c = next((c for c in cols if "ROE" in c), None)
        yield_c = next((c for c in cols if "殖利率" in c), None)
        pe_c = next((c for c in cols if "本益比" in c), None)
        
        def parse(v):
            try: return float(v)
            except: return 0.0

        return {
            "EPS": parse(target_df[eps_c].iloc[0]) if eps_c else 0,
            "ROE": parse(target_df[roe_c].iloc[0]) if roe_c else 0,
            "Yield": parse(target_df[yield_c].iloc[0]) if yield_c else 0,
            "PER": parse(target_df[pe_c].iloc[0]) if pe_c else 0
        }
    except: return None

# --- 4. [修復重點] 安全獲取 YFinance 資料 ---
@st.cache_data(ttl=300) # 加上快取，減少對 Yahoo 的請求頻率
def load_data_safe(symbol, period):
    try:
        # 嘗試抓取
        data = yf.Ticker(symbol).history(period=period)
        
        # 容錯嘗試
        if data.empty:
            if ".TW" in symbol: data = yf.Ticker(symbol.replace(".TW", "")).history(period=period)
            else: data = yf.Ticker(f"{symbol}.TW").history(period=period)
            
        if data.empty: return None
        
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        data.dropna(subset=['Close'], inplace=True)
        return data
    except Exception as e:
        # 這裡會捕捉 Rate Limit 錯誤，回傳 None 而不是讓程式崩潰
        print(f"Yahoo Data Error: {e}")
        return None

# [新增] 安全獲取基本面 Info，避免 info 屬性報錯
def get_stock_info_safe(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # 這裡最容易觸發 RateLimitError，所以一定要包起來
        return ticker.info
    except Exception:
        return {} # 失敗就回傳空字典，程式繼續跑

def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    k, d = [50], [50]
    for i in range(1, len(rsv)):
        k.append(k[-1]*2/3 + rsv.iloc[i]/3)
        d.append(d[-1]*2/3 + k[-1]/3)
    df['K'] = k; df['D'] = d
    return df

def calculate_support_resistance(df, window=60):
    recent = df[-window:]
    return recent['Low'].min(), recent['High'].max()

def analyze_16_patterns(df):
    if len(df) < 3: return "資料不足", "無法判斷", 0
    t0 = df.iloc[-1]; t1 = df.iloc[-2]; t2 = df.iloc[-3]
    
    def get_body(row): return abs(row['Close'] - row['Open'])
    def get_upper(row): return row['High'] - max(row['Open'], row['Close'])
    def get_lower(row): return min(row['Open'], row['Close']) - row['Low']
    def is_red(row): return row['Close'] > row['Open']
    def is_black(row): return row['Close'] < row['Open']
    def body_ratio(row): return get_body(row) / (row['High'] - row['Low'] + 0.001)
    
    is_downtrend = t0['Close'] < t0['MA20']
    
    # 簡化版判斷邏輯，避免過長
    if body_ratio(t0) < 0.1:
        if get_upper(t0) > 0.6*(t0['High']-t0['Low']): return "墓碑線", "❌ 遇壓收低，強烈看空", -2
        if get_lower(t0) > 0.6*(t0['High']-t0['Low']): return "蜻蜓線", "✅ 支撐強勁，強烈看多", 2
        return "十字線", "⚠️ 多空僵持，變盤前兆", 0
    
    if body_ratio(t0) > 0.8:
        if is_red(t0): return "大陽線", "🔥 買盤強勁，趨勢看漲", 2
        else: return "大陰線", "💧 賣壓沉重，趨勢看跌", -2
        
    return "一般震盪", "無特殊型態", 0

# --- 新聞 ---
def get_enhanced_news(clean_id, clean_name):
    queries = [f"{clean_id} {clean_name}", f"{clean_name} 股價"]
    all_news = []
    seen = set()
    blacklist = ["NBA","MLB","職棒","籃球","啦啦隊","藝人","緋聞","演唱會","彩券"]
    for q in queries:
        encoded = urllib.parse.quote(q)
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
        for entry in feed.entries:
            if entry.link in seen: continue
            if any(x in entry.title for x in blacklist): continue
            all_news.append(entry)
            seen.add(entry.link)
    all_news.sort(key=lambda x: x.published_parsed if x.get('published_parsed') else time.localtime(0), reverse=True)
    return all_news

# --- 回測 ---
def run_backtest(df, fund):
    capital = fund; position = 0; history = []
    for i in range(1, len(df)):
        p = df['Close'].iloc[i]; d = df['Date'].iloc[i]; ma20 = df['MA20'].iloc[i]
        macd_gold = df['MACD'].iloc[i-1] < df['Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['Signal'].iloc[i]
        kdj_gold = df['K'].iloc[i-1] < df['D'].iloc[i-1] and df['K'].iloc[i] > df['D'].iloc[i] and df['K'].iloc[i] < 50
        trend_up = p > ma20
        if position == 0 and trend_up and (macd_gold or kdj_gold):
            position = capital / p; capital = 0
            history.append(f"🔴 {d.strftime('%Y-%m-%d')} 買進 @ {p:.2f}")
        elif position > 0 and p < ma20:
            capital = position * p; position = 0
            history.append(f"🟢 {d.strftime('%Y-%m-%d')} 賣出 @ {p:.2f}")
    final = capital if position == 0 else position * df['Close'].iloc[-1]
    return final, history

# --- 主程式 ---
# [重要修改] 使用 Safe 版函數，這裡不會崩潰，只會回傳 None
data = load_data_safe(stock_id, date_range)

if data is not None and not data.empty:
    data = calculate_indicators(data)
    
    # [重要修改] 使用 Safe 版 Info，如果 Yahoo 鎖 IP，這裡會拿到空字典 {}
    info = get_stock_info_safe(stock_id) 
    
    # 容錯處理：如果 info 是空的，就給預設值
    name = info.get('longName', stock_id) if info else stock_id
    clean_name = name.split(' ')[0]
    
    # Goodinfo 爬蟲
    try: goodinfo_data = get_goodinfo_data(clean_id)
    except: goodinfo_data = None
    
    # 數據整合 (Yahoo 失敗就全靠 Goodinfo)
    if goodinfo_data:
        eps = goodinfo_data['EPS']; roe = goodinfo_data['ROE']; dy = goodinfo_data['Yield']; per = goodinfo_data['PER']
        src_tag = "✅ 數據來源: Goodinfo (Yahoo被限流，已自動切換)"
    else:
        # 如果 Yahoo Info 也被鎖，這些都會是 0
        eps = info.get('trailingEps', 0) if info else 0
        roe = (info.get('returnOnEquity', 0) or 0)*100 if info else 0
        dy = (info.get('dividendYield', 0) or 0)*100 if info else 0
        per = info.get('trailingPE', 0) if info else 0
        src_tag = "⚠️ 數據來源: Yahoo (若數據為0代表被限流)"

    # 漲跌幅
    curr_p = data['Close'].iloc[-1]
    prev_p = data['Close'].iloc[-2]
    change = curr_p - prev_p
    pct_change = (change / prev_p) * 100
    
    # 目標價
    target_low = eps * 15; target_high = eps * 22
    if target_low < 0: target_low = 0; target_high = 0
    
    st.subheader(f"{name} ({stock_id})")
    st.caption(src_tag)
    
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric("股價", f"{curr_p:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    col_k2.metric("EPS", f"{eps:.2f}元")
    supp, res = calculate_support_resistance(data)
    col_k3.metric("壓力", f"{res:.2f}")
    col_k4.metric("支撐", f"{supp:.2f}")
    with col_k5:
        st.metric("估算目標區間", f"{target_low:.0f} - {target_high:.0f}")
        st.link_button("🔍 搜券商報告", f"https://www.google.com/search?q={clean_name}+{clean_id}+目標價+券商+報告")

    st.divider()
    
    # AI 診斷
    last = data.iloc[-1]
    score = 0; reasons = []
    
    k_name, k_meaning, k_score = analyze_16_patterns(data)
    score += k_score
    reasons.append(f"🕯️ [型態] **{k_name}**：{k_meaning}")
    
    if curr_p > last['MA20']: score += 2; reasons.append("✅ [趨勢] 站上月線 +2")
    else: score -= 2; reasons.append("🔻 跌破月線 -2")
    if last['MACD'] > last['Signal']: score += 2; reasons.append("✅ [動能] MACD金叉 +2")
    else: score -= 2; reasons.append("🔻 [動能] MACD死叉 -2")
    if last['K'] > last['D']: score += 1; reasons.append("✅ [波段] KDJ金叉 +1")
    else: score -= 1; reasons.append("🔻 [波段] KDJ死叉 -1")
    if eps > 0: score += 1; reasons.append("✅ [基本] EPS獲利 +1")
    else: score -= 1; reasons.append("🔻 [基本] EPS虧損 -1")
    
    bg = "#d4edda" if score >= 4 else "#f8d7da" if score <= -4 else "#fff3cd"
    sugg = "強烈買進 🚀" if score >= 4 else "強烈賣出 🐻" if score <= -4 else "觀望"
    
    st.markdown(f"""<div style="background-color: {bg}; padding: 15px; border-radius: 10px;">
        <h3>🤖 AI 診斷: {sugg} (總分: {score})</h3>
        <ul style="margin-top:10px;">
            {''.join([f'<li style="margin-bottom:5px;">{r}</li>' for r in reasons])}
        </ul>
    </div>""", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 全能圖表", "💰 策略回測", "🏢 財報詳情", "🕵️ 籌碼面", "📰 新聞"])
    
    with tab1:
        data['DateStr'] = data['Date'].dt.strftime('%Y-%m-%d')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=data['DateStr'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['MA20'], line=dict(color='blue'), name='MA20'), row=1, col=1)
        fig.add_hline(y=res, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_hline(y=supp, line_dash="dot", line_color="green", row=1, col=1)
        clrs = ['red' if v < 0 else 'green' for v in data['Hist']]
        fig.add_trace(go.Bar(x=data['DateStr'], y=data['Hist'], marker_color=clrs, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['MACD'], line=dict(color='black'), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['Signal'], line=dict(color='red'), name='DEM'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['K'], line=dict(color='orange'), name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['D'], line=dict(color='blue'), name='D'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['DateStr'], y=data['RSI'], line=dict(color='purple'), name='RSI'), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
        fig.update_xaxes(type='category', tickmode='auto', nticks=20) 
        fig.update_layout(height=900, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        final_val, trade_log = run_backtest(data, initial_capital)
        ret = (final_val - initial_capital)/initial_capital * 100
        st.metric("策略回報", f"{ret:.2f}%")
        with st.expander("交易明細", expanded=True):
            if trade_log:
                for log in trade_log: st.markdown(log)
            else: st.info("無交易訊號")

    with tab3:
        if goodinfo_data:
            st.success("Goodinfo 數據抓取成功")
            st.json(goodinfo_data)
        else:
            st.warning("Yahoo Finance 限流中，且 Goodinfo 備援失敗。建議稍後再試。")
            st.link_button("前往 Goodinfo", f"https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={clean_id}")

    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1: st.link_button("三大法人 (玩股網)", f"https://www.wantgoo.com/stock/{clean_id}/institutional-investors")
        with c2: st.link_button("主力進出 (玩股網)", f"https://www.wantgoo.com/stock/{clean_id}/major-investors")
        with c3: st.link_button("籌碼K線 (Goodinfo)", f"https://goodinfo.tw/tw/ShowK_Chart.asp?STOCK_ID={clean_id}&CHT_CAT=DATE")

    with tab5:
        news = get_enhanced_news(clean_id, clean_name)
        if news:
            for n in news[:30]:
                try: t_str = datetime(*n.published_parsed[:6]).strftime('%m-%d %H:%M')
                except: t_str = "Recent"
                st.markdown(f"**{t_str}** [{n.title}]({n.link})")
                st.divider()
        else: st.info("無新聞")
        st.link_button("🔎 去 Google 搜尋更多", f"https://www.google.com/search?q={clean_name}+新聞")

else:
    # 這裡顯示給用戶看，讓他們知道為什麼現在沒畫面
    st.error("⚠️ 伺服器忙碌中 (Yahoo Finance 限流)")
    st.info("由於雲端主機共用 IP，Yahoo 暫時阻擋了連線。請嘗試：\n1. 等待 5~10 分鐘後重新整理。\n2. 晚點再試。") #streamlit run app.py
