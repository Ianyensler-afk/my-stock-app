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
st.set_page_config(page_title="台股戰情室 V20.0 (K線權威版)", layout="wide")
st.title("📈 台股 AI 投資決策系統 (16種K線權威版)")

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
        res = requests.get(url, headers=headers, timeout=10)
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

# --- 4. 資料處理 ---
@st.cache_data
def load_data(symbol, period):
    try:
        data = yf.Ticker(symbol).history(period=period)
        if data.empty:
            if ".TW" in symbol: data = yf.Ticker(symbol.replace(".TW", "")).history(period=period)
            else: data = yf.Ticker(f"{symbol}.TW").history(period=period)
        if data.empty: return None
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        data.dropna(subset=['Close'], inplace=True)
        return data
    except: return None

def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean() # 月線
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / loss.ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    # KDJ
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

# --- 5. [新增] 16種 K 線型態辨識引擎 ---
def analyze_16_patterns(df):
    """
    綜合參考 Yahoo, OANDA, QuantPass 等資料，辨識 16 種 K 線型態。
    需要 dataframe 最後三筆資料來判斷組合型態。
    """
    if len(df) < 3: return "資料不足", "無法判斷", 0
    
    # 提取近三天數據 (T=今天, T-1=昨天, T-2=前天)
    t0 = df.iloc[-1]
    t1 = df.iloc[-2]
    t2 = df.iloc[-3]
    
    # 輔助計算函數
    def get_body(row): return abs(row['Close'] - row['Open'])
    def get_upper(row): return row['High'] - max(row['Open'], row['Close'])
    def get_lower(row): return min(row['Open'], row['Close']) - row['Low']
    def is_red(row): return row['Close'] > row['Open']
    def is_black(row): return row['Close'] < row['Open']
    def body_ratio(row): return get_body(row) / (row['High'] - row['Low'] + 0.001)
    
    # 趨勢判斷 (利用 MA20 或 近三日走勢)
    is_uptrend = t0['Close'] > t0['MA20']
    is_downtrend = t0['Close'] < t0['MA20']
    
    pattern = "一般震盪"
    meaning = "多空力道均衡，等待方向"
    score = 0

    # === A. 三根 K 線組合型態 (強度最高) ===
    
    # 1. 晨星 (Morning Star) [底] - 黑K + 十字/小K + 紅K
    if is_downtrend and is_black(t2) and body_ratio(t2) > 0.5 and \
       body_ratio(t1) < 0.3 and \
       is_red(t0) and t0['Close'] > (t2['Open'] + t2['Close'])/2:
       return "晨星 (Morning Star)", "✅ [強烈看多] 黎明到來，空頭力竭，趨勢反轉向上", 3

    # 2. 夜星 (Evening Star) [頭] - 紅K + 十字/小K + 黑K
    if is_uptrend and is_red(t2) and body_ratio(t2) > 0.5 and \
       body_ratio(t1) < 0.3 and \
       is_black(t0) and t0['Close'] < (t2['Open'] + t2['Close'])/2:
       return "夜星 (Evening Star)", "❌ [強烈看空] 夜幕降臨，多頭力竭，趨勢反轉向下", -3

    # 3. 紅三兵 (Three White Soldiers) [多] - 連三紅
    if is_red(t2) and is_red(t1) and is_red(t0) and \
       t0['Close'] > t1['Close'] > t2['Close']:
       return "紅三兵 (Three White Soldiers)", "✅ [持續看多] 多頭氣勢如虹，穩健上攻", 2

    # 4. 黑三鴉 (Three Black Crows) [空] - 連三黑
    if is_black(t2) and is_black(t1) and is_black(t0) and \
       t0['Close'] < t1['Close'] < t2['Close']:
       return "黑三鴉 (Three Black Crows)", "❌ [持續看空] 賣壓沈重，恐慌性拋售", -2

    # === B. 兩根 K 線組合型態 (強度中等) ===

    # 5. 多頭吞噬 (Bullish Engulfing) [底] - 黑K後接大紅K包覆
    if is_downtrend and is_black(t1) and is_red(t0) and \
       t0['Open'] <= t1['Close'] and t0['Close'] >= t1['Open']:
       return "多頭吞噬 (Bullish Engulfing)", "✅ [強勢反轉] 一根長紅吃掉昨日跌幅，買盤強勁", 2

    # 6. 空頭吞噬 (Bearish Engulfing) [頭] - 紅K後接大黑K包覆
    if is_uptrend and is_red(t1) and is_black(t0) and \
       t0['Open'] >= t1['Close'] and t0['Close'] <= t1['Open']:
       return "空頭吞噬 (Bearish Engulfing)", "❌ [強勢反轉] 一根長黑吃掉昨日漲幅，主力出貨", -2

    # 7. 多頭母子 (Bullish Harami) [底] - 長黑包小紅
    if is_downtrend and is_black(t1) and is_red(t0) and \
       t0['Close'] < t1['Open'] and t0['Open'] > t1['Close']:
       return "多頭母子 (Bullish Harami)", "✅ [止跌訊號] 跌勢受阻，空方力量減弱，醞釀反彈", 1

    # 8. 空頭母子 (Bearish Harami) [頭] - 長紅包小黑
    if is_uptrend and is_red(t1) and is_black(t0) and \
       t0['Close'] > t1['Open'] and t0['Open'] < t1['Close']:
       return "空頭母子 (Bearish Harami)", "❌ [漲勢受阻] 上攻無力，多方力量減弱，小心回檔", -1

    # 9. 烏雲蓋頂 (Dark Cloud Cover) [頭] - 紅K後黑K插入實體一半
    if is_uptrend and is_red(t1) and is_black(t0) and \
       t0['Open'] > t1['High'] and t0['Close'] < (t1['Open'] + t1['Close'])/2:
       return "烏雲蓋頂 (Dark Cloud Cover)", "❌ [看空] 高檔爆量長黑，主力高檔倒貨", -2

    # 10. 貫穿線 (Piercing Line) [底] - 黑K後紅K插入實體一半
    if is_downtrend and is_black(t1) and is_red(t0) and \
       t0['Open'] < t1['Low'] and t0['Close'] > (t1['Open'] + t1['Close'])/2:
       return "貫穿線 (Piercing Line)", "✅ [看多] 低檔承接力道強，多方展開反擊", 2

    # === C. 單根 K 線型態 (強度取決於位置) ===
    
    # 11. 錘頭線 (Hammer) [底] - 下影線長，實體小
    if is_downtrend and get_lower(t0) > 2 * get_body(t0) and get_upper(t0) < get_body(t0):
        return "錘頭線 (Hammer)", "✅ [底部反轉] 低檔有人抄底，支撐強勁", 2

    # 12. 吊人線 (Hanging Man) [頭] - 下影線長，實體小 (與錘頭一樣，但發生在高檔)
    if is_uptrend and get_lower(t0) > 2 * get_body(t0) and get_upper(t0) < get_body(t0):
        return "吊人線 (Hanging Man)", "❌ [頂部反轉] 高檔出現下影線，主力在拉高出貨，誘多", -2

    # 13. 倒錘線 (Inverted Hammer) [底] - 上影線長，實體小
    if is_downtrend and get_upper(t0) > 2 * get_body(t0) and get_lower(t0) < get_body(t0):
        return "倒錘線 (Inverted Hammer)", "⚠️ [潛在反轉] 低檔試盤，需等待明日紅K確認", 1

    # 14. 流星線 (Shooting Star) [頭] - 上影線長，實體小
    if is_uptrend and get_upper(t0) > 2 * get_body(t0) and get_lower(t0) < get_body(t0):
        return "流星線 (Shooting Star)", "❌ [射擊之星] 高檔賣壓湧現，多頭上攻失敗", -2

    # 15. 墓碑線 / 蜻蜓線 (Doji 變體)
    if body_ratio(t0) < 0.1:
        if get_upper(t0) > 0.6 * (t0['High'] - t0['Low']):
            return "墓碑線 (Gravestone Doji)", "❌ [強烈看空] 多頭進攻完全失敗，收在最低", -2
        elif get_lower(t0) > 0.6 * (t0['High'] - t0['Low']):
            return "蜻蜓線 (Dragonfly Doji)", "✅ [強烈看多] 空頭殺盤完全失敗，收在最高", 2
        else:
            return "十字線 (Doji)", "⚠️ [變盤訊號] 多空僵持，市場猶豫，即將表態", 0

    # 16. 大長紅 / 大長黑 (Marubozu)
    if body_ratio(t0) > 0.8:
        if is_red(t0): return "大陽線 (Long White Candle)", "🔥 [極強勢] 買盤從頭買到尾，強勢上攻", 2
        else: return "大陰線 (Long Black Candle)", "💧 [極弱勢] 賣盤從頭殺到尾，恐慌殺盤", -2

    return pattern, meaning, score

# --- 6. 新聞 ---
def get_enhanced_news(clean_id, clean_name):
    queries = [f"{clean_id} {clean_name}", f"{clean_name} 股價", f"{clean_name} 營收"]
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

# --- 7. 回測 ---
def run_backtest(df, fund):
    capital = fund; position = 0; history = []
    for i in range(1, len(df)):
        p = df['Close'].iloc[i]; d = df['Date'].iloc[i]; ma20 = df['MA20'].iloc[i]
        macd_gold = df['MACD'].iloc[i-1] < df['Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['Signal'].iloc[i]
        kdj_gold = df['K'].iloc[i-1] < df['D'].iloc[i-1] and df['K'].iloc[i] > df['D'].iloc[i] and df['K'].iloc[i] < 50
        trend_up = p > ma20
        
        if position == 0 and trend_up and (macd_gold or kdj_gold):
            position = capital / p; capital = 0
            reason = "MACD金叉" if macd_gold else "KDJ金叉"
            history.append(f"🔴 {d.strftime('%Y-%m-%d')} 買進 @ {p:.2f} ({reason} & 站上月線)")
        elif position > 0 and p < ma20:
            capital = position * p; position = 0
            history.append(f"🟢 {d.strftime('%Y-%m-%d')} 賣出 @ {p:.2f} (跌破月線)")
    final = capital if position == 0 else position * df['Close'].iloc[-1]
    return final, history

# --- 主程式 ---
data = load_data(stock_id, date_range)

if data is not None and not data.empty:
    data = calculate_indicators(data)
    info = yf.Ticker(stock_id).info
    name = info.get('longName', stock_id)
    clean_name = name.split(' ')[0]
    
    try: goodinfo_data = get_goodinfo_data(clean_id)
    except: goodinfo_data = None
    
    if goodinfo_data:
        eps = goodinfo_data['EPS']; roe = goodinfo_data['ROE']; dy = goodinfo_data['Yield']; per = goodinfo_data['PER']
        src_tag = "✅ 數據來源: Goodinfo (爬蟲成功)"
    else:
        eps = info.get('trailingEps', 0); roe = (info.get('returnOnEquity', 0) or 0)*100
        dy = (info.get('dividendYield', 0) or 0)*100
        per = info.get('trailingPE', 0)
        src_tag = "⚠️ 數據來源: Yahoo (Goodinfo 連線失敗)"

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
    
    # 看板
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
    
    # === AI 綜合診斷 (含 16種 K線權威版) ===
    last = data.iloc[-1]
    score = 0; reasons = []
    
    # 1. 16種 K 線型態辨識 (V20 新增核心)
    # 我們傳入整個 dataframe 讓函數去讀取最後三天
    k_name, k_meaning, k_score = analyze_16_patterns(data)
    score += k_score
    reasons.append(f"🕯️ [型態] **{k_name}**：{k_meaning}")
    
    # 2. 技術指標
    if curr_p > last['MA20']: score += 2; reasons.append("✅ [趨勢] 站上月線 (短多) +2")
    else: score -= 2; reasons.append("🔻 [趨勢] 跌破月線 (短空) -2")
    
    if last['MACD'] > last['Signal']: score += 2; reasons.append("✅ [動能] MACD金叉 +2")
    else: score -= 2; reasons.append("🔻 [動能] MACD死叉 -2")
    
    if last['K'] > last['D']: score += 1; reasons.append("✅ [波段] KDJ金叉 +1")
    else: score -= 1; reasons.append("🔻 [波段] KDJ死叉 -1")
    
    # 3. 基本面
    if eps > 0: score += 1; reasons.append("✅ [基本] EPS獲利中 +1")
    else: score -= 1; reasons.append("🔻 [基本] EPS虧損中 -1")
    
    bg = "#d4edda" if score >= 4 else "#f8d7da" if score <= -4 else "#fff3cd"
    sugg = "強烈買進 🚀" if score >= 4 else "強烈賣出 🐻" if score <= -4 else "觀望/區間操作"
    
    st.markdown(f"""<div style="background-color: {bg}; padding: 15px; border-radius: 10px;">
        <h3>🤖 AI 診斷: {sugg} (總分: {score})</h3>
        <ul style="margin-top:10px;">
            {''.join([f'<li style="margin-bottom:5px;">{r}</li>' for r in reasons])}
        </ul>
    </div>""", unsafe_allow_html=True)
    
    # === 分頁 ===
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
            st.warning("目前使用 Yahoo 數據，建議前往 Goodinfo 查看完整財報")
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
    st.error("查無資料") #streamlit run app.py
