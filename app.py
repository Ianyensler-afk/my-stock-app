import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import feedparser
import urllib.parse
import time

# --- 1. 設定頁面 ---
st.set_page_config(page_title="台股戰情室 V13.0", layout="wide")
st.title("📈 台股 AI 投資決策系統 (全方位版)")

# --- 2. 側邊欄 ---
st.sidebar.header("🔍 查詢參數")
ticker_input = st.sidebar.text_input("股票代號:", "2330")
date_range = st.sidebar.select_slider(
    "資料區間", 
    options=["3mo", "6mo", "1y", "2y", "5y"], 
    value="1y"
)
initial_capital = st.sidebar.number_input("回測初始資金 (元)", value=100000)

def format_ticker(symbol):
    symbol = symbol.strip()
    if symbol.isdigit():
        return f"{symbol}.TW"
    return symbol.upper()

stock_id = format_ticker(ticker_input)

# --- 3. 核心數據與指標計算 ---
# --- 替換原本的 load_data 函數 ---
@st.cache_data
def load_data(symbol, period):
    try:
        # 改法 1: 使用 Ticker.history (更穩定)
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        # 如果抓不到，嘗試加上 .TW 或移除 .TW 再試一次 (容錯機制)
        if data.empty:
            if ".TW" not in symbol:
                data = yf.Ticker(f"{symbol}.TW").history(period=period)
            else:
                data = yf.Ticker(symbol.replace(".TW", "")).history(period=period)
        
        # 如果還是空的，回傳 None
        if data.empty:
            return None
        
        # 重設索引，讓 Date 變成一個欄位
        data.reset_index(inplace=True)
        
        # 處理時區問題 (Yahoo 有時會回傳帶時區的日期，這會導致畫圖失敗)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
        # 基本清洗
        data.dropna(subset=['Close'], inplace=True)
        
        return data
    except Exception as e:
        # 在網頁上印出錯誤訊息，方便除錯 (Debug)
        st.error(f"資料抓取失敗: {e}")
        return None

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# 新增：KDJ 指標計算
def calculate_kdj(df, period=9):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    
    # RSV = (今日收盤 - 最近9天最低) / (最近9天最高 - 最近9天最低) * 100
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    
    # 處理 NaN (開頭數據不足時)
    rsv = rsv.fillna(50)
    
    k = [50]
    d = [50]
    
    for i in range(1, len(rsv)):
        k_val = (2/3) * k[-1] + (1/3) * rsv.iloc[i]
        d_val = (2/3) * d[-1] + (1/3) * k_val
        k.append(k_val)
        d.append(d_val)
        
    df['K'] = k
    df['D'] = d
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def calculate_support_resistance(df, window=60):
    recent_data = df[-window:]
    resistance = recent_data['High'].max()
    support = recent_data['Low'].min()
    return support, resistance

def get_stock_info_safe(symbol):
    try:
        return yf.Ticker(symbol).info
    except:
        return {}

# 新增：K線型態識別 (根據當日 K 線判斷)
def identify_candlestick_pattern(open_p, high_p, low_p, close_p):
    # 實體長度
    body = abs(close_p - open_p)
    # 上影線
    upper_shadow = high_p - max(open_p, close_p)
    # 下影線
    lower_shadow = min(open_p, close_p) - low_p
    # 全長
    total_range = high_p - low_p
    
    if total_range == 0: return "十字線 (Doji) - 多空僵持", 0
    
    # 判斷邏輯
    ratio_body = body / total_range
    ratio_upper = upper_shadow / total_range
    ratio_lower = lower_shadow / total_range
    
    pattern = "一般震盪"
    score = 0 # 1=多, -1=空, 0=中性
    
    # 1. 十字線 (實體極小)
    if ratio_body < 0.1:
        if ratio_upper > 0.6: pattern = "墓碑線 (Gravestone Doji) - 強烈看空"; score = -2
        elif ratio_lower > 0.6: pattern = "蜻蜓線 (Dragonfly Doji) - 強烈看多"; score = 2
        else: pattern = "十字線 (Doji) - 變盤訊號"; score = 0
            
    # 2. 紡錘線 (實體小，影線長)
    elif ratio_body < 0.3:
        if ratio_upper > 0.5: pattern = "流星/倒鎚 (Shooting Star) - 逢高遇壓"; score = -1
        elif ratio_lower > 0.5: pattern = "鎚頭/吊人 (Hammer) - 低檔支撐"; score = 1
        else: pattern = "紡錘線 (Spinning Top) - 猶豫不決"; score = 0
            
    # 3. 大實體
    elif ratio_body > 0.7:
        if close_p > open_p: pattern = "大陽線 (Big Red Candle) - 多頭強勢"; score = 2
        else: pattern = "大陰線 (Big Black Candle) - 空頭強勢"; score = -2
        
    # 4. 特殊：仙人指路 (實體中等，長上影線，發生在紅K或黑K)
    # 定義：上影線長，下影線短，實體不大
    elif ratio_upper > 0.5 and ratio_lower < 0.1:
        pattern = "仙人指路 (高檔有人出貨/低檔測試壓力)"; score = 0 # 需配合位階，暫給中性
        
    return pattern, score

# --- 新聞過濾 ---
def is_irrelevant_news(title):
    exclude_keywords = [
        "NBA", "MLB", "職棒", "中華職棒", "籃球", "棒球", "足球", "羽球", "網球", "奧運",
        "統一獅", "富邦悍將", "中信兄弟", "味全龍", "樂天桃猿", "台鋼雄鷹", 
        "全壘打", "安打", "比分", "勝投", "敗投", "啦啦隊", "女神", "應援",
        "藝人", "緋聞", "娛樂", "演唱會", "劇透", "影評", "八卦", "星座", "運勢", 
        "網紅", "Youtuber", "直播主", "狗仔", "戀情", "分手", "離婚",
        "酒駕", "車禍", "互嗆", "打架"
    ]
    for kw in exclude_keywords:
        if kw in title: return True
    return False

def get_aggregated_news(stock_id, stock_name):
    clean_id = stock_id.split('.')[0]
    clean_name = stock_name.split(' ')[0]
    keywords = [f"{clean_id} {clean_name}", f"{clean_name} 營收", f"{clean_name} 股價", f"{clean_name} 法說", f"{clean_name} 殖利率"]
    all_news = []
    seen_links = set()
    for kw in keywords:
        encoded_query = urllib.parse.quote(kw)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if entry.link in seen_links or is_irrelevant_news(entry.title): continue
            all_news.append(entry)
            seen_links.add(entry.link)
    all_news.sort(key=lambda x: x.published_parsed if x.get('published_parsed') else time.localtime(0), reverse=True)
    return all_news

# --- 策略回測 ---
def run_advanced_backtest(df, initial_fund):
    capital = initial_fund
    position = 0 
    history = [] 
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        date = df['Date'].iloc[i]
        ma20 = df['MA20'].iloc[i]
        
        # MACD
        macd_curr = df['MACD'].iloc[i]
        signal_curr = df['Signal'].iloc[i]
        macd_prev = df['MACD'].iloc[i-1]
        signal_prev = df['Signal'].iloc[i-1]
        
        # KDJ
        k_curr, d_curr = df['K'].iloc[i], df['D'].iloc[i]
        k_prev, d_prev = df['K'].iloc[i-1], df['D'].iloc[i-1]
        
        # 訊號
        macd_gold = (macd_prev < signal_prev) and (macd_curr > signal_curr)
        macd_death = (macd_prev > signal_prev) and (macd_curr < signal_curr)
        kdj_gold = (k_prev < d_prev) and (k_curr > d_curr) and (k_curr < 50) # 低檔金叉才算
        kdj_death = (k_prev > d_prev) and (k_curr < d_curr) and (k_curr > 80) # 高檔死叉才算
        trend_up = price > ma20
        trend_down = price < ma20
        
        # 買進：趨勢向上 + (MACD金叉 或 KDJ低檔金叉)
        if position == 0 and trend_up and (macd_gold or kdj_gold):
            position = capital / price
            capital = 0
            reason = []
            if macd_gold: reason.append("MACD金叉")
            if kdj_gold: reason.append("KDJ金叉")
            history.append(f"🔴 {date.strftime('%Y-%m-%d')} 買進 @ {price:.2f} | 依據: 站上月線 & {'+'.join(reason)}")
            
        # 賣出：趨勢向下 或 (MACD死叉 或 KDJ高檔死叉)
        elif position > 0 and (trend_down or macd_death or kdj_death):
            capital = position * price
            position = 0
            reason = []
            if trend_down: reason.append("跌破月線")
            if macd_death: reason.append("MACD死叉")
            if kdj_death: reason.append("KDJ死叉")
            history.append(f"🟢 {date.strftime('%Y-%m-%d')} 賣出 @ {price:.2f} | 依據: {'+'.join(reason)}")

    final_value = capital
    if position > 0:
        final_value = position * df['Close'].iloc[-1]
    return final_value, history

# --- 5. 主程式 ---
data = load_data(stock_id, date_range)

if data is not None and not data.empty:
    
    # 計算所有指標
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()
    data = calculate_kdj(data) # 計算 KDJ
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Hist'] = calculate_macd(data)

    stock_info = get_stock_info_safe(stock_id)
    name = stock_info.get('longName', stock_id)
    clean_name = name.split(' ')[0]
    current_price = data['Close'].iloc[-1]
    support_price, resistance_price = calculate_support_resistance(data)
    
    st.subheader(f"{name} ({stock_id})")

    # === 1. 關鍵價位看板 ===
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    target_price = stock_info.get('targetMeanPrice')
    target_str = f"{target_price:.2f}" if target_price else "N/A"
    
    col_p1.metric("目前股價", f"{current_price:.2f}")
    col_p2.metric("壓力 (前高)", f"{resistance_price:.2f}", f"差 {resistance_price - current_price:.2f}")
    col_p3.metric("支撐 (前低)", f"{support_price:.2f}", f"差 {support_price - current_price:.2f}", delta_color="inverse")
    col_p4.metric("法人目標價", target_str)
    st.divider()

    # === 2. 綜合 AI 診斷 (K線 + 技術 + 基本) ===
    last_idx = -1
    last_ma20 = data['MA20'].iloc[last_idx]
    last_macd = data['MACD'].iloc[last_idx]; last_signal = data['Signal'].iloc[last_idx]
    last_k = data['K'].iloc[last_idx]; last_d = data['D'].iloc[last_idx]
    last_rsi = data['RSI'].iloc[last_idx]
    
    # K線型態識別
    open_p, high_p, low_p, close_p = data['Open'].iloc[last_idx], data['High'].iloc[last_idx], data['Low'].iloc[last_idx], data['Close'].iloc[last_idx]
    k_pattern, k_score = identify_candlestick_pattern(open_p, high_p, low_p, close_p)
    
    # 基本面簡單檢核 (EPS)
    eps = stock_info.get('trailingEps')
    fund_score = 0
    fund_msg = "基本面數據不足"
    if eps is not None:
        if eps > 0: fund_score = 1; fund_msg = f"EPS 為正 ({eps:.2f})，具獲利能力"
        else: fund_score = -1; fund_msg = f"EPS 為負 ({eps:.2f})，虧損中"

    score = 0
    reasons = []
    
    # 評分邏輯
    # 1. 趨勢
    if current_price > last_ma20: score += 2; reasons.append("✅ [趨勢] 站上月線 (短多) +2")
    else: score -= 2; reasons.append("🔻 [趨勢] 跌破月線 (短空) -2")
    
    # 2. 動能 (MACD)
    if last_macd > last_signal: score += 2; reasons.append("✅ [動能] MACD 黃金交叉 +2")
    else: score -= 2; reasons.append("🔻 [動能] MACD 死亡交叉 -2")
    
    # 3. KDJ
    if last_k > last_d: score += 1; reasons.append("✅ [波段] KDJ 黃金交叉 +1")
    else: score -= 1; reasons.append("🔻 [波段] KDJ 死亡交叉 -1")
    
    # 4. K線型態
    score += k_score
    if k_score != 0: reasons.append(f"{'✅' if k_score>0 else '🔻'} [型態] {k_pattern} {k_score:+}")
    else: reasons.append(f"ℹ️ [型態] {k_pattern}")
    
    # 5. 基本面
    score += fund_score
    reasons.append(f"{'✅' if fund_score>0 else '🔻'} [基本] {fund_msg} {fund_score:+}")

    bg_color = "#fff3cd"
    suggestion = "觀望"
    if score >= 4: bg_color = "#d4edda"; suggestion = "強烈買進訊號 🚀"
    elif score >= 1: bg_color = "#e2e3e5"; suggestion = "偏多操作 (謹慎)"
    elif score <= -4: bg_color = "#f8d7da"; suggestion = "強烈賣出訊號 🐻"
    elif score <= -1: bg_color = "#f8d7da"; suggestion = "偏空操作 (避險)"

    st.markdown(f"""<div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border-left: 5px solid #666;">
        <h3 style="margin:0;">🤖 AI 綜合診斷：{suggestion} (總分: {score})</h3>
        <hr style="margin: 10px 0;">
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            {''.join([f'<span style="background: rgba(255,255,255,0.5); padding: 2px 8px; border-radius: 4px;">{r}</span>' for r in reasons])}
        </div>
    </div>""", unsafe_allow_html=True)

    # === 分頁 ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 技術分析 (含KDJ)", "🏢 基本面分析", "🕵️ 籌碼面分析", "💰 策略回測", "📰 新聞情報"])

    with tab1:
        # 4層圖表: K線, MACD, KDJ, RSI
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.03)
        
        # K線
        fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], line=dict(color='blue', width=1), name='MA20'), row=1, col=1)
        fig.add_hline(y=resistance_price, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_hline(y=support_price, line_dash="dot", line_color="green", row=1, col=1)
        
        # MACD
        colors = ['red' if val < 0 else 'green' for val in data['Hist']]
        fig.add_trace(go.Bar(x=data['Date'], y=data['Hist'], marker_color=colors, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], line=dict(color='black', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal'], line=dict(color='red', width=1), name='DEM'), row=2, col=1)
        
        # KDJ (新加入)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['K'], line=dict(color='orange', width=1), name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['D'], line=dict(color='blue', width=1), name='D'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['J'], line=dict(color='purple', width=1), name='J'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1) # 超買
        fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1) # 超賣
        
        # RSI
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], line=dict(color='purple', width=2), name='RSI'), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
        
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(xaxis_rangeslider_visible=False, height=900, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### 🏢 基本面分析 (Fundamental Analysis)")
        st.caption("結合「量化數據」與「質化描述」的雙軌分析")
        
        # 量化數據 (Quantitative)
        st.subheader("1. 財務關鍵數據 (量化)")
        q1, q2, q3, q4 = st.columns(4)
        
        # 嘗試獲取數據，若無則顯示 N/A
        roe = stock_info.get('returnOnEquity', None)
        roe_str = f"{roe*100:.2f}%" if roe else "N/A"
        
        gross_margin = stock_info.get('grossMargins', None)
        gm_str = f"{gross_margin*100:.2f}%" if gross_margin else "N/A"
        
        rev_growth = stock_info.get('revenueGrowth', None)
        rev_str = f"{rev_growth*100:.2f}%" if rev_growth else "N/A"
        
        q1.metric("每股盈餘 (EPS)", f"{eps:.2f}" if eps else "N/A")
        q2.metric("股東權益報酬率 (ROE)", roe_str)
        q3.metric("毛利率", gm_str)
        q4.metric("營收成長率 (YoY)", rev_str)
        
        st.divider()
        
        # 質化描述 (Qualitative)
        st.subheader("2. 公司簡介與產業地位 (質化)")
        summary = stock_info.get('longBusinessSummary', '暫無公司描述資料。')
        # 簡單翻譯提示 (因 Yahoo API 多為英文)
        if summary and summary != '暫無公司描述資料。':
            st.info("💡 小提示：下方為公司官方業務描述，可透過翻譯工具了解其「產業地位」與「技術優勢」。")
            st.write(summary)
        else:
            st.warning("無法取得公司質化描述資料。")

    with tab3:
        clean_id_str = stock_id.split('.')[0].strip()
        st.write("### 🕵️ 籌碼面分析 (Chip Analysis)")
        st.markdown("""
        **籌碼分析的核心邏輯：**
        * **跟隨聰明錢 (Smart Money)**：大戶、法人（外資、投信）通常擁有比散戶更多的資訊優勢與資金控盤能力。
        * **觀察籌碼流向**：當「千張大戶」持股增加，而「散戶」持股減少時，代表籌碼趨於集中，有利股價上漲。
        """)
        
        st.info(f"👇 點擊下方按鈕，直達 {clean_id_str} 的籌碼核心數據戰場：")
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.link_button("🏦 三大法人買賣超 (玩股網)", f"https://www.wantgoo.com/stock/{clean_id_str}/institutional-investors")
            st.caption("觀察外資、投信是否連續買超")
        with c2: 
            st.link_button("🏯 主力進出與分點 (玩股網)", f"https://www.wantgoo.com/stock/{clean_id_str}/major-investors")
            st.caption("是否有特定券商在吃貨？")
        with c3: 
            st.link_button("📊 千張大戶持股 (Goodinfo)", f"https://goodinfo.tw/tw/ShowK_Chart.asp?STOCK_ID={clean_id_str}&CHT_CAT=DATE")
            st.caption("大戶持股比例 vs 股價走勢")

    with tab4:
        st.write("### 💰 綜合策略回測 (Trend + MACD + KDJ)")
        st.caption("策略邏輯：當股價站上月線(趨勢多) 且 (MACD金叉 或 KDJ低檔金叉) 時買進。")
        
        final_val, trade_log = run_advanced_backtest(data, initial_capital)
        profit = final_val - initial_capital
        ret_rate = (profit / initial_capital) * 100
        bh_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        
        c1, c2 = st.columns(2)
        c1.metric("策略回報", f"{ret_rate:.2f}%")
        c2.metric("持有回報", f"{bh_return:.2f}%")
        
        with st.expander("📄 查看詳細交易紀錄", expanded=True):
            if trade_log:
                for log in trade_log: st.markdown(log)
            else: st.info("無交易訊號")

    with tab5:
        st.write(f"### 🔥 {clean_name} 純淨情報")
        col_src1, col_src2, col_src3 = st.columns(3)
        clean_id_str = stock_id.split('.')[0].strip()
        with col_src1: st.link_button("鉅亨網", f"https://www.cnyes.com/search/news?q={clean_name}")
        with col_src2: st.link_button("Yahoo", f"https://tw.stock.yahoo.com/quote/{clean_id_str}/news")
        with col_src3: st.link_button("MoneyDJ", f"https://www.moneydj.com/KMDJ/search/list.aspx?_Query_={clean_name}")
        st.divider()

        news_items = get_aggregated_news(stock_id, clean_name)
        if news_items:
            for item in news_items[:30]:
                title = item.title
                link = item.link
                try: dt = datetime(*item.published_parsed[:6]); published_time = dt.strftime('%m-%d %H:%M')
                except: published_time = "最近"
                
                tag = ""
                t_low = title.lower()
                if any(x in t_low for x in ['漲','增','高','旺','強','多']): tag = "🔴"
                elif any(x in t_low for x in ['跌','減','低','弱','空','賣']): tag = "🟢"
                
                st.markdown(f"**{published_time}** {tag} [{title}]({link})")
                st.markdown("---")
        else:
            st.warning("暫無相關新聞")

else:

    st.error("查無資料，請確認代號。") #streamlit run app.py
