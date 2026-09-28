import streamlit as st
import feedparser
import urllib.parse
import pandas as pd
import yfinance as yf
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="Automated Multi-Index Analytics Dashboard", layout="wide")

# VADER Lexicon డౌన్‌లోడ్
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ప్రధాన ఇండెక్స్‌లు మరియు అధికారిక NSE CSV మూలాలు
INDEX_CONFIG = {
    "NIFTY 50": {
        "ticker": "^NSEI",
        "search_term": "Nifty 50 share market India",
        "csv_url": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    },
    "NIFTY NEXT 50": {
        "ticker": "^NSMIDCP", # లేదా ప్రాతినిధ్య ఇండెక్స్
        "search_term": "Nifty Next 50 stock market India",
        "csv_url": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv"
    },
    "NIFTY BANK": {
        "ticker": "^NSEBANK",
        "search_term": "Bank Nifty share market India",
        "csv_url": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv"
    },
    "NIFTY MIDCAP 50": {
        "ticker": "^NSEMDCP50",
        "search_term": "Nifty Midcap 50 share market India",
        "csv_url": "https://archives.nseindia.com/content/indices/ind_niftymidcap50list.csv"
    },
    "NIFTY IT": {
        "ticker": "^CNXIT",
        "search_term": "Nifty IT share market India",
        "csv_url": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv"
    }
}

# 1. ఇండెక్స్ లోని స్టాక్స్ లిస్ట్‌ను ఆటోమేటిక్‌గా డౌన్‌లోడ్ చేసుకునే ఫంక్షన్
@st.cache_data(ttl=86400) # రోజుకు ఒకసారి మాత్రమే లోడ్ అవుతుంది
def get_index_constituents(csv_url):
    """NSE అధికారిక CSV నుండి స్టాక్స్ లిస్ట్‌ను ఆటోమేటిక్‌గా ఫెచ్ చేస్తుంది"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        df = pd.read_csv(csv_url, storage_options=headers)
        # Yahoo Finance ఫార్మాట్ కోసం '.NS' జోడించడం
        symbols = [f"{sym.strip()}.NS" for sym in df['Symbol'].dropna()]
        return symbols
    except Exception:
        # బ్యాకప్: ఒకవేళ సర్వర్ ఆంక్షలు ఉంటే డిఫాల్ట్ కోర్ లిస్ట్
        return ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "SBIN.NS"]

# 2. అన్ని స్టాక్స్ లైవ్ ధరలు మరియు ఆటోమేటిక్ వెయిటేజ్ లెక్కింపు
@st.cache_data(ttl=300)
def fetch_bulk_stock_data(symbol_list):
    """ఒకేసారి బల్క్‌గా అన్ని స్టాక్స్ ధరలు & మార్కెట్ క్యాప్ ఆధారంగా ఆటో వెయిటేజ్ లెక్కిస్తుంది"""
    if not symbol_list:
        return pd.DataFrame()
        
    tickers_str = " ".join(symbol_list)
    tickers_obj = yf.Tickers(tickers_str)
    
    # గత 15 రోజుల చారిత్రక డేటా ఒకే నెట్‌వర్క్ కాల్‌లో డౌన్‌లోడ్
    hist_data = yf.download(tickers_str, period="15d", group_by='ticker', threads=True, progress=False)
    
    stock_rows = []
    
    for sym in symbol_list:
        clean_name = sym.replace(".NS", "")
        try:
            # ధర మరియు మార్పు లెక్కింపు
            sub_df = hist_data[sym] if len(symbol_list) > 1 else hist_data
            sub_df = sub_df.dropna(how='all')
            if len(sub_df) < 2:
                continue
            
            cmp = round(sub_df['Close'].iloc[-1], 2)
            prev_close = sub_df['Close'].iloc[-2]
            chg_pct = round(((cmp - prev_close) / prev_close) * 100, 2)
            
            # 14-రోజుల సగటు శ్రేణి (ATR)
            hl = sub_df['High'] - sub_df['Low']
            atr = round(hl.mean(), 2)
            
            # ఫ్రీ-ఫ్లోట్ మార్కెట్ క్యాప్ (వెయిటేజ్ కోసం)
            m_cap = 1.0
            try:
                fast_info = tickers_obj.tickers[sym].fast_info
                m_cap = fast_info.market_cap or 1.0
            except Exception:
                m_cap = 1.0

            stock_rows.append({
                "Stock": clean_name,
                "Symbol": sym,
                "CMP": cmp,
                "Change%": chg_pct,
                "ATR": atr,
                "MarketCap": m_cap
            })
        except Exception:
            continue
            
    df_all = pd.DataFrame(stock_rows)
    
    # ఆటోమేటిక్ వెయిటేజ్ ఫార్ములా: (కంపెనీ M-Cap / మొత్తం ఇండెక్స్ M-Cap) * 100
    if not df_all.empty and df_all['MarketCap'].sum() > 0:
        total_mcap = df_all['MarketCap'].sum()
        df_all['Weight%'] = round((df_all['MarketCap'] / total_mcap) * 100, 2)
    else:
        df_all['Weight%'] = round(100.0 / len(df_all), 2)
        
    return df_all.sort_values(by='Weight%', ascending=False)

# 3. ఇండెక్స్ లెవెల్ టెక్నికల్ డేటా
@st.cache_data(ttl=300)
def get_index_tech(ticker_symbol):
    try:
        data = yf.Ticker(ticker_symbol).history(period="15d")
        if data.empty or len(data) < 2:
            return None
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        cmp = round(latest['Close'], 2)
        day_change = round(((cmp - prev['Close']) / prev['Close']) * 100, 2)
        
        atr = round((data['High'] - data['Low']).mean(), 2)
        high, low, close = prev['High'], prev['Low'], prev['Close']
        pivot = (high + low + close) / 3
        
        return {
            "CMP": cmp, "Change%": day_change, "ATR": atr,
            "MaxUp": round(cmp + atr, 2), "MaxDown": round(cmp - atr, 2),
            "Pivot": round(pivot, 2),
            "R1": round((2 * pivot) - low, 2),
            "R2": round(pivot + (high - low), 2),
            "S1": round((2 * pivot) - high, 2),
            "S2": round(pivot - (high - low), 2)
        }
    except Exception:
        return None

# 4. వార్తల ఇంపాక్ట్ మరియు సెంటిమెంట్ విశ్లేషణ
def analyze_news_impact(score, title):
    title_lower = title.lower()
    high_impact_keywords = ['crash', 'plunge', 'surge', 'rally', 'rbi', 'inflation', 'fed', 'rate hike', 'rate cut', 'war', 'fall', 'hit']
    is_high_impact = any(k in title_lower for k in high_impact_keywords) or abs(score) >= 0.45
    impact_level = "High Impact ⚡" if is_high_impact else "Moderate Impact 🔹"
    
    if score >= 0.2:
        interpretation = "సానుకూల వార్త: ఇండెక్స్‌లో కొనుగోళ్ల మద్దతు లేదా రికవరీకి దోహదపడవచ్చు."
    elif score <= -0.2:
        interpretation = "ప్రతికూల వార్త: ప్రాఫిట్ బుకింగ్ లేదా స్వల్పకాలిక అమ్మకాల ఒత్తిడి సూచన."
    else:
        interpretation = "తటస్థ పరిణామం: పెద్ద మార్పు లేకుండా రేంజ్‌బౌండ్ కదలికలకు అవకాశం."
    return impact_level, interpretation

@st.cache_data(ttl=300)
def fetch_news_and_sentiment(query_term, limit=4):
    encoded_query = urllib.parse.quote(query_term)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    news_list, total_score = [], 0.0
    
    for entry in feed.entries[:limit]:
        title = entry.title
        link = entry.link
        published_parsed = getattr(entry, 'published_parsed', None)
        pub_date = datetime(*published_parsed[:6]).strftime("%d-%b-%Y | %I:%M %p") if published_parsed else datetime.now().strftime("%d-%b-%Y | %I:%M %p")
        score = sia.polarity_scores(title)['compound']
        total_score += score
        impact, analysis = analyze_news_impact(score, title)
        
        news_list.append({
            "DateTime": pub_date, "Title": title, "Score": round(score, 2),
            "Impact": impact, "Analysis": analysis, "URL": link
        })
    avg_score = round(total_score / len(news_list), 3) if news_list else 0.0
    return news_list, avg_score

# --- UI ప్రారంభం ---
st.title("🏛️ ఆటోమేటెడ్ మల్టీ-ఇండెక్స్ & లైవ్ న్యూస్ అనలిటిక్స్")
st.caption("అన్ని కంపెనీల ఆటోమేటిక్ వెయిటేజ్, టెక్నికల్ లెవెల్స్ మరియు రియల్-టైమ్ వార్తల ప్రభావం.")

# సైడ్‌బార్‌లో ఇండెక్స్ ఎంపిక
selected_index = st.sidebar.selectbox("ఇండెక్స్ ఎంచుకోండి (Select Index)", list(INDEX_CONFIG.keys()))
news_limit = st.sidebar.slider("వార్తల సంఖ్య (News Count)", 2, 8, 4)

if st.sidebar.button("🔄 డేటా రీఫ్రెష్ చేయండి"):
    st.cache_data.clear()
    st.rerun()

cfg = INDEX_CONFIG[selected_index]

with st.spinner(f"{selected_index} లిస్టెడ్ స్టాక్స్, ఆటో-వెయిటేజ్ మరియు వార్తలను విశ్లేషిస్తోంది..."):
    # 1. స్టాక్స్ లిస్ట్ ఆటోమేటిక్‌గా తీసుకోవడం
    constituents = get_index_constituents(cfg["csv_url"])
    
    # 2. బల్క్ స్టాక్ డేటా & ఆటో వెయిటేజ్
    df_stocks = fetch_bulk_stock_data(constituents)
    
    # 3. ఇండెక్స్ టెక్నికల్స్ & వార్తలు
    idx_tech = get_index_tech(cfg["ticker"])
    idx_news, idx_news_score = fetch_news_and_sentiment(cfg["search_term"], limit=news_limit)

# వెయిటెడ్ మూమెంటం లెక్కింపు
if not df_stocks.empty:
    composite_momentum = round((df_stocks['Change%'] * df_stocks['Weight%']).sum() / df_stocks['Weight%'].sum(), 2)
    bullish_cnt = len(df_stocks[df_stocks['Change%'] > 0])
    bearish_cnt = len(df_stocks[df_stocks['Change%'] < 0])
else:
    composite_momentum, bullish_cnt, bearish_cnt = 0.0, 0, 0

# 1. సారాంశం మెట్రిక్స్
st.subheader(f"📊 {selected_index} ట్రెండ్ & సెంటిమెంట్ సారాంశం")
m1, m2, m3, m4 = st.columns(4)

cmp_val = f"{idx_tech['CMP']}" if idx_tech else "N/A"
chg_val = f"{idx_tech['Change%']}%" if idx_tech else "N/A"
m1.metric("ఇండెక్స్ CMP", cmp_val, chg_val)
m2.metric("వెయిటెడ్ మూమెంటం (All Stocks)", f"{composite_momentum}%")
news_tag = "పాజిటివ్ 🟢" if idx_news_score >= 0.05 else ("నెగెటివ్ 🔴" if idx_news_score <= -0.05 else "తటస్థం ⚪")
m3.metric("వార్తల సెంటిమెంట్", news_tag, f"స్కోర్: {idx_news_score}")
m4.metric("మార్కెట్ బ్రెడ్త్ (A/D)", f"{bullish_cnt} లాభాల్లో / {bearish_cnt} నష్టాల్లో")

if idx_tech:
    st.markdown(f"""
    | సపోర్ట్ జోన్స్ (Support) | రోజువారీ అంచనా పరిధి (Expected Range) | రెసిస్టెన్స్ జోన్స్ (Resistance) |
    | :---: | :---: | :---: |
    | **S2:** {idx_tech['S2']} &nbsp;&nbsp;|&nbsp;&nbsp; **S1:** {idx_tech['S1']} | **Max Down:** {idx_tech['MaxDown']} <br> **Max Up:** {idx_tech['MaxUp']} | **R1:** {idx_tech['R1']} &nbsp;&nbsp;|&nbsp;&nbsp; **R2:** {idx_tech['R2']} |
    """, unsafe_allow_html=True)

st.divider()

# 2. తాజా వార్తల ప్రభావ విశ్లేషణ
st.subheader(f"📰 {selected_index} తాజా వార్తలు & ప్రభావ విశ్లేషణ (Latest News Impact)")
if idx_news:
    for item in idx_news:
        icon = "🟢" if item['Score'] >= 0.05 else ("🔴" if item['Score'] <= -0.05 else "⚪")
        with st.expander(f"{icon} [{item['DateTime']}] — {item['Impact']} | స్కోర్: {item['Score']}"):
            st.markdown(f"**హెడ్‌లైన్:** [{item['Title']}]({item['URL']})")
            st.info(f"📌 **విశ్లేషణ:** {item['Analysis']}")

st.divider()

# 3. ఆటోమేటిక్ వెయిటేజ్‌తో స్టాక్స్ టేబుల్
st.subheader(f"🧱 {selected_index} లోని అన్ని కంపెనీలు & ఆటో-వెయిటేజ్ ({len(df_stocks)} Stocks)")
if not df_stocks.empty:
    display_df = df_stocks[['Stock', 'Weight%', 'CMP', 'Change%', 'ATR']].copy()
    
    def color_change(val):
        color = '#28a745' if val > 0 else ('#dc3545' if val < 0 else '#6c757d')
        return f'color: {color}; font-weight: bold;'

    styled_df = display_df.style.map(color_change, subset=['Change%']).format({
        'Weight%': '{:.2f}%',
        'CMP': '₹{:.2f}',
        'Change%': '{:+.2f}%',
        'ATR': '{:.2f}'
    })
    st.dataframe(styled_df, use_container_width=True, height=450)