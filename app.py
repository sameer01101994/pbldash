import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

"""
Sales Insights Dashboard
========================
• Dependent variable  : total value (Sales)
• Independent vars    : unit price, rating, channel, paymentmethod, color, shoetype, size
• Visualisations      : 20+ (see tabs below)
• Target Audience     : Director‑level decision makers & HR/Stakeholders

Place this file (`app.py`) together with `IA_PBL_DA_Sameer.csv` and `requirements.txt`
in a GitHub repo, then deploy on Streamlit Community Cloud.
"""

st.set_page_config(page_title='Sales Insights Dashboard', layout='wide', page_icon='💰')

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV/Excel dataset with safe‑guards."""
    if not Path(file_path).exists():
        return pd.DataFrame()
    try:
        df_temp = pd.read_csv(file_path)
    except Exception:
        df_temp = pd.read_excel(file_path)
    df_temp.columns = df_temp.columns.str.strip().str.lower().str.replace(' ', '_')
    return df_temp

# Sidebar — dataset upload (default loads the sample in repo root)
st.sidebar.header('📂 Dataset')
user_file = st.sidebar.file_uploader('Upload IA_PBL_DA_Sameer.csv', type=['csv', 'xlsx'])

DATA_PATH = 'IA_PBL_DA_Sameer.csv'
if user_file:
    df = load_data(user_file)
else:
    df = load_data(DATA_PATH)

if df.empty:
    st.error('Dataset not found or empty. Please upload a valid file to continue.')
    st.stop()

# ------------------ COLUMN MAPPING ------------------
cols = df.columns.tolist()
req_cols = {
    'total_value': ['total_value', 'totalvalue', 'sales', 'total sales', 'total amount'],
    'unit_price': ['unit_price', 'unitprice', 'price'],
    'rating': ['rating', 'review_rating', 'customer_rating'],
    'channel': ['channel', 'sales_channel'],
    'paymentmethod': ['paymentmethod', 'payment_method'],
    'color': ['color', 'colour'],
    'shoetype': ['shoetype', 'shoe_type', 'product_type'],
    'size': ['size', 'shoe_size']
}

mapped = {}
for key, aliases in req_cols.items():
    for alias in aliases:
        if alias in cols:
            mapped[key] = alias
            break

missing = [k for k in req_cols if k not in mapped]
if missing:
    st.warning(f"Missing expected columns: {', '.join(missing)}. Functionality may be limited.")

# Helper to fetch mapped column safely
def col(name):
    return mapped.get(name, None)

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header('🔎 Filters')

# Categorical multi‑select filters
for cat in ['channel', 'paymentmethod', 'color', 'shoetype', 'size']:
    c = col(cat)
    if c and df[c].dtype == 'O':
        opts = df[c].dropna().unique().tolist()
        sel = st.sidebar.multiselect(f'{cat.replace("paymentmethod","payment method").title()}', opts, default=opts)
        if sel and set(sel) != set(opts):
            df = df[df[c].isin(sel)]

# Numeric range sliders
for num in ['unit_price', 'rating', 'total_value']:
    n = col(num)
    if n and pd.api.types.is_numeric_dtype(df[n]):
        mn, mx = float(df[n].min()), float(df[n].max())
        rng = st.sidebar.slider(f'{num.replace("_", " ").title()} Range', mn, mx, (mn, mx))
        df = df[df[n].between(*rng)]

# ------------------ KPI METRICS ------------------
st.markdown('## 💰 Sales Insights Dashboard')

k1, k2, k3, k4 = st.columns(4)

total_col = col('total_value')
price_col = col('unit_price')
rating_col = col('rating')

if total_col:
    total_sales = df[total_col].sum()
    k1.metric('Total Sales', f"${total_sales:,.0f}")

if price_col:
    avg_price = df[price_col].mean()
    k2.metric('Average Unit Price', f"${avg_price:,.2f}")

if rating_col:
    avg_rating = df[rating_col].mean()
    k3.metric('Average Rating', f"{avg_rating:.2f} ⭐")

k4.metric('Transactions', f"{len(df):,}")

# ------------------ Helper ------------------
def explain(text: str):
    st.markdown(f"<div style='font-size:0.9rem;color:#6c757d;margin-bottom:0.3rem'>{text}</div>", unsafe_allow_html=True)

def vbar(df_, x, y, color=None, barmode='group'):
    fig = px.bar(df_, x=x, y=y, color=color, barmode=barmode, text_auto='.2s', height=430)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ TABS ------------------
(tab_overview,
 tab_price_rating,
 tab_product,
 tab_payment_channel,
 tab_corr,
 tab_data) = st.tabs([
    'Overview', 'Pricing & Rating', 'Product Attributes', 'Payment & Channel', 'Correlation', 'Raw Data'])

# ------- 1. OVERVIEW (5 charts) -------
with tab_overview:
    if total_col:
        explain('**Chart 1**: Distribution of transaction sales amounts.')
        st.plotly_chart(px.histogram(df, x=total_col, nbins=40, height=430), use_container_width=True)

    if col('channel') and total_col:
        explain('**Chart 2**: Proportion of sales across channels.')
        sales_by_channel = df.groupby(col('channel'))[total_col].sum().sort_values(ascending=False)
        st.plotly_chart(px.pie(values=sales_by_channel, names=sales_by_channel.index, hole=0.35, height=430), use_container_width=True)

    if col('paymentmethod') and total_col:
        explain('**Chart 3**: Average order value by payment method to gauge purchase behaviour.')
        aov_pm = df.groupby(col('paymentmethod'))[total_col].mean().reset_index()
        vbar(aov_pm, x=col('paymentmethod'), y=total_col)

    if rating_col and total_col:
        explain('**Chart 4**: Relationship between customer rating and basket size.')
        st.plotly_chart(px.box(df, x=rating_col, y=total_col, points='all', height=430), use_container_width=True)

    if price_col:
        explain('**Chart 5**: Distribution of unit prices across all products.')
        st.plotly_chart(px.histogram(df, x=price_col, nbins=30, height=430), use_container_width=True)

# ------- 2. PRICING & RATING (5 charts) -------
with tab_price_rating:
    if price_col and total_col:
        explain('**Chart 6**: Relationship between unit price and total sales per transaction.')
        # Removed trendline="ols" to avoid statsmodels dependency
        st.plotly_chart(px.scatter(df, x=price_col, y=total_col, color=col('channel'), height=430), use_container_width=True)

    if price_col and col('channel'):
        explain('**Chart 7**: Pricing variations by sales channel.')
        st.plotly_chart(px.box(df, x=col('channel'), y=price_col, points='all', height=430), use_container_width=True)

    if rating_col:
        explain('**Chart 8**: Overall distribution of customer ratings.')
        st.plotly_chart(px.histogram(df, x=rating_col, nbins=10, height=430), use_container_width=True)

    if rating_col and col('channel'):
        explain('**Chart 9**: Which channel garners better ratings?')
        avg_rating_ch = df.groupby(col('channel'))[rating_col].mean().reset_index()
        vbar(avg_rating_ch, x=col('channel'), y=rating_col)

    if price_col and rating_col:
        explain('**Chart 10**: Density heatmap of unit price vs rating.')
        st.plotly_chart(px.density_heatmap(df, x=price_col, y=rating_col, nbinsx=30, nbinsy=10, height=430), use_container_width=True)

# ------- 3. PRODUCT ATTRIBUTES (5 charts) -------
with tab_product:
    if col('color') and total_col:
        explain('**Chart 11**: Sales volume by product colour.')
        sales_color = df.groupby(col('color'))[total_col].sum().reset_index().sort_values(total_col, ascending=False)
        vbar(sales_color, x=col('color'), y=total_col)

    if col('shoetype') and total_col:
        explain('**Chart 12**: Popularity of shoe categories.')
        sales_st = df.groupby(col('shoetype'))[total_col].sum().reset_index().sort_values(total_col, ascending=False)
        vbar(sales_st, x=col('shoetype'), y=total_col)

    if col('size') and total_col:
        explain('**Chart 13**: Which sizes contribute most to revenue?')
        sales_size = df.groupby(col('size'))[total_col].sum().reset_index()
        vbar(sales_size, x=col('size'), y=total_col)

    if col('color') and col('channel') and total_col:
        explain('**Chart 14**: Colour preferences segmented by sales channel.')
        pivot_cc = df.pivot_table(index=col('color'), columns=col('channel'), values=total_col, aggfunc='sum').fillna(0)
        fig = go.Figure()
        for ch in pivot_cc.columns:
            fig.add_bar(name=ch, x=pivot_cc.index, y=pivot_cc[ch])
        fig.update_layout(barmode='stack', height=430)
        st.plotly_chart(fig, use_container_width=True)

    if col('shoetype') and col('color') and total_col:
        explain('**Chart 15**: Hierarchical view of sales by shoe type and colour.')
        treemap_df = df.groupby([col('shoetype'), col('color')])[total_col].sum().reset_index()
        st.plotly_chart(px.treemap(treemap_df, path=[col('shoetype'), col
