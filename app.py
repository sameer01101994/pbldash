import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title='Sales Insights Dashboard', layout='wide', page_icon='💰')

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    if not Path(file_path).exists():
        return pd.DataFrame()
    try:
        df_temp = pd.read_csv(file_path)
    except Exception:
        df_temp = pd.read_excel(file_path)
    df_temp.columns = df_temp.columns.str.strip().str.lower().str.replace(' ', '_')
    return df_temp

st.sidebar.header('📂 Dataset')
user_file = st.sidebar.file_uploader('Upload IA_PBL_DA_Sameer.csv', type=['csv', 'xlsx'])

DATA_PATH = 'IA_PBL_DA_Sameer.csv'
df = load_data(user_file) if user_file else load_data(DATA_PATH)
if df.empty:
    st.error('Dataset not found or empty. Please upload a valid file to continue.')
    st.stop()

# COLUMN MAPPING
cols = df.columns.tolist()
req_cols = {
    'total_value': ['total_value', 'totalvalue', 'sales'],
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

def col(name): return mapped.get(name, None)
missing = [k for k in req_cols if k not in mapped]
if missing:
    st.warning(f"Missing expected columns: {', '.join(missing)}")

# SIDEBAR FILTERS
st.sidebar.header('🔎 Filters')
for cat in ['channel', 'paymentmethod', 'color', 'shoetype', 'size']:
    c = col(cat)
    if c and df[c].dtype == 'O':
        opts = df[c].dropna().unique().tolist()
        sel = st.sidebar.multiselect(f'{cat.title()}', opts, default=opts)
        if sel and set(sel) != set(opts):
            df = df[df[c].isin(sel)]

for num in ['unit_price', 'rating', 'total_value']:
    n = col(num)
    if n and pd.api.types.is_numeric_dtype(df[n]):
        mn, mx = float(df[n].min()), float(df[n].max())
        rng = st.sidebar.slider(f'{num.replace("_", " ").title()} Range', mn, mx, (mn, mx))
        df = df[df[n].between(*rng)]

# KPI METRICS
st.markdown('## 💰 Sales Insights Dashboard')
k1, k2, k3, k4 = st.columns(4)
total_col, price_col, rating_col = col('total_value'), col('unit_price'), col('rating')
if total_col: k1.metric('Total Sales', f"${df[total_col].sum():,.0f}")
if price_col: k2.metric('Average Unit Price', f"${df[price_col].mean():.2f}")
if rating_col: k3.metric('Average Rating', f"{df[rating_col].mean():.2f} ⭐")
k4.metric('Transactions', f"{len(df):,}")

def explain(text: str):
    st.markdown(f"<div style='font-size:0.9rem;color:#6c757d;margin-bottom:0.3rem'>{text}</div>", unsafe_allow_html=True)

def vbar(df_, x, y, color=None, barmode='group'):
    fig = px.bar(df_, x=x, y=y, color=color, barmode=barmode, text_auto='.2s', height=430)
    st.plotly_chart(fig, use_container_width=True)

# TABS
(tab_overview, tab_price_rating, tab_product, tab_payment_channel, tab_corr, tab_data) = st.tabs([
    'Overview', 'Pricing & Rating', 'Product Attributes', 'Payment & Channel', 'Correlation', 'Raw Data'])

# OVERVIEW TAB
with tab_overview:
    if total_col:
        explain('**Chart 1**: Distribution of transaction sales amounts.')
        st.plotly_chart(px.histogram(df, x=total_col, nbins=40, height=430), use_container_width=True)
    if col('channel') and total_col:
        explain('**Chart 2**: Sales proportion across channels.')
        data = df.groupby(col('channel'))[total_col].sum().sort_values(ascending=False)
        st.plotly_chart(px.pie(values=data, names=data.index, hole=0.35, height=430), use_container_width=True)
    if col('paymentmethod') and total_col:
        explain('**Chart 3**: AOV by payment method.')
        avg = df.groupby(col('paymentmethod'))[total_col].mean().reset_index()
        vbar(avg, x=col('paymentmethod'), y=total_col)
    if rating_col and total_col:
        explain('**Chart 4**: Basket size vs customer rating.')
        st.plotly_chart(px.box(df, x=rating_col, y=total_col, points='all', height=430), use_container_width=True)
    if price_col:
        explain('**Chart 5**: Unit price distribution.')
        st.plotly_chart(px.histogram(df, x=price_col, nbins=30, height=430), use_container_width=True)

# PRICING & RATING TAB
with tab_price_rating:
    if price_col and total_col:
        explain('**Chart 6**: Unit price vs sales (scatter).')
        st.plotly_chart(px.scatter(df, x=price_col, y=total_col, color=col('channel'), height=430), use_container_width=True)
    if price_col and col('channel'):
        explain('**Chart 7**: Unit price by channel.')
        st.plotly_chart(px.box(df, x=col('channel'), y=price_col, points='all', height=430), use_container_width=True)
    if rating_col:
        explain('**Chart 8**: Rating distribution.')
        st.plotly_chart(px.histogram(df, x=rating_col, nbins=10, height=430), use_container_width=True)
    if rating_col and col('channel'):
        explain('**Chart 9**: Avg rating by channel.')
        avg = df.groupby(col('channel'))[rating_col].mean().reset_index()
        vbar(avg, x=col('channel'), y=rating_col)
    if price_col and rating_col:
        explain('**Chart 10**: Heatmap: price vs rating.')
        st.plotly_chart(px.density_heatmap(df, x=price_col, y=rating_col, nbinsx=30, nbinsy=10, height=430), use_container_width=True)

# PRODUCT ATTRIBUTES TAB
with tab_product:
    if col('color') and total_col:
        explain('**Chart 11**: Sales by product color.')
        data = df.groupby(col('color'))[total_col].sum().reset_index()
        vbar(data, x=col('color'), y=total_col)
    if col('shoetype') and total_col:
        explain('**Chart 12**: Sales by shoe type.')
        data = df.groupby(col('shoetype'))[total_col].sum().reset_index()
        vbar(data, x=col('shoetype'), y=total_col)
    if col('size') and total_col:
        explain('**Chart 13**: Sales by size.')
        data = df.groupby(col('size'))[total_col].sum().reset_index()
        vbar(data, x=col('size'), y=total_col)
    if col('color') and col('channel') and total_col:
        explain('**Chart 14**: Color preference by channel.')
        pivot = df.pivot_table(index=col('color'), columns=col('channel'), values=total_col, aggfunc='sum').fillna(0)
        fig = go.Figure()
        for ch in pivot.columns:
            fig.add_bar(name=ch, x=pivot.index, y=pivot[ch])
        fig.update_layout(barmode='stack', height=430)
        st.plotly_chart(fig, use_container_width=True)
    if col('shoetype') and col('color') and total_col:
        explain('**Chart 15**: Sales hierarchy by type and color.')
        treemap_df = df.groupby([col('shoetype'), col('color')])[total_col].sum().reset_index()
        st.plotly_chart(px.treemap(
            treemap_df,
            path=[col('shoetype'), col('color')],
            values=total_col,
            height=430
        ), use_container_width=True)

# CORRELATION TAB
with tab_corr:
    explain('**Chart 16**: Correlation heatmap for numeric variables.')
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect='auto', height=500, color_continuous_scale='RdBu_r'), use_container_width=True)

# RAW DATA TAB
with tab_data:
    explain('Raw dataset view. All applied filters affect this table.')
    st.dataframe(df, use_container_width=True)
