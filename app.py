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

Put this file (app.py) together with IA_PBL_DA_Sameer.csv and requirements.txt
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
# normalise to snake_case
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

# helper to fetch col safely
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

if col('total_value'):
    total_sales = df[col('total_value')].sum()
    k1.metric('Total Sales', f"${total_sales:,.0f}")

if col('unit_price'):
    avg_price = df[col('unit_price')].mean()
    k2.metric('Average Unit Price', f"${avg_price:,.2f}")

if col('rating'):
    avg_rating = df[col('rating')].mean()
    k3.metric('Average Rating', f"{avg_rating:.2f} ⭐")

k4.metric('Transactions', f"{len(df):,}")

# ------------------ Helper ------------------
def explain(text: str):
    st.markdown(f"<div style='font-size:0.9rem;color:#6c757d;margin-bottom:0.3rem'>{text}</div>", unsafe_allow_html=True)

def vbar(df_, x, y, color=None, barmode='group'):  # reusable bar plot
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
    # Chart 1: Sales distribution
    if col('total_value'):
        explain('**Chart 1**: Distribution of transaction sales amounts.')
        fig = px.histogram(df, x=col('total_value'), nbins=40, height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Sales share by channel
    if col('channel'):
        explain('**Chart 2**: Proportion of sales across channels.')
        sales_by_channel = df.groupby(col('channel'))[col('total_value')].sum().sort_values(ascending=False)
        fig = px.pie(values=sales_by_channel, names=sales_by_channel.index, hole=0.35, height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 3: Average sales by payment method
    if col('paymentmethod'):
        explain('**Chart 3**: Average order value by payment method to gauge purchase behaviour.')
        aov_pm = df.groupby(col('paymentmethod'))[col('total_value')].mean().reset_index()
        vbar(aov_pm, x=col('paymentmethod'), y=col('total_value'))

    # Chart 4: Box plot of sales vs rating buckets
    if col('rating') and col('total_value'):
        explain('**Chart 4**: Relationship between customer rating and basket size.')
        fig = px.box(df, x=col('rating'), y=col('total_value'), points='all', height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 5: Unit price distribution
    if col('unit_price'):
        explain('**Chart 5**: Distribution of unit prices across all products.')
        fig = px.histogram(df, x=col('unit_price'), nbins=30, height=430, color_discrete_sequence=['#00A'])
        st.plotly_chart(fig, use_container_width=True)

# ------- 2. PRICING & RATING (5 charts) -------
with tab_price_rating:
    # Chart 6: Scatter Unit Price vs Sales
    if col('unit_price') and col('total_value'):
        explain('**Chart 6**: Correlation between unit price and total sales per transaction.')
        fig = px.scatter(df, x=col('unit_price'), y=col('total_value'), color=col('channel'), trendline='ols', height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 7: Box of Unit Price by Channel
    if col('unit_price') and col('channel'):
        explain('**Chart 7**: Pricing variations by sales channel.')
        fig = px.box(df, x=col('channel'), y=col('unit_price'), points='all', height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 8: Distribution of Ratings
    if col('rating'):
        explain('**Chart 8**: Overall distribution of customer ratings.')
        fig = px.histogram(df, x=col('rating'), nbins=10, height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 9: Average Rating by Channel
    if col('rating') and col('channel'):
        explain('**Chart 9**: Which channel garners better ratings?')
        avg_rating_ch = df.groupby(col('channel'))[col('rating')].mean().reset_index()
        vbar(avg_rating_ch, x=col('channel'), y=col('rating'))

    # Chart 10: Heatmap Unit Price vs Rating density
    if col('unit_price') and col('rating'):
        explain('**Chart 10**: Density heatmap of unit price vs rating.')
        fig = px.density_heatmap(df, x=col('unit_price'), y=col('rating'), nbinsx=30, nbinsy=10, height=430)
        st.plotly_chart(fig, use_container_width=True)

# ------- 3. PRODUCT ATTRIBUTES (5 charts) -------
with tab_product:
    # Chart 11: Sales by Color
    if col('color'):
        explain('**Chart 11**: Sales volume by product colour.')
        sales_color = df.groupby(col('color'))[col('total_value')].sum().reset_index().sort_values(col('total_value'), ascending=False)
        vbar(sales_color, x=col('color'), y=col('total_value'))

    # Chart 12: Sales by Shoe Type
    if col('shoetype'):
        explain('**Chart 12**: Popularity of shoe categories.')
        sales_st = df.groupby(col('shoetype'))[col('total_value')].sum().reset_index().sort_values(col('total_value'), ascending=False)
        vbar(sales_st, x=col('shoetype'), y=col('total_value'))

    # Chart 13: Sales by Size
    if col('size'):
        explain('**Chart 13**: Which sizes contribute most to revenue?')
        sales_size = df.groupby(col('size'))[col('total_value')].sum().reset_index()
        vbar(sales_size, x=col('size'), y=col('total_value'))

    # Chart 14: Stacked bar – Color by Channel
    if col('color') and col('channel'):
        explain('**Chart 14**: Colour preferences segmented by sales channel.')
        pivot = df.pivot_table(index=col('color'), columns=col('channel'), values=col('total_value'), aggfunc='sum').fillna(0)
        fig = go.Figure()
        for ch in pivot.columns:
            fig.add_bar(name=ch, x=pivot.index, y=pivot[ch])
        fig.update_layout(barmode='stack', height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 15: Treemap of Sales by ShoeType & Color
    if col('shoetype') and col('color'):
        explain('**Chart 15**: Hierarchical view of sales by shoe type and colour.')
        treemap_df = df.groupby([col('shoetype'), col('color')])[col('total_value')].sum().reset_index()
        fig = px.treemap(treemap_df, path=[col('shoetype'), col('color')], values=col('total_value'), height=430)
        st.plotly_chart(fig, use_container_width=True)

# ------- 4. PAYMENT & CHANNEL (3 charts) -------
with tab_payment_channel:
    # Chart 16: Sales by Payment Method
    if col('paymentmethod'):
        explain('**Chart 16**: Revenue split by payment methods.')
        sales_pm = df.groupby(col('paymentmethod'))[col('total_value')].sum().reset_index()
        vbar(sales_pm, x=col('paymentmethod'), y=col('total_value'))

    # Chart 17: Heatmap – Payment vs Channel
    if col('paymentmethod') and col('channel'):
        explain('**Chart 17**: Which payment methods are preferred on each channel?')
        pivot = df.pivot_table(index=col('paymentmethod'), columns=col('channel'), values=col('total_value'), aggfunc='sum')
        fig = px.imshow(pivot, text_auto=True, aspect='auto', height=430)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 18: Average Ticket Size by Channel & Payment
    if col('paymentmethod') and col('channel'):
        explain('**Chart 18**: Average order value by channel‑payment combination.')
        aov = df.groupby([col('channel'), col('paymentmethod')])[col('total_value')].mean().reset_index()
        fig = px.sunburst(aov, path=[col('channel'), col('paymentmethod')], values=col('total_value'), color=col('total_value'), height=430)
        st.plotly_chart(fig, use_container_width=True)

# ------- 5. CORRELATION (2 charts) -------
with tab_corr:
    # Chart 19: Correlation Matrix
    explain('**Chart 19**: Pearson correlation among numeric variables.')
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', height=500, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    # Chart 20: Scatter Matrix
    if num_df.shape[1] >= 2:
        explain('**Chart 20**: Scatter‑plot matrix for pairwise relationships.')
        fig = px.scatter_matrix(num_df, dimensions=num_df.columns, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ------- 6. RAW DATA -------
with tab_data:
    explain('Interactive data table – apply any combination of filters from the sidebar to refine.')
    st.dataframe(df, use_container_width=True)
