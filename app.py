"""
True Margin Calculator v1.0.0
Calculate true product margins by incorporating vendor promo credits

Merges Blaze POS sales data with Haven Promo Performance vendor credit data
to show the actual margin after accounting for vendor-paid promotions.

CHANGELOG:
v1.0.0 (2026-01-03)
- Initial release
- Fuzzy matching within transactions for credit allocation
- Profile Template (SKU Type) matching from Product Catalog
- Private Label brand identification
- Aggregation levels: Network, Shop, Brand, Category, SKU Type, Product
- Shop filtering with multi-select support
- Export functionality for all views

Author: Haven Cannabis Data Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="True Margin Calculator v1.0.0",
    page_icon="💰",
    layout="wide"
)

VERSION = "1.0.0"

# Shop name mapping
SHOP_NAME_MAPPING = {
    'HAVEN - Maywood': 'Maywood',
    'HAVEN - LB#1 - Los Alamitos': 'Los Alamitos', 
    'HAVEN - LB#2 - Paramount': 'Paramount',
    'HAVEN - LB#3 - Downtown LB': 'DTLB',
    'HAVEN - LB#4 - Belmont': 'Belmont',
    'HAVEN - San Bernardino': 'San Bernardino',
    'HAVEN - Porterville': 'Porterville',
    'HAVEN - Lakewood': 'Lakewood',
    'HAVEN - Orange County': 'Stanton',
    'HAVEN - Fresno': 'Fresno',
    'HAVEN - Corona': 'Corona',
    'HAVEN - Hawthorne South Bay': 'Hawthorne'
}

# Default Private Label brands (can be overridden by upload)
DEFAULT_PRIVATE_LABELS = [
    'Astral Farms', 'Besos', 'Black Label', 'Black Label Platinum', 
    'Block Party', 'Block Party Exotics', 'Bud Society', 'Coastal Cowboys',
    'Crave', 'Daily Dose', 'Dope St.', 'Dope St. Exotics', 'Dunzo', 
    'Fat Stash', 'High Five', 'Honey Habit', "Lil' Buzzies", 'Lokaal',
    'Made from Dirt', 'MikroDose', 'Nuggies', 'Outrun', 'Pretty Dope',
    'PTO', 'Roll & Ready', 'Side Hustle'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fuzzy_match_within_transaction(trans_no, credit_df, sales_df):
    """
    Match credit products to sales products within a transaction using
    process of elimination - each credit product matches to closest available.
    """
    credit_trans = credit_df[credit_df['Trans No'] == trans_no]
    sales_trans = sales_df[sales_df['Trans No'] == trans_no]
    
    credit_products = credit_trans['Product'].tolist()
    sales_products = sales_trans['Product'].unique().tolist()
    
    matches = {}
    used_sales = set()
    
    # Sort by length for better matching (longer/more specific first)
    credit_products_sorted = sorted(credit_products, key=len, reverse=True)
    
    for credit_prod in credit_products_sorted:
        best_match = None
        best_score = 0
        
        for sales_prod in sales_products:
            if sales_prod in used_sales:
                continue
            
            score = SequenceMatcher(None, credit_prod.lower(), sales_prod.lower()).ratio()
            
            if score > best_score:
                best_score = score
                best_match = sales_prod
        
        if best_match:
            used_sales.add(best_match)
            matches[credit_prod] = {'sales_product': best_match, 'score': best_score}
    
    return matches


def match_to_profile_template(product_name, brand, catalog_df):
    """Match a product to its Profile Template (SKU Type) from catalog"""
    if pd.isna(product_name) or pd.isna(brand):
        return None
    
    brand_templates = catalog_df[catalog_df['Brand'] == brand]['Profile Template'].unique()
    
    if len(brand_templates) == 0:
        return None
    
    # Try exact match first
    for template in brand_templates:
        if str(product_name).lower() == str(template).lower():
            return template
    
    # Try STRAIN/COLOR/FLAVOR placeholder matching
    product_upper = str(product_name).upper()
    for template in brand_templates:
        template_upper = str(template).upper()
        for placeholder in ['STRAIN', 'COLOR', 'FLAVOR']:
            if placeholder in template_upper:
                parts = template_upper.split(placeholder)
                if len(parts) == 2:
                    prefix, suffix = parts
                    if product_upper.startswith(prefix) and product_upper.endswith(suffix):
                        return template
    
    # If single template for brand, use it
    if len(brand_templates) == 1:
        return brand_templates[0]
    
    return None


def format_currency(value):
    """Format value as currency"""
    if pd.isna(value):
        return "—"
    return f"${value:,.2f}"


def format_percentage(value):
    """Format value as percentage"""
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"


def calculate_margin_pct(margin, net_sales):
    """Calculate margin percentage safely"""
    if net_sales == 0 or pd.isna(net_sales) or pd.isna(margin):
        return 0
    return (margin / net_sales) * 100


# ============================================================================
# DATA PROCESSING
# ============================================================================

@st.cache_data
def process_data(sales_df, credit_df, catalog_df, private_labels):
    """
    Process sales and credit data to calculate true margins.
    
    Returns merged DataFrame with all margin calculations.
    """
    # Step 1: Fuzzy match credits to sales within transactions
    all_credit_matches = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    unique_trans = credit_df['Trans No'].unique()
    
    for i, trans_no in enumerate(unique_trans):
        if i % 100 == 0:
            progress_bar.progress(i / len(unique_trans))
            progress_text.text(f"Matching credits to sales... {i}/{len(unique_trans)} transactions")
        
        matches = fuzzy_match_within_transaction(trans_no, credit_df, sales_df)
        
        credit_trans = credit_df[credit_df['Trans No'] == trans_no]
        
        for _, row in credit_trans.iterrows():
            credit_prod = row['Product']
            match_info = matches.get(credit_prod, {'sales_product': None, 'score': 0})
            
            all_credit_matches.append({
                'Trans_No': trans_no,
                'Credit_Product': credit_prod,
                'Sales_Product': match_info['sales_product'],
                'Match_Score': match_info['score'],
                'Vendor_Pays': row['Vendor Pays'],
                'Haven_Pays': row['Haven Pays'],
                'Promo_Credit_New': row['Promo Credit New'],
                'Reporting_Promo': row['Reporting Promo'],
                'Promo_Type': row['Promo Type']
            })
    
    progress_bar.progress(1.0)
    progress_text.text("Credit matching complete!")
    
    credit_matched_df = pd.DataFrame(all_credit_matches)
    
    # Step 2: Aggregate sales by Trans No + Product
    sales_agg = sales_df.groupby(['Trans No', 'Product']).agg({
        'Brand': 'first',
        'Product Category': 'first',
        'Quantity Sold': 'sum',
        'Gross Sales': 'sum',
        'Net Sales': 'sum',
        'COGS': 'sum',
        'Product Discounts': 'sum',
        'Cart Discounts': 'sum',
        'Unit Cost': 'first',
        'Product ID': 'first',
        'Shop': 'first',
        'Date': 'first'
    }).reset_index()
    
    # Step 3: Merge with credit data (high/medium confidence only)
    good_matches = credit_matched_df[credit_matched_df['Match_Score'] >= 0.60]
    
    merged = pd.merge(
        sales_agg,
        good_matches[['Trans_No', 'Sales_Product', 'Vendor_Pays', 'Haven_Pays', 
                      'Promo_Credit_New', 'Reporting_Promo', 'Promo_Type', 'Match_Score']],
        left_on=['Trans No', 'Product'],
        right_on=['Trans_No', 'Sales_Product'],
        how='left'
    )
    
    # Step 4: Add Profile Template (SKU Type) from catalog
    if catalog_df is not None and len(catalog_df) > 0:
        progress_text.text("Matching to Profile Templates...")
        merged['Profile_Template'] = merged.apply(
            lambda row: match_to_profile_template(row['Product'], row['Brand'], catalog_df), axis=1
        )
    else:
        merged['Profile_Template'] = None
    
    # Step 5: Add Private Label flag
    private_labels_lower = [b.strip().lower() for b in private_labels]
    merged['Brand_Lower'] = merged['Brand'].str.lower().str.strip()
    merged['Is_Private_Label'] = merged['Brand_Lower'].isin(private_labels_lower)
    
    # Step 6: Calculate True Margin
    merged['Vendor_Pays'] = merged['Vendor_Pays'].fillna(0)
    merged['Haven_Pays'] = merged['Haven_Pays'].fillna(0)
    
    merged['True_COGS'] = merged['COGS'] - merged['Vendor_Pays'] + merged['Haven_Pays']
    merged['Standard_Margin'] = merged['Net Sales'] - merged['COGS']
    merged['True_Margin'] = merged['Net Sales'] - merged['True_COGS']
    merged['Margin_Lift'] = merged['Vendor_Pays'] - merged['Haven_Pays']
    
    progress_text.empty()
    progress_bar.empty()
    
    return merged, credit_matched_df


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_network_overview(df):
    """Display network-level metrics"""
    st.subheader("📊 Network Overview")
    
    total_net_sales = df['Net Sales'].sum()
    total_std_margin = df['Standard_Margin'].sum()
    total_true_margin = df['True_Margin'].sum()
    total_margin_lift = df['Margin_Lift'].sum()
    total_vendor_pays = df['Vendor_Pays'].sum()
    total_haven_pays = df['Haven_Pays'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Net Sales", format_currency(total_net_sales))
    with col2:
        std_pct = calculate_margin_pct(total_std_margin, total_net_sales)
        st.metric("Standard Margin", format_currency(total_std_margin), f"{std_pct:.1f}%")
    with col3:
        true_pct = calculate_margin_pct(total_true_margin, total_net_sales)
        delta = true_pct - std_pct
        st.metric("True Margin", format_currency(total_true_margin), f"{true_pct:.1f}% ({delta:+.1f}%)")
    with col4:
        st.metric("Margin Lift", format_currency(total_margin_lift), 
                  help="Vendor Pays - Haven Pays")
    
    # Credit breakdown
    st.write("**Promo Credit Breakdown:**")
    credit_col1, credit_col2, credit_col3 = st.columns(3)
    with credit_col1:
        st.metric("Vendor Pays (Credit)", format_currency(total_vendor_pays),
                  help="Credits received from vendors - reduces effective COGS")
    with credit_col2:
        st.metric("Haven Pays (Cost)", format_currency(total_haven_pays),
                  help="Discount cost Haven absorbs - increases effective COGS")
    with credit_col3:
        net_credit = total_vendor_pays - total_haven_pays
        st.metric("Net Credit Impact", format_currency(net_credit),
                  help="Net benefit from promo programs")


def display_private_label_comparison(df):
    """Display Private Label vs 3rd Party comparison"""
    st.subheader("🏷️ Private Label vs 3rd Party")
    
    summary = df.groupby('Is_Private_Label').agg({
        'Net Sales': 'sum',
        'COGS': 'sum',
        'True_COGS': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum'
    }).reset_index()
    
    summary['Label'] = summary['Is_Private_Label'].apply(lambda x: '🏠 Private Label' if x else '📦 3rd Party')
    summary['Std_Margin_%'] = (summary['Standard_Margin'] / summary['Net Sales'] * 100).round(1)
    summary['True_Margin_%'] = (summary['True_Margin'] / summary['Net Sales'] * 100).round(1)
    summary['Margin_Change'] = summary['True_Margin_%'] - summary['Std_Margin_%']
    
    col1, col2 = st.columns(2)
    
    for i, row in summary.iterrows():
        col = col1 if row['Is_Private_Label'] else col2
        
        with col:
            st.write(f"### {row['Label']}")
            st.metric("Net Sales", format_currency(row['Net Sales']))
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Standard Margin", f"{row['Std_Margin_%']:.1f}%")
            with m2:
                st.metric("True Margin", f"{row['True_Margin_%']:.1f}%", 
                          f"{row['Margin_Change']:+.1f}%")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Vendor Pays", format_currency(row['Vendor_Pays']))
            with c2:
                st.metric("Haven Pays", format_currency(row['Haven_Pays']))


def display_brand_analysis(df):
    """Display brand-level analysis"""
    st.subheader("🏢 Brand Performance")
    
    brand_summary = df.groupby('Brand').agg({
        'Net Sales': 'sum',
        'COGS': 'sum',
        'True_COGS': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Is_Private_Label': 'first',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    brand_summary['Std_Margin_%'] = (brand_summary['Standard_Margin'] / brand_summary['Net Sales'] * 100).round(1)
    brand_summary['True_Margin_%'] = (brand_summary['True_Margin'] / brand_summary['Net Sales'] * 100).round(1)
    brand_summary['Margin_Change'] = (brand_summary['True_Margin_%'] - brand_summary['Std_Margin_%']).round(1)
    brand_summary['Type'] = brand_summary['Is_Private_Label'].apply(lambda x: '🏠 Private' if x else '📦 3rd Party')
    
    # Sort options
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox("Sort by:", 
                               ['Net Sales', 'True_Margin_%', 'Margin_Lift', 'Margin_Change'],
                               index=0)
    
    brand_summary = brand_summary.sort_values(sort_by, ascending=False)
    
    # Display columns
    display_cols = ['Type', 'Brand', 'Net Sales', 'Std_Margin_%', 'True_Margin_%', 
                    'Margin_Change', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    # Format for display
    display_df = brand_summary[display_cols].copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Vendor_Pays'] = display_df['Vendor_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Haven_Pays'] = display_df['Haven_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Margin_Lift'] = display_df['Margin_Lift'].apply(lambda x: f"${x:+,.0f}")
    display_df['Std_Margin_%'] = display_df['Std_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['True_Margin_%'] = display_df['True_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['Margin_Change'] = display_df['Margin_Change'].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    brand_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Brand Data", csv_buffer.getvalue(), 
                       "brand_true_margins.csv", "text/csv")


def display_category_analysis(df):
    """Display category-level analysis"""
    st.subheader("📂 Category Performance")
    
    cat_summary = df.groupby('Product Category').agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    cat_summary['Std_Margin_%'] = (cat_summary['Standard_Margin'] / cat_summary['Net Sales'] * 100).round(1)
    cat_summary['True_Margin_%'] = (cat_summary['True_Margin'] / cat_summary['Net Sales'] * 100).round(1)
    cat_summary['Margin_Change'] = (cat_summary['True_Margin_%'] - cat_summary['Std_Margin_%']).round(1)
    
    cat_summary = cat_summary.sort_values('Net Sales', ascending=False)
    
    # Format for display
    display_df = cat_summary.copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Vendor_Pays'] = display_df['Vendor_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Haven_Pays'] = display_df['Haven_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Margin_Lift'] = display_df['Margin_Lift'].apply(lambda x: f"${x:+,.0f}")
    display_df['Std_Margin_%'] = display_df['Std_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['True_Margin_%'] = display_df['True_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['Margin_Change'] = display_df['Margin_Change'].apply(lambda x: f"{x:+.1f}%")
    
    display_cols = ['Product Category', 'Net Sales', 'Std_Margin_%', 'True_Margin_%',
                    'Margin_Change', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    cat_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Category Data", csv_buffer.getvalue(),
                       "category_true_margins.csv", "text/csv")


def display_sku_type_analysis(df):
    """Display SKU Type (Profile Template) analysis"""
    st.subheader("📦 SKU Type Performance (Profile Template)")
    
    # Filter to products with Profile Template
    df_with_template = df[df['Profile_Template'].notna()].copy()
    
    if len(df_with_template) == 0:
        st.warning("No products matched to Profile Templates. Upload a Product Catalog to enable SKU Type analysis.")
        return
    
    sku_summary = df_with_template.groupby(['Profile_Template', 'Brand']).agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum',
        'Is_Private_Label': 'first'
    }).reset_index()
    
    sku_summary['Std_Margin_%'] = (sku_summary['Standard_Margin'] / sku_summary['Net Sales'] * 100).round(1)
    sku_summary['True_Margin_%'] = (sku_summary['True_Margin'] / sku_summary['Net Sales'] * 100).round(1)
    sku_summary['Margin_Change'] = (sku_summary['True_Margin_%'] - sku_summary['Std_Margin_%']).round(1)
    sku_summary['Type'] = sku_summary['Is_Private_Label'].apply(lambda x: '🏠' if x else '📦')
    
    # Sort and filter options
    col1, col2 = st.columns([1, 2])
    with col1:
        sort_by = st.selectbox("Sort by:", 
                               ['Net Sales', 'True_Margin_%', 'Margin_Lift'],
                               key='sku_sort')
    with col2:
        min_sales = st.slider("Minimum Net Sales", 0, 5000, 500, step=100)
    
    sku_summary = sku_summary[sku_summary['Net Sales'] >= min_sales]
    sku_summary = sku_summary.sort_values(sort_by, ascending=False)
    
    # Format for display
    display_df = sku_summary.copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Margin_Lift'] = display_df['Margin_Lift'].apply(lambda x: f"${x:+,.0f}")
    display_df['Std_Margin_%'] = display_df['Std_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['True_Margin_%'] = display_df['True_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['Margin_Change'] = display_df['Margin_Change'].apply(lambda x: f"{x:+.1f}%")
    
    display_cols = ['Type', 'Brand', 'Profile_Template', 'Net Sales', 
                    'Std_Margin_%', 'True_Margin_%', 'Margin_Change', 'Margin_Lift']
    
    st.dataframe(display_df[display_cols].head(50), use_container_width=True, hide_index=True)
    
    st.info(f"Showing {min(50, len(display_df))} of {len(sku_summary)} SKU Types with ≥${min_sales} Net Sales")
    
    # Export
    csv_buffer = io.StringIO()
    sku_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download SKU Type Data", csv_buffer.getvalue(),
                       "sku_type_true_margins.csv", "text/csv")


def display_shop_analysis(df):
    """Display shop-level analysis"""
    st.subheader("🏪 Shop Performance")
    
    shop_summary = df.groupby('Shop').agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    shop_summary['Std_Margin_%'] = (shop_summary['Standard_Margin'] / shop_summary['Net Sales'] * 100).round(1)
    shop_summary['True_Margin_%'] = (shop_summary['True_Margin'] / shop_summary['Net Sales'] * 100).round(1)
    shop_summary['Margin_Change'] = (shop_summary['True_Margin_%'] - shop_summary['Std_Margin_%']).round(1)
    shop_summary['Shop_Short'] = shop_summary['Shop'].str.replace('HAVEN - ', '')
    
    shop_summary = shop_summary.sort_values('Net Sales', ascending=False)
    
    # Format for display
    display_df = shop_summary.copy()
    display_df['Net Sales'] = display_df['Net Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Vendor_Pays'] = display_df['Vendor_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Haven_Pays'] = display_df['Haven_Pays'].apply(lambda x: f"${x:,.0f}")
    display_df['Margin_Lift'] = display_df['Margin_Lift'].apply(lambda x: f"${x:+,.0f}")
    display_df['Std_Margin_%'] = display_df['Std_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['True_Margin_%'] = display_df['True_Margin_%'].apply(lambda x: f"{x:.1f}%")
    display_df['Margin_Change'] = display_df['Margin_Change'].apply(lambda x: f"{x:+.1f}%")
    
    display_cols = ['Shop_Short', 'Net Sales', 'Std_Margin_%', 'True_Margin_%',
                    'Margin_Change', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    shop_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Shop Data", csv_buffer.getvalue(),
                       "shop_true_margins.csv", "text/csv")


def display_product_detail(df):
    """Display product-level detail"""
    st.subheader("🔍 Product Detail")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brands = ['All'] + sorted(df['Brand'].dropna().unique().tolist())
        selected_brand = st.selectbox("Filter by Brand:", brands)
    
    with col2:
        categories = ['All'] + sorted(df['Product Category'].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by Category:", categories)
    
    with col3:
        show_promo_only = st.checkbox("Show only products with promo credits", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_brand != 'All':
        filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Product Category'] == selected_category]
    
    if show_promo_only:
        filtered_df = filtered_df[(filtered_df['Vendor_Pays'] > 0) | (filtered_df['Haven_Pays'] > 0)]
    
    # Aggregate by Product
    product_summary = filtered_df.groupby(['Product', 'Brand', 'Product Category']).agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum',
        'Profile_Template': 'first'
    }).reset_index()
    
    product_summary['Std_Margin_%'] = (product_summary['Standard_Margin'] / product_summary['Net Sales'] * 100).round(1)
    product_summary['True_Margin_%'] = (product_summary['True_Margin'] / product_summary['Net Sales'] * 100).round(1)
    
    product_summary = product_summary.sort_values('Net Sales', ascending=False)
    
    st.info(f"Showing {len(product_summary):,} products")
    
    # Display
    display_cols = ['Brand', 'Product', 'Net Sales', 'Quantity Sold', 
                    'Std_Margin_%', 'True_Margin_%', 'Vendor_Pays', 'Haven_Pays']
    
    st.dataframe(product_summary[display_cols].head(100), use_container_width=True, hide_index=True)
    
    # Export full data
    csv_buffer = io.StringIO()
    product_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Product Data", csv_buffer.getvalue(),
                       "product_true_margins.csv", "text/csv")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title(f"💰 True Margin Calculator v{VERSION}")
    st.markdown("Calculate true product margins by incorporating vendor promo credits")
    
    # Sidebar - Data Upload
    st.sidebar.header("📁 Data Sources")
    
    st.sidebar.subheader("Required Files")
    
    sales_file = st.sidebar.file_uploader(
        "📊 Sales Report (Blaze POS)",
        type=['csv'],
        help="Total Sales Products export from Blaze POS"
    )
    
    credit_file = st.sidebar.file_uploader(
        "💳 Promo Credit Report",
        type=['csv'],
        help="Promo Credit by Brand/Vendor report from Haven Reporting"
    )
    
    st.sidebar.subheader("Optional Files")
    
    catalog_file = st.sidebar.file_uploader(
        "📋 Product Catalog",
        type=['csv'],
        help="Haven Product Catalog for SKU Type matching (Profile Template)"
    )
    
    private_label_file = st.sidebar.file_uploader(
        "🏠 Private Label Brands",
        type=['csv'],
        help="CSV with 'Name' column listing private label brands"
    )
    
    # Shop Filter
    st.sidebar.subheader("🏪 Shop Filter")
    
    # Changelog
    with st.sidebar.expander("📋 Version History"):
        st.markdown("""
        **v1.0.0** (2026-01-03)
        - 🚀 Initial release
        - 🔗 Fuzzy matching within transactions
        - 📦 Profile Template (SKU Type) matching
        - 🏠 Private Label identification
        - 📊 Multi-level aggregation views
        - 🏪 Shop filtering
        - 📥 Export functionality
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version {VERSION}**")
    
    # Main content
    if sales_file is None or credit_file is None:
        st.info("👆 Upload Sales Report and Promo Credit Report to get started")
        
        st.subheader("📚 How It Works")
        
        st.markdown("""
        ### True Margin Formula
        
        ```
        True COGS = Standard COGS - Vendor Pays + Haven Pays
        True Margin = Net Sales - True COGS
        Margin Lift = Vendor Pays - Haven Pays
        ```
        
        ### Key Concepts
        
        **Vendor Pays** 💚 - Credit received from vendors that reduces your effective COGS.
        When vendors subsidize promotions, this is money back in your pocket.
        
        **Haven Pays** 🔴 - Discount cost Haven absorbs that increases your effective COGS.
        For private label brands or promos without vendor support, Haven covers the discount.
        
        **Margin Lift** 📈 - The net impact of promo programs on your margins.
        Positive = Vendor credits exceed Haven costs. Negative = Haven is absorbing more than receiving.
        
        ### Aggregation Levels
        
        1. **Network** - Total performance across all shops
        2. **Shop** - Individual location performance (with multi-select filter)
        3. **Brand** - Brand-level performance with Private Label flag
        4. **Category** - Product category performance
        5. **SKU Type** - Profile Template level (requires Product Catalog)
        6. **Product** - Individual product detail
        """)
        return
    
    # Load data
    try:
        sales_df = pd.read_csv(sales_file, low_memory=False)
        credit_df = pd.read_csv(credit_file)
        
        # Load optional files
        catalog_df = None
        if catalog_file is not None:
            catalog_df = pd.read_csv(catalog_file)
        
        private_labels = DEFAULT_PRIVATE_LABELS
        if private_label_file is not None:
            pl_df = pd.read_csv(private_label_file)
            if 'Name' in pl_df.columns:
                private_labels = pl_df['Name'].tolist()
        
        st.success(f"✅ Loaded: {len(sales_df):,} sales rows, {len(credit_df):,} credit rows")
        
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return
    
    # Process data
    if 'processed_data' not in st.session_state or st.sidebar.button("🔄 Reprocess Data"):
        with st.spinner("Processing data..."):
            merged_df, credit_matched_df = process_data(sales_df, credit_df, catalog_df, private_labels)
            st.session_state['processed_data'] = merged_df
            st.session_state['credit_matched'] = credit_matched_df
    
    merged_df = st.session_state['processed_data']
    
    # Shop filter in sidebar
    available_shops = sorted(merged_df['Shop'].dropna().unique().tolist())
    selected_shops = st.sidebar.multiselect(
        "Select Shops:",
        options=available_shops,
        default=available_shops,
        help="Filter data to specific shops"
    )
    
    if selected_shops:
        filtered_df = merged_df[merged_df['Shop'].isin(selected_shops)]
    else:
        filtered_df = merged_df
    
    st.sidebar.info(f"📊 {len(filtered_df):,} products in view")
    
    # Validation metrics
    vendor_captured = merged_df['Vendor_Pays'].sum()
    haven_captured = merged_df['Haven_Pays'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Products Analyzed", f"{len(filtered_df):,}")
    with col2:
        st.metric("Vendor Credits Captured", format_currency(vendor_captured))
    with col3:
        st.metric("Haven Costs Captured", format_currency(haven_captured))
    
    # Tabs for different views
    tab_names = ["📊 Overview", "🏷️ Private Label", "🏢 Brands", 
                 "📂 Categories", "📦 SKU Types", "🏪 Shops", "🔍 Products"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        display_network_overview(filtered_df)
    
    with tabs[1]:
        display_private_label_comparison(filtered_df)
    
    with tabs[2]:
        display_brand_analysis(filtered_df)
    
    with tabs[3]:
        display_category_analysis(filtered_df)
    
    with tabs[4]:
        display_sku_type_analysis(filtered_df)
    
    with tabs[5]:
        display_shop_analysis(filtered_df)
    
    with tabs[6]:
        display_product_detail(filtered_df)
    
    # Full data export
    st.sidebar.subheader("📥 Export")
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.sidebar.download_button(
        "📥 Download Full Dataset",
        csv_buffer.getvalue(),
        "true_margin_full_data.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()