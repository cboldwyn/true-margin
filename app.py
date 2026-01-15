"""
True Margin Calculator v1.1.0
Calculate true product margins by incorporating vendor promo credits

Merges Blaze POS sales data with Haven Promo Performance vendor credit data
to show the actual margin after accounting for vendor-paid promotions.

CHANGELOG:
v1.1.0 (2026-01-14)
- ENHANCEMENT: Multi-file upload for Sales Detail reports
- ENHANCEMENT: Auto-processing on file upload (no Load Data button)
- ENHANCEMENT: Chunked file reading for large CSVs (prevents timeouts)
- ENHANCEMENT: Progress indicators throughout processing
- ENHANCEMENT: Session state optimization for faster re-renders

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
    page_title="True Margin Calculator v1.1.0",
    page_icon="💰",
    layout="wide"
)

VERSION = "1.1.0"

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

def format_currency(value):
    """Format number as currency"""
    if pd.isna(value):
        return "$0.00"
    return f"${value:,.2f}"

def format_percent(value):
    """Format number as percentage"""
    if pd.isna(value):
        return "0.0%"
    return f"{value:.1f}%"

def clean_price(price_str):
    """Clean and convert price string to float"""
    if pd.isna(price_str) or price_str == '':
        return 0.0
    try:
        cleaned = str(price_str).replace('$', '').replace(',', '').strip()
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0

def similarity_score(a, b):
    """Calculate fuzzy similarity between two strings"""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# ============================================================================
# FILE LOADING FUNCTIONS (OPTIMIZED FOR LARGE FILES)
# ============================================================================

def load_multiple_sales_files(uploaded_files, progress_callback=None):
    """
    Load and combine multiple Total Sales Detail CSV files
    
    Uses chunked reading for large files to prevent memory issues
    and provides progress updates via callback
    
    Args:
        uploaded_files: List of uploaded file objects
        progress_callback: Optional function(progress, status_text) for updates
        
    Returns:
        Combined DataFrame or None if error
    """
    if not uploaded_files:
        return None
    
    all_dfs = []
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        try:
            if progress_callback:
                progress_callback((i / total_files) * 0.5, f"Loading file {i+1}/{total_files}: {file.name}")
            
            # Get file size to decide on chunking strategy
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            # For files > 50MB, use chunked reading
            if file_size > 50 * 1024 * 1024:
                chunks = []
                chunk_size = 50000  # rows per chunk
                
                for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file, low_memory=False)
            
            # Add source file identifier
            df['_Source_File'] = file.name
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
            continue
    
    if not all_dfs:
        return None
    
    if progress_callback:
        progress_callback(0.6, "Combining files...")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if progress_callback:
        progress_callback(0.7, "Removing duplicates...")
    
    # Remove exact duplicates (same transaction might appear in overlapping exports)
    # Use Trans No + Product + Shop as key (EXACT Blaze column names)
    key_cols = ['Trans No', 'Product', 'Shop']
    existing_key_cols = [col for col in key_cols if col in combined_df.columns]
    
    if existing_key_cols:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=existing_key_cols, keep='first')
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
        
        if duplicates_removed > 0 and progress_callback:
            progress_callback(0.75, f"Removed {duplicates_removed:,} duplicate rows")
    
    if progress_callback:
        progress_callback(0.8, "Finalizing...")
    
    return combined_df

def load_credit_file(uploaded_file, progress_callback=None):
    """
    Load Promo Credit Report CSV
    
    Args:
        uploaded_file: Uploaded file object
        progress_callback: Optional function(progress, status_text) for updates
        
    Returns:
        DataFrame or None if error
    """
    if uploaded_file is None:
        return None
    
    try:
        if progress_callback:
            progress_callback(0.85, f"Loading credit report: {uploaded_file.name}")
        
        df = pd.read_csv(uploaded_file, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading credit file: {str(e)}")
        return None

# ============================================================================
# MATCHING FUNCTIONS
# ============================================================================

def match_credits_to_sales(sales_df, credit_df, progress_callback=None):
    """
    Match credit rows to sales rows using transaction-level fuzzy matching
    
    Strategy:
    1. Group by Trans No
    2. Within each transaction, match credit products to sales products
    3. Use process of elimination - each credit product matches to closest available
    
    Args:
        sales_df: Sales DataFrame with Trans No, Product, etc. (Blaze export)
        credit_df: Credit DataFrame with Trans No, Product, Vendor Pays, etc.
        progress_callback: Optional function(progress, status_text) for updates
        
    Returns:
        Sales DataFrame with credit columns merged
    """
    if progress_callback:
        progress_callback(0.0, "Preparing credit matching...")
    
    # Initialize credit columns in sales
    sales_df = sales_df.copy()
    sales_df['Vendor_Pays'] = 0.0
    sales_df['Haven_Pays'] = 0.0
    sales_df['Credit_Match_Score'] = 0.0
    sales_df['Credit_Product_Matched'] = None
    
    # Standardize transaction column names
    # Blaze Total Sales Detail uses 'Trans No'
    # Credit report uses 'Trans No' (matching)
    trans_col_sales = 'Trans No' if 'Trans No' in sales_df.columns else None
    trans_col_credit = 'Trans No' if 'Trans No' in credit_df.columns else ('Transaction ID' if 'Transaction ID' in credit_df.columns else None)
    
    if not trans_col_sales or not trans_col_credit:
        st.warning("Could not find transaction columns for matching")
        return sales_df
    
    # Get unique transactions from credits
    credit_transactions = credit_df[trans_col_credit].dropna().unique()
    total_trans = len(credit_transactions)
    
    if progress_callback:
        progress_callback(0.05, f"Matching {total_trans:,} transactions with credits...")
    
    # Build credit lookup by transaction
    credit_by_trans = credit_df.groupby(trans_col_credit)
    
    matched_count = 0
    
    for i, trans_no in enumerate(credit_transactions):
        if i % 500 == 0 and progress_callback:
            progress = 0.05 + (i / total_trans) * 0.9
            progress_callback(progress, f"Matching transaction {i+1:,}/{total_trans:,}")
        
        # Get credits for this transaction
        trans_credits = credit_by_trans.get_group(trans_no).copy()
        
        # Get sales for this transaction
        sales_mask = sales_df[trans_col_sales] == trans_no
        if not sales_mask.any():
            continue
        
        trans_sales_indices = sales_df[sales_mask].index.tolist()
        
        # Match each credit product to a sales product
        available_sales_indices = trans_sales_indices.copy()
        
        for _, credit_row in trans_credits.iterrows():
            if not available_sales_indices:
                break
            
            credit_product = str(credit_row.get('Product', ''))
            vendor_pays = clean_price(credit_row.get('Vendor Pays', 0))
            haven_pays = clean_price(credit_row.get('Haven Pays', 0))
            
            # Find best match among available sales products
            best_idx = None
            best_score = 0.0
            
            for sales_idx in available_sales_indices:
                sales_product = str(sales_df.at[sales_idx, 'Product'])
                score = similarity_score(credit_product, sales_product)
                
                if score > best_score:
                    best_score = score
                    best_idx = sales_idx
            
            # Apply match if good enough (threshold 0.5)
            if best_idx is not None and best_score >= 0.5:
                sales_df.at[best_idx, 'Vendor_Pays'] = vendor_pays
                sales_df.at[best_idx, 'Haven_Pays'] = haven_pays
                sales_df.at[best_idx, 'Credit_Match_Score'] = best_score
                sales_df.at[best_idx, 'Credit_Product_Matched'] = credit_product
                available_sales_indices.remove(best_idx)
                matched_count += 1
    
    if progress_callback:
        progress_callback(0.95, f"Matched {matched_count:,} credit rows to sales")
    
    return sales_df

def add_profile_template_matching(sales_df, catalog_df, progress_callback=None):
    """
    Add Profile Template (SKU Type) from Product Catalog
    
    Matches on Company Product ID (Product ID)
    """
    if catalog_df is None or catalog_df.empty:
        sales_df['Profile_Template'] = None
        return sales_df
    
    if progress_callback:
        progress_callback(0.0, "Matching Profile Templates...")
    
    # Try to match on Product ID -> Company Product ID
    if 'Product ID' in sales_df.columns and 'Company Product ID' in catalog_df.columns:
        # Build lookup
        template_lookup = {}
        for _, row in catalog_df.iterrows():
            prod_id = row.get('Company Product ID')
            template = row.get('Profile Template')
            if pd.notna(prod_id) and pd.notna(template):
                template_lookup[str(prod_id)] = template
        
        # Apply lookup
        sales_df['Profile_Template'] = sales_df['Product ID'].apply(
            lambda x: template_lookup.get(str(x)) if pd.notna(x) else None
        )
        
        matched = sales_df['Profile_Template'].notna().sum()
        total = len(sales_df)
        
        if progress_callback:
            progress_callback(1.0, f"Matched {matched:,}/{total:,} products to Profile Templates")
    else:
        sales_df['Profile_Template'] = None
    
    return sales_df

# ============================================================================
# MARGIN CALCULATION FUNCTIONS
# ============================================================================

def calculate_margins(df, private_labels):
    """
    Calculate standard and true margins
    
    Formulas:
    - Standard COGS = Unit Cost * Quantity Sold
    - Standard Margin = Net Sales - Standard COGS
    - True COGS = Standard COGS - Vendor Pays + Haven Pays
    - True Margin = Net Sales - True COGS
    - Margin Lift = Vendor Pays - Haven Pays
    """
    df = df.copy()
    
    # Clean numeric columns
    # EXACT column names from Blaze Total Sales Detail export
    df['Net Sales'] = df['Net Sales'].apply(clean_price)
    df['Unit Cost'] = df['Unit Cost'].apply(clean_price)
    df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce').fillna(0)
    
    # Calculate Standard COGS and Margin
    df['Standard_COGS'] = df['Unit Cost'] * df['Quantity Sold']
    df['Standard_Margin'] = df['Net Sales'] - df['Standard_COGS']
    
    # Calculate True COGS and Margin
    df['True_COGS'] = df['Standard_COGS'] - df['Vendor_Pays'] + df['Haven_Pays']
    df['True_Margin'] = df['Net Sales'] - df['True_COGS']
    
    # Margin Lift (impact of promo program)
    df['Margin_Lift'] = df['Vendor_Pays'] - df['Haven_Pays']
    
    # Private Label flag
    df['Is_Private_Label'] = df['Brand'].isin(private_labels)
    
    # Calculate margin percentages
    df['Standard_Margin_Pct'] = np.where(
        df['Net Sales'] > 0,
        (df['Standard_Margin'] / df['Net Sales']) * 100,
        0
    )
    df['True_Margin_Pct'] = np.where(
        df['Net Sales'] > 0,
        (df['True_Margin'] / df['Net Sales']) * 100,
        0
    )
    
    return df

# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def process_data(sales_df, credit_df, catalog_df, private_labels, progress_container):
    """
    Main processing pipeline
    
    Steps:
    1. Clean and validate sales data
    2. Match credits to sales
    3. Add Profile Template from catalog
    4. Calculate margins
    5. Return processed DataFrame
    """
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    def update_progress(pct, text):
        progress_bar.progress(min(pct, 1.0))
        status_text.text(text)
    
    # Step 1: Clean sales data
    update_progress(0.05, "Validating sales data...")
    
    # EXACT column names from Blaze Total Sales Detail export
    required_cols = ['Trans No', 'Product', 'Net Sales', 'Unit Cost', 'Quantity Sold']
    missing_cols = [col for col in required_cols if col not in sales_df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns in sales data: {missing_cols}")
        return None
    
    # Step 2: Match credits to sales
    update_progress(0.1, "Matching credits to sales transactions...")
    
    if credit_df is not None and not credit_df.empty:
        sales_df = match_credits_to_sales(
            sales_df, credit_df, 
            lambda p, t: update_progress(0.1 + p * 0.5, t)
        )
    else:
        sales_df['Vendor_Pays'] = 0.0
        sales_df['Haven_Pays'] = 0.0
    
    # Step 3: Add Profile Template
    update_progress(0.65, "Adding Profile Templates...")
    
    sales_df = add_profile_template_matching(
        sales_df, catalog_df,
        lambda p, t: update_progress(0.65 + p * 0.15, t)
    )
    
    # Step 4: Calculate margins
    update_progress(0.85, "Calculating margins...")
    
    sales_df = calculate_margins(sales_df, private_labels)
    
    # Step 5: Final cleanup
    update_progress(0.95, "Finalizing...")
    
    # Ensure Shop column exists
    if 'Shop' not in sales_df.columns:
        sales_df['Shop'] = 'Unknown'
    
    update_progress(1.0, "✅ Processing complete!")
    
    return sales_df

# ============================================================================
# AGGREGATION VIEWS
# ============================================================================

def render_network_view(df):
    """Network-level summary"""
    st.subheader("🌐 Network Performance")
    
    # Key metrics
    total_sales = df['Net Sales'].sum()
    std_margin = df['Standard_Margin'].sum()
    true_margin = df['True_Margin'].sum()
    vendor_pays = df['Vendor_Pays'].sum()
    haven_pays = df['Haven_Pays'].sum()
    margin_lift = df['Margin_Lift'].sum()
    
    std_margin_pct = (std_margin / total_sales * 100) if total_sales > 0 else 0
    true_margin_pct = (true_margin / total_sales * 100) if total_sales > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Net Sales", format_currency(total_sales))
    with col2:
        st.metric("Standard Margin", format_currency(std_margin), f"{std_margin_pct:.1f}%")
    with col3:
        st.metric("True Margin", format_currency(true_margin), f"{true_margin_pct:.1f}%")
    with col4:
        lift_delta = f"+{format_currency(margin_lift)}" if margin_lift >= 0 else format_currency(margin_lift)
        st.metric("Margin Lift", lift_delta)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💚 Vendor Pays", format_currency(vendor_pays), help="Credits received from vendors")
    with col2:
        st.metric("🔴 Haven Pays", format_currency(haven_pays), help="Discount cost absorbed by Haven")
    with col3:
        transactions = df['Trans No'].nunique() if 'Trans No' in df.columns else 0
        st.metric("📊 Transactions", f"{transactions:,}")

def render_shop_view(df):
    """Shop-level breakdown"""
    st.subheader("🏪 Shop Performance")
    
    shop_summary = df.groupby('Shop').agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Trans No': 'nunique'
    }).reset_index()
    
    shop_summary.columns = ['Shop', 'Net Sales', 'Std Margin', 'True Margin', 
                            'Vendor Pays', 'Haven Pays', 'Margin Lift', 'Transactions']
    
    shop_summary['Std %'] = (shop_summary['Std Margin'] / shop_summary['Net Sales'] * 100).round(1)
    shop_summary['True %'] = (shop_summary['True Margin'] / shop_summary['Net Sales'] * 100).round(1)
    
    shop_summary = shop_summary.sort_values('Net Sales', ascending=False)
    
    # Format for display
    display_df = shop_summary.copy()
    for col in ['Net Sales', 'Std Margin', 'True Margin', 'Vendor Pays', 'Haven Pays', 'Margin Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    shop_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Shop Data", csv_buffer.getvalue(), 
                       "shop_true_margins.csv", "text/csv")

def render_brand_view(df, show_private_label_only=False):
    """Brand-level breakdown"""
    st.subheader("🏷️ Brand Performance")
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        pl_filter = st.selectbox(
            "Filter:",
            ["All Brands", "Private Label Only", "Non-Private Label"],
            key="brand_filter"
        )
    
    filtered = df.copy()
    if pl_filter == "Private Label Only":
        filtered = filtered[filtered['Is_Private_Label'] == True]
    elif pl_filter == "Non-Private Label":
        filtered = filtered[filtered['Is_Private_Label'] == False]
    
    brand_summary = filtered.groupby(['Brand', 'Is_Private_Label']).agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    brand_summary['Std %'] = (brand_summary['Standard_Margin'] / brand_summary['Net Sales'] * 100).round(1)
    brand_summary['True %'] = (brand_summary['True_Margin'] / brand_summary['Net Sales'] * 100).round(1)
    brand_summary['PL'] = brand_summary['Is_Private_Label'].map({True: '🏠', False: ''})
    
    brand_summary = brand_summary.sort_values('Net Sales', ascending=False)
    
    st.info(f"Showing {len(brand_summary):,} brands")
    
    # Display columns
    display_cols = ['PL', 'Brand', 'Net Sales', 'Quantity Sold', 'Std %', 'True %', 
                    'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    display_df = brand_summary[display_cols].head(100).copy()
    for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    brand_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Brand Data", csv_buffer.getvalue(),
                       "brand_true_margins.csv", "text/csv")

def render_category_view(df):
    """Category-level breakdown"""
    st.subheader("📂 Category Performance")
    
    cat_col = 'Product Category' if 'Product Category' in df.columns else 'Category'
    if cat_col not in df.columns:
        st.warning("No category column found in data")
        return
    
    cat_summary = df.groupby(cat_col).agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    cat_summary['Std %'] = (cat_summary['Standard_Margin'] / cat_summary['Net Sales'] * 100).round(1)
    cat_summary['True %'] = (cat_summary['True_Margin'] / cat_summary['Net Sales'] * 100).round(1)
    
    cat_summary = cat_summary.sort_values('Net Sales', ascending=False)
    
    # Format for display
    display_df = cat_summary.copy()
    for col in ['Net Sales', 'Standard_Margin', 'True_Margin', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    cat_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Category Data", csv_buffer.getvalue(),
                       "category_true_margins.csv", "text/csv")

def render_sku_type_view(df):
    """SKU Type (Profile Template) breakdown"""
    st.subheader("📦 SKU Type Performance")
    
    if 'Profile_Template' not in df.columns or df['Profile_Template'].isna().all():
        st.warning("No Profile Template data available. Upload Product Catalog to enable this view.")
        return
    
    # Exclude unmatched products for cleaner view
    matched = df[df['Profile_Template'].notna()]
    
    if len(matched) == 0:
        st.warning("No products matched to Profile Templates")
        return
    
    sku_summary = matched.groupby('Profile_Template').agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum',
        'Brand': 'first'
    }).reset_index()
    
    sku_summary['Std %'] = (sku_summary['Standard_Margin'] / sku_summary['Net Sales'] * 100).round(1)
    sku_summary['True %'] = (sku_summary['True_Margin'] / sku_summary['Net Sales'] * 100).round(1)
    
    sku_summary = sku_summary.sort_values('Net Sales', ascending=False)
    
    st.info(f"Showing {len(sku_summary):,} SKU Types ({len(matched):,}/{len(df):,} products matched)")
    
    display_cols = ['Profile_Template', 'Brand', 'Net Sales', 'Quantity Sold', 'Std %', 'True %',
                    'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    display_df = sku_summary[display_cols].head(100).copy()
    for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export
    csv_buffer = io.StringIO()
    sku_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download SKU Type Data", csv_buffer.getvalue(),
                       "sku_type_true_margins.csv", "text/csv")

def render_product_view(df):
    """Product-level detail"""
    st.subheader("📋 Product Detail")
    
    product_summary = df.groupby(['Product', 'Brand', 'Product Category'] if 'Product Category' in df.columns 
                                  else ['Product', 'Brand']).agg({
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum',
        'Profile_Template': 'first'
    }).reset_index()
    
    product_summary['Std %'] = (product_summary['Standard_Margin'] / product_summary['Net Sales'] * 100).round(1)
    product_summary['True %'] = (product_summary['True_Margin'] / product_summary['Net Sales'] * 100).round(1)
    
    product_summary = product_summary.sort_values('Net Sales', ascending=False)
    
    st.info(f"Showing top 100 of {len(product_summary):,} products")
    
    display_cols = ['Brand', 'Product', 'Net Sales', 'Quantity Sold', 'Std %', 'True %',
                    'Vendor_Pays', 'Haven_Pays']
    
    display_df = product_summary[display_cols].head(100).copy()
    for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
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
    
    # =====================
    # SIDEBAR - Data Upload
    # =====================
    st.sidebar.header("📁 Data Sources")
    
    st.sidebar.subheader("Required Files")
    
    # Multi-file upload for Sales Detail
    sales_files = st.sidebar.file_uploader(
        "📊 Sales Reports (Blaze POS)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more Total Sales Detail exports from Blaze POS. Files will be combined automatically."
    )
    
    if sales_files:
        st.sidebar.success(f"✅ {len(sales_files)} sales file(s) selected")
        for f in sales_files:
            st.sidebar.text(f"  • {f.name}")
    
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
    
    # Changelog
    with st.sidebar.expander("📋 Version History"):
        st.markdown("""
        **v1.1.0** (Current - 2026-01-14)
        - 📁 Multi-file upload for Sales Detail
        - ⚡ Auto-processing (no Load button)
        - 🔄 Chunked loading for large files
        - 📊 Progress indicators throughout
        
        **v1.0.0** (2026-01-03)
        - 🚀 Initial release
        - 🔗 Fuzzy matching within transactions
        - 📦 Profile Template matching
        - 🏠 Private Label identification
        - 📊 Multi-level aggregation views
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version {VERSION}**")
    
    # =====================
    # MAIN CONTENT
    # =====================
    
    # Welcome screen when no files uploaded
    if not sales_files or not credit_file:
        st.info("👆 Upload Sales Reports and Promo Credit Report to get started")
        
        st.subheader("📚 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### True Margin Formula
            
            ```
            True COGS = Standard COGS - Vendor Pays + Haven Pays
            True Margin = Net Sales - True COGS
            Margin Lift = Vendor Pays - Haven Pays
            ```
            
            ### Key Concepts
            
            **💚 Vendor Pays** - Credit received from vendors that reduces your effective COGS.
            
            **🔴 Haven Pays** - Discount cost Haven absorbs that increases your effective COGS.
            
            **📈 Margin Lift** - Net impact of promo programs on margins.
            """)
        
        with col2:
            st.markdown("""
            ### v1.1.0 New Features
            
            **📁 Multi-File Upload**
            - Upload multiple monthly exports
            - Files are combined automatically
            - Duplicates removed by transaction
            
            **⚡ Auto-Processing**
            - No more Load Data button
            - Processes immediately on upload
            - Optimized for large files
            
            **🔄 Chunked Loading**
            - Handles files >100MB
            - No more timeouts
            - Progress indicators throughout
            """)
        
        return
    
    # =====================
    # AUTO-PROCESS ON UPLOAD
    # =====================
    
    # Create a unique key from uploaded files to detect changes
    file_key = "_".join(sorted([f.name for f in sales_files])) + "_" + (credit_file.name if credit_file else "")
    
    # Check if we need to reprocess
    needs_processing = (
        'processed_data' not in st.session_state or
        st.session_state.get('file_key') != file_key
    )
    
    if needs_processing:
        st.info("🔄 Processing data...")
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        def update_progress(pct, text):
            progress_bar.progress(min(pct, 1.0))
            status_text.text(text)
        
        # Step 1: Load Sales Files
        update_progress(0.05, "Loading sales files...")
        sales_df = load_multiple_sales_files(sales_files, update_progress)
        
        if sales_df is None:
            st.error("❌ Failed to load sales files")
            return
        
        st.success(f"✅ Loaded {len(sales_df):,} sales rows from {len(sales_files)} file(s)")
        
        # Step 2: Load Credit File
        update_progress(0.25, "Loading credit report...")
        credit_df = load_credit_file(credit_file, update_progress)
        
        if credit_df is not None:
            st.success(f"✅ Loaded {len(credit_df):,} credit rows")
        
        # Step 3: Load Optional Files
        catalog_df = None
        if catalog_file is not None:
            update_progress(0.3, "Loading product catalog...")
            try:
                catalog_df = pd.read_csv(catalog_file)
                st.success(f"✅ Loaded {len(catalog_df):,} catalog entries")
            except Exception as e:
                st.warning(f"Could not load catalog: {e}")
        
        private_labels = DEFAULT_PRIVATE_LABELS
        if private_label_file is not None:
            try:
                pl_df = pd.read_csv(private_label_file)
                if 'Name' in pl_df.columns:
                    private_labels = pl_df['Name'].tolist()
                    st.success(f"✅ Loaded {len(private_labels)} private label brands")
            except Exception as e:
                st.warning(f"Could not load private labels: {e}")
        
        # Step 4: Process Data
        processed_df = process_data(sales_df, credit_df, catalog_df, private_labels, progress_container)
        
        if processed_df is None:
            st.error("❌ Data processing failed")
            return
        
        # Store in session state
        st.session_state['processed_data'] = processed_df
        st.session_state['file_key'] = file_key
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Processed {len(processed_df):,} products successfully!")
        
        # Force rerun to show results
        st.rerun()
    
    # =====================
    # DISPLAY RESULTS
    # =====================
    
    processed_df = st.session_state['processed_data']
    
    # Shop filter
    st.sidebar.subheader("🏪 Shop Filter")
    available_shops = sorted(processed_df['Shop'].dropna().unique().tolist())
    
    selected_shops = st.sidebar.multiselect(
        "Select Shops:",
        options=available_shops,
        default=available_shops,
        help="Filter data to specific shops"
    )
    
    if selected_shops:
        filtered_df = processed_df[processed_df['Shop'].isin(selected_shops)]
    else:
        filtered_df = processed_df
    
    st.sidebar.info(f"📊 {len(filtered_df):,} products in view")
    
    # Reprocess button
    if st.sidebar.button("🔄 Reprocess Data"):
        if 'processed_data' in st.session_state:
            del st.session_state['processed_data']
        if 'file_key' in st.session_state:
            del st.session_state['file_key']
        st.rerun()
    
    # Validation metrics
    st.subheader("📊 Data Summary")
    
    vendor_captured = filtered_df['Vendor_Pays'].sum()
    haven_captured = filtered_df['Haven_Pays'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", f"{len(filtered_df):,}")
    with col2:
        transactions = filtered_df['Trans No'].nunique() if 'Trans No' in filtered_df.columns else 0
        st.metric("Transactions", f"{transactions:,}")
    with col3:
        st.metric("Vendor Credits", format_currency(vendor_captured))
    with col4:
        st.metric("Haven Costs", format_currency(haven_captured))
    
    # Tabs for different views
    tab_names = ["📊 Overview", "🏷️ Brands", "🏪 Shops", "📂 Categories", "📦 SKU Types", "📋 Products"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Overview
        render_network_view(filtered_df)
    
    with tabs[1]:  # Brands
        render_brand_view(filtered_df)
    
    with tabs[2]:  # Shops
        render_shop_view(filtered_df)
    
    with tabs[3]:  # Categories
        render_category_view(filtered_df)
    
    with tabs[4]:  # SKU Types
        render_sku_type_view(filtered_df)
    
    with tabs[5]:  # Products
        render_product_view(filtered_df)

if __name__ == "__main__":
    main()