"""
True Margin Calculator v1.2.0
Calculate true product margins by incorporating vendor promo credits

Merges Blaze POS sales data with Haven Promo Performance vendor credit data
to show the actual margin after accounting for vendor-paid promotions.

MATCHING ENGINE: Adapted from Price Checker v4.3.3
- Uses same Brand + Category + Weight + Keyword matching logic
- Column mapping: Blaze 'Product' → 'Item', 'Product Category' → 'Category'

CHANGELOG:
v1.2.0 (2026-01-14)
- RESTORED: Load Data button for consistency with other Haven apps
- FIXED: Profile Template matching now uses Price Checker matching engine
- ADDED: Column mapping for Blaze exports (Product→Item, Product Category→Category)
- ADDED: Full smart matching with weight/keyword extraction
- NOTE: Matching engine adapted from Price Checker v4.3.3

v1.1.0 (2026-01-14)
- ENHANCEMENT: Multi-file upload for Sales Detail reports
- ENHANCEMENT: Chunked file reading for large CSVs
- ENHANCEMENT: Progress indicators throughout processing

v1.0.0 (2026-01-03)
- Initial release
- Fuzzy matching within transactions for credit allocation
- Aggregation levels: Network, Shop, Brand, Category, SKU Type, Product

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
    page_title="True Margin Calculator v1.2.0",
    page_icon="💰",
    layout="wide"
)

VERSION = "1.2.0"
MATCHING_ENGINE_VERSION = "Price Checker v4.3.3"

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

# Brands requiring exact product-level matching (from Price Checker v4.3.3)
EXACT_PRODUCT_MATCH_BRANDS = {
    'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide', 
    'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt', 
    'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD', 
    'Wyld', 'Yummi Karma', "Not Your Father's"
}

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
# EXTRACTION FUNCTIONS - FROM PRICE CHECKER v4.3.3
# ============================================================================

def extract_weight_from_item(item_text):
    """Extract weight from item text (e.g., "Blue Dream 3.5g" → "3.5g")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    
    # Try patterns at end of string first (most common)
    end_patterns = [
        r'(\d+\.?\d*g)$',
        r'(\d+\.\d+\s?oz?)$',
        r'(\d+\s?oz?)$',
        r'(1/8\s?oz?)$',
        r'(1/4\s?oz?)$',
        r'(1/2\s?oz?)$',
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    # If not found at end, try finding anywhere in string
    anywhere_patterns = [
        r'(\d+\.?\d*g)',
        r'(\d+\.\d+\s?oz?)',
        r'(\d+\s?oz?)',
        r'(1/8\s?oz?)',
        r'(1/4\s?oz?)',
        r'(1/2\s?oz?)',
    ]
    
    for pattern in anywhere_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    return None

def extract_pack_size_from_item(item_text):
    """Extract pack size from item text (e.g., "OG Kush 3pk 1.5g" → "3pk")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    
    # Pattern 1: Pack before weight
    pack_before_weight_patterns = [
        r'(\d+pk)\s+\d+\.?\d*g',
        r'(\d+pk)\s+\d+\s?oz',
        r'(\d+pk)\s+1/[248]\s?oz',
    ]
    
    for pattern in pack_before_weight_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Pattern 2: Weight before pack
    weight_before_pack_patterns = [
        r'\d+\.?\d*g\s*(\d+pk)',
        r'\d+\s?oz\s*(\d+pk)',
        r'1/[248]\s?oz\s*(\d+pk)',
    ]
    
    for pattern in weight_before_pack_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Pattern 3: Standalone pack size anywhere (fallback)
    standalone_pattern = r'(\d+pk)'
    match = re.search(standalone_pattern, item_str, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    return None

def extract_category_keywords(item_text, category):
    """Extract category-specific distinguishing keywords from item text"""
    if pd.isna(item_text) or pd.isna(category):
        return None
    
    item_str = str(item_text).lower()
    category_lower = str(category).lower()
    
    if category_lower == 'vape':
        vape_keywords = ['originals', 'ascnd', 'dna', 'exotics', 'disposable', 'live resin', 'reload', 'rtu', 'curepen', 'curebar']
        found_keywords = [keyword for keyword in vape_keywords if keyword in item_str]
        return ', '.join(found_keywords) if found_keywords else None
    
    if 'flower' in category_lower:
        quality_tiers = ['top shelf', 'headstash', 'exotic', 'premium', 'private reserve', 'reserve']
        found_keywords = [tier for tier in quality_tiers if tier in item_str]
        return ', '.join(found_keywords) if found_keywords else None
    
    if category_lower == 'extract':
        found_keywords = []
        
        if 'live rosin' in item_str:
            found_keywords.append('live rosin')
        elif 'live resin' in item_str:
            found_keywords.append('live resin')
        elif 'hash rosin' in item_str:
            found_keywords.append('hash rosin')
        elif 'rosin' in item_str:
            found_keywords.append('rosin')
        elif 'resin' in item_str:
            found_keywords.append('resin')
        
        if any(brand in item_str for brand in ['bear labs', 'west coast cure']):
            tier_match = re.search(r'tier\s*([1-4])', item_str)
            if tier_match:
                found_keywords.append(f"tier {tier_match.group(1)}")
        
        modifiers = ['cold cure', 'fresh press', 'curated', 'hte blend', 'dino eggz']
        found_keywords.extend([modifier for modifier in modifiers if modifier in item_str])
        
        consistencies = ['diamonds', 'budder', 'badder', 'sauce', 'sugar', 'jam']
        found_keywords.extend([consistency for consistency in consistencies if consistency in item_str])
        
        product_types = ['rso', 'syringe']
        found_keywords.extend([product_type for product_type in product_types if product_type in item_str])
        
        return ', '.join(found_keywords) if found_keywords else None
    
    if category_lower == 'preroll':
        found_keywords = []
        preroll_types = ['blunts', 'preroll', 'prerolls', 'joints', 'mini']
        found_keywords.extend([preroll_type for preroll_type in preroll_types if preroll_type in item_str])
        
        if 'infused' in item_str:
            found_keywords.append('infused')
        
        return ', '.join(found_keywords) if found_keywords else None
    
    return None

# ============================================================================
# PATTERN MATCHING FUNCTIONS - FROM PRICE CHECKER v4.3.3
# ============================================================================

def match_placeholder_pattern(product_name, template_name):
    """Check if a product name matches a template with placeholder patterns (COLOR, STRAIN, FLAVOR)"""
    if pd.isna(product_name) or pd.isna(template_name):
        return False
    
    placeholders = ['STRAIN', 'COLOR', 'FLAVOR', 'SIZE', 'VARIANT']
    
    has_placeholder = any(placeholder in str(template_name).upper() for placeholder in placeholders)
    if not has_placeholder:
        return False
    
    product_upper = str(product_name).upper()
    template_upper = str(template_name).upper()
    
    for placeholder in placeholders:
        if placeholder in template_upper:
            parts = template_upper.split(placeholder)
            
            if len(parts) != 2:
                continue
            
            prefix, suffix = parts
            
            if product_upper.startswith(prefix) and product_upper.endswith(suffix):
                placeholder_value = product_upper[len(prefix):-len(suffix) if suffix else len(product_upper)]
                
                if placeholder_value and len(placeholder_value.strip()) > 0 and len(placeholder_value.strip()) < 50:
                    return True
    
    return False

# ============================================================================
# FILE LOADING FUNCTIONS
# ============================================================================

def load_multiple_sales_files(uploaded_files, progress_callback=None):
    """Load and combine multiple Total Sales Detail CSV files"""
    if not uploaded_files:
        return None
    
    all_dfs = []
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        try:
            if progress_callback:
                progress_callback((i / total_files) * 0.5, f"Loading file {i+1}/{total_files}: {file.name}")
            
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > 50 * 1024 * 1024:
                chunks = []
                chunk_size = 50000
                for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file, low_memory=False)
            
            df['_Source_File'] = file.name
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
            continue
    
    if not all_dfs:
        return None
    
    if progress_callback:
        progress_callback(0.6, "Combining files...")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if progress_callback:
        progress_callback(0.7, "Removing duplicates...")
    
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
    """Load Promo Credit Report CSV"""
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
# CREDIT MATCHING FUNCTIONS
# ============================================================================

def match_credits_to_sales(sales_df, credit_df, progress_callback=None):
    """Match credit rows to sales rows using transaction-level fuzzy matching"""
    if progress_callback:
        progress_callback(0.0, "Preparing credit matching...")
    
    sales_df = sales_df.copy()
    sales_df['Vendor_Pays'] = 0.0
    sales_df['Haven_Pays'] = 0.0
    sales_df['Credit_Match_Score'] = 0.0
    sales_df['Credit_Product_Matched'] = None
    
    # Blaze uses 'Trans No', Credit report uses 'Trans No'
    trans_col_sales = 'Trans No' if 'Trans No' in sales_df.columns else None
    trans_col_credit = 'Trans No' if 'Trans No' in credit_df.columns else ('Transaction ID' if 'Transaction ID' in credit_df.columns else None)
    
    if not trans_col_sales or not trans_col_credit:
        st.warning("Could not find transaction columns for matching")
        return sales_df
    
    credit_transactions = credit_df[trans_col_credit].dropna().unique()
    total_trans = len(credit_transactions)
    
    if progress_callback:
        progress_callback(0.05, f"Matching {total_trans:,} transactions with credits...")
    
    credit_by_trans = credit_df.groupby(trans_col_credit)
    matched_count = 0
    
    for i, trans_no in enumerate(credit_transactions):
        if i % 500 == 0 and progress_callback:
            progress = 0.05 + (i / total_trans) * 0.9
            progress_callback(progress, f"Matching transaction {i+1:,}/{total_trans:,}")
        
        trans_credits = credit_by_trans.get_group(trans_no).copy()
        
        sales_mask = sales_df[trans_col_sales] == trans_no
        if not sales_mask.any():
            continue
        
        trans_sales_indices = sales_df[sales_mask].index.tolist()
        available_sales_indices = trans_sales_indices.copy()
        
        for _, credit_row in trans_credits.iterrows():
            if not available_sales_indices:
                break
            
            credit_product = str(credit_row.get('Product', ''))
            vendor_pays = clean_price(credit_row.get('Vendor Pays', 0))
            haven_pays = clean_price(credit_row.get('Haven Pays', 0))
            
            best_idx = None
            best_score = 0.0
            
            for sales_idx in available_sales_indices:
                sales_product = str(sales_df.at[sales_idx, 'Product'])
                score = similarity_score(credit_product, sales_product)
                
                if score > best_score:
                    best_score = score
                    best_idx = sales_idx
            
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

# ============================================================================
# PROFILE TEMPLATE MATCHING - ADAPTED FROM PRICE CHECKER v4.3.3
# ============================================================================

def normalize_category(category):
    """Normalize category names to match Product Catalog"""
    if pd.isna(category):
        return 'Unknown'
    
    category_str = str(category).strip()
    
    # Blaze uses "Flower (Indica)" etc, catalog uses "Flower"
    flower_mapping = {
        'Flower (Indica)': 'Flower',
        'Flower (Sativa)': 'Flower', 
        'Flower (Hybrid)': 'Flower'
    }
    
    return flower_mapping.get(category_str, category_str)

def add_profile_template_matching(sales_df, catalog_df, progress_callback=None):
    """
    Match sales products to Profile Templates using Price Checker's smart matching engine
    
    Column mapping:
    - Blaze 'Product' → Price Checker 'Item'
    - Blaze 'Product Category' → Price Checker 'Category'
    """
    if catalog_df is None or catalog_df.empty:
        sales_df['Profile_Template'] = None
        sales_df['Match_Type'] = None
        return sales_df
    
    # Check required columns in catalog
    if 'Brand' not in catalog_df.columns or 'Profile Template' not in catalog_df.columns:
        st.warning(f"⚠️ Product Catalog missing required columns. Found: {list(catalog_df.columns)[:10]}")
        st.warning("Required: 'Brand', 'Profile Template'")
        sales_df['Profile_Template'] = None
        sales_df['Match_Type'] = None
        return sales_df
    
    if progress_callback:
        progress_callback(0.0, "Building catalog lookup tables...")
    
    # Initialize matching columns
    sales_df['Profile_Template'] = None
    sales_df['Match_Type'] = None
    sales_df['Match_Keywords'] = None
    
    # Map Blaze column names to matching engine expectations
    # Blaze: 'Product' → 'Item', 'Product Category' → 'Category'
    product_col = 'Product'
    category_col = 'Product Category' if 'Product Category' in sales_df.columns else 'Category'
    
    # Normalize categories in sales data
    sales_df['_Normalized_Category'] = sales_df[category_col].apply(normalize_category)
    
    # Build brand and category mappings from catalog
    brand_catalog_map = {}
    brand_category_catalog_map = {}
    
    for _, cat_row in catalog_df.iterrows():
        brand = cat_row['Brand']
        template = cat_row['Profile Template']
        category = cat_row.get('Category', 'Unknown')
        
        if pd.notna(brand) and pd.notna(template) and str(template).strip():
            if brand not in brand_catalog_map:
                brand_catalog_map[brand] = []
            if template not in brand_catalog_map[brand]:
                brand_catalog_map[brand].append(template)
            
            brand_category_key = f"{brand}|{category}"
            if brand_category_key not in brand_category_catalog_map:
                brand_category_catalog_map[brand_category_key] = []
            if template not in brand_category_catalog_map[brand_category_key]:
                brand_category_catalog_map[brand_category_key].append(template)
    
    # Categorize brands by complexity
    single_entry_brands = {b: t[0] for b, t in brand_catalog_map.items() if len(t) == 1}
    multiple_entry_brands = {b: t for b, t in brand_catalog_map.items() if len(t) > 1}
    
    single_entry_brand_categories = {k: t[0] for k, t in brand_category_catalog_map.items() if len(t) == 1}
    
    # Filter out exact match brands
    filtered_single_entry_brands = {b: t for b, t in single_entry_brands.items() 
                                    if b not in EXACT_PRODUCT_MATCH_BRANDS}
    filtered_single_entry_brand_categories = {k: t for k, t in single_entry_brand_categories.items() 
                                              if k.split('|')[0] not in EXACT_PRODUCT_MATCH_BRANDS}
    
    if progress_callback:
        progress_callback(0.1, f"Matching {len(sales_df):,} products to {len(brand_catalog_map)} brands...")
    
    # Extract weights and keywords for matching
    sales_df['_Extracted_Weight'] = sales_df[product_col].apply(extract_weight_from_item)
    sales_df['_Extracted_Pack_Size'] = sales_df[product_col].apply(extract_pack_size_from_item)
    sales_df['_Extracted_Keywords'] = sales_df.apply(
        lambda row: extract_category_keywords(row[product_col], row['_Normalized_Category']), axis=1
    )
    
    # Matching counters
    exact_matches = 0
    placeholder_matches = 0
    single_entry_matches = 0
    brand_category_matches = 0
    advanced_matches = 0
    no_matches = 0
    
    total_rows = len(sales_df)
    
    for counter, (idx, row) in enumerate(sales_df.iterrows()):
        if counter % 1000 == 0 and progress_callback:
            progress = 0.1 + (counter / total_rows) * 0.85
            progress_callback(progress, f"Matching product {counter+1:,}/{total_rows:,}")
        
        brand = row['Brand']
        product = row[product_col]
        category = row['_Normalized_Category']
        
        if pd.isna(brand) or pd.isna(product):
            no_matches += 1
            continue
        
        match_found = False
        
        # Strategy 1: Exact match
        if brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if str(product).lower() == str(template).lower():
                    sales_df.at[idx, 'Profile_Template'] = template
                    sales_df.at[idx, 'Match_Type'] = 'exact'
                    exact_matches += 1
                    match_found = True
                    break
        
        # Strategy 2: Placeholder pattern match (COLOR, STRAIN, FLAVOR)
        if not match_found and brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if match_placeholder_pattern(product, template):
                    sales_df.at[idx, 'Profile_Template'] = template
                    sales_df.at[idx, 'Match_Type'] = 'placeholder'
                    placeholder_matches += 1
                    match_found = True
                    break
        
        # Strategy 3: Single entry brand auto-match
        if not match_found and brand in filtered_single_entry_brands:
            template = filtered_single_entry_brands[brand]
            sales_df.at[idx, 'Profile_Template'] = template
            sales_df.at[idx, 'Match_Type'] = 'brand_auto'
            single_entry_matches += 1
            match_found = True
        
        # Strategy 4: Brand+category auto-match
        if not match_found:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in filtered_single_entry_brand_categories:
                template = filtered_single_entry_brand_categories[brand_category_key]
                sales_df.at[idx, 'Profile_Template'] = template
                sales_df.at[idx, 'Match_Type'] = 'brand_category_auto'
                brand_category_matches += 1
                match_found = True
        
        # Strategy 5: Advanced weight/keyword matching
        if not match_found and category in ['Flower', 'Preroll', 'Vape', 'Extract']:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in brand_category_catalog_map:
                templates = brand_category_catalog_map[brand_category_key]
                
                if len(templates) > 1:
                    # Try weight matching first
                    product_weight = row['_Extracted_Weight']
                    if product_weight:
                        weight_matched = []
                        for t in templates:
                            t_weight = extract_weight_from_item(t)
                            if t_weight == product_weight:
                                weight_matched.append(t)
                        
                        if len(weight_matched) == 1:
                            sales_df.at[idx, 'Profile_Template'] = weight_matched[0]
                            sales_df.at[idx, 'Match_Type'] = 'weight_match'
                            sales_df.at[idx, 'Match_Keywords'] = f"weight: {product_weight}"
                            advanced_matches += 1
                            match_found = True
        
        if not match_found:
            no_matches += 1
    
    if progress_callback:
        progress_callback(0.95, "Finalizing matching...")
    
    # Clean up temporary columns
    temp_cols = ['_Normalized_Category', '_Extracted_Weight', '_Extracted_Pack_Size', '_Extracted_Keywords']
    for col in temp_cols:
        if col in sales_df.columns:
            sales_df.drop(columns=[col], inplace=True)
    
    # Report results
    total_matched = exact_matches + placeholder_matches + single_entry_matches + brand_category_matches + advanced_matches
    
    if progress_callback:
        progress_callback(1.0, f"Matched {total_matched:,}/{total_rows:,} products to Profile Templates")
    
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
    
    # Clean numeric columns (EXACT Blaze column names)
    df['Net Sales'] = df['Net Sales'].apply(clean_price)
    df['Unit Cost'] = df['Unit Cost'].apply(clean_price)
    df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce').fillna(0)
    
    # Calculate Standard COGS and Margin
    df['Standard_COGS'] = df['Unit Cost'] * df['Quantity Sold']
    df['Standard_Margin'] = df['Net Sales'] - df['Standard_COGS']
    
    # Calculate True COGS and Margin
    df['True_COGS'] = df['Standard_COGS'] - df['Vendor_Pays'] + df['Haven_Pays']
    df['True_Margin'] = df['Net Sales'] - df['True_COGS']
    
    # Margin Lift
    df['Margin_Lift'] = df['Vendor_Pays'] - df['Haven_Pays']
    
    # Private Label flag
    df['Is_Private_Label'] = df['Brand'].isin(private_labels)
    
    # Margin percentages
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
    """Main processing pipeline"""
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    def update_progress(pct, text):
        progress_bar.progress(min(pct, 1.0))
        status_text.text(text)
    
    # Step 1: Validate sales data
    update_progress(0.05, "Validating sales data...")
    
    # EXACT column names from Blaze Total Sales Detail export
    required_cols = ['Trans No', 'Product', 'Net Sales', 'Unit Cost', 'Quantity Sold']
    missing_cols = [col for col in required_cols if col not in sales_df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns in sales data: {missing_cols}")
        st.info(f"Available columns: {list(sales_df.columns)[:15]}...")
        return None
    
    # Step 2: Match credits to sales
    update_progress(0.1, "Matching credits to sales transactions...")
    
    if credit_df is not None and not credit_df.empty:
        sales_df = match_credits_to_sales(
            sales_df, credit_df, 
            lambda p, t: update_progress(0.1 + p * 0.4, t)
        )
    else:
        sales_df['Vendor_Pays'] = 0.0
        sales_df['Haven_Pays'] = 0.0
    
    # Step 3: Match Profile Templates
    update_progress(0.55, "Matching Profile Templates...")
    
    sales_df = add_profile_template_matching(
        sales_df, catalog_df,
        lambda p, t: update_progress(0.55 + p * 0.3, t)
    )
    
    # Step 4: Calculate margins
    update_progress(0.9, "Calculating margins...")
    
    sales_df = calculate_margins(sales_df, private_labels)
    
    # Step 5: Final cleanup
    update_progress(0.95, "Finalizing...")
    
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
    
    display_df = shop_summary.copy()
    for col in ['Net Sales', 'Std Margin', 'True Margin', 'Vendor Pays', 'Haven Pays', 'Margin Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    csv_buffer = io.StringIO()
    shop_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Shop Data", csv_buffer.getvalue(), 
                       "shop_true_margins.csv", "text/csv")

def render_brand_view(df):
    """Brand-level breakdown"""
    st.subheader("🏷️ Brand Performance")
    
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
    
    display_cols = ['PL', 'Brand', 'Net Sales', 'Quantity Sold', 'Std %', 'True %', 
                    'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    display_df = brand_summary[display_cols].head(100).copy()
    for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
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
    
    display_df = cat_summary.copy()
    for col in ['Net Sales', 'Standard_Margin', 'True_Margin', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    csv_buffer = io.StringIO()
    cat_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Category Data", csv_buffer.getvalue(),
                       "category_true_margins.csv", "text/csv")

def render_sku_type_view(df):
    """SKU Type (Profile Template) breakdown"""
    st.subheader("📦 SKU Type Performance")
    
    if 'Profile_Template' not in df.columns:
        st.warning("Profile Template column not found. Upload Product Catalog and click Load Data.")
        return
    
    matched = df[df['Profile_Template'].notna()]
    unmatched = df[df['Profile_Template'].isna()]
    
    if len(matched) == 0:
        st.warning("⚠️ No products matched to Profile Templates.")
        st.info("Check that your Product Catalog has 'Brand' and 'Profile Template' columns.")
        
        # Show matching diagnostics
        if 'Match_Type' in df.columns:
            st.write("**Matching Diagnostics:**")
            st.write(f"- Total products: {len(df):,}")
            st.write(f"- Matched: {len(matched):,}")
            st.write(f"- Unmatched: {len(unmatched):,}")
        return
    
    match_rate = len(matched) / len(df) * 100
    st.success(f"✅ {len(matched):,}/{len(df):,} products matched ({match_rate:.1f}%)")
    
    # Show match type breakdown
    if 'Match_Type' in df.columns:
        match_types = df[df['Profile_Template'].notna()]['Match_Type'].value_counts()
        with st.expander("📊 Match Type Breakdown"):
            for match_type, count in match_types.items():
                st.write(f"- **{match_type}**: {count:,}")
    
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
    
    st.info(f"Showing {len(sku_summary):,} SKU Types")
    
    display_cols = ['Profile_Template', 'Brand', 'Net Sales', 'Quantity Sold', 'Std %', 'True %',
                    'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']
    
    display_df = sku_summary[display_cols].head(100).copy()
    for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
        display_df[col] = display_df[col].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    csv_buffer = io.StringIO()
    sku_summary.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download SKU Type Data", csv_buffer.getvalue(),
                       "sku_type_true_margins.csv", "text/csv")

def render_product_view(df):
    """Product-level detail"""
    st.subheader("📋 Product Detail")
    
    group_cols = ['Product', 'Brand']
    if 'Product Category' in df.columns:
        group_cols.append('Product Category')
    
    product_summary = df.groupby(group_cols).agg({
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
        st.markdown(f"""
        **v1.2.0** (Current - 2026-01-14)
        - 🔧 RESTORED: Load Data button
        - 🔧 FIXED: Profile Template matching
        - 🧠 Uses {MATCHING_ENGINE_VERSION} matching engine
        - 📁 Multi-file upload for Sales Detail
        
        **v1.1.0** (2026-01-14)
        - 📁 Multi-file upload (broken Profile Template)
        
        **v1.0.0** (2026-01-03)
        - 🚀 Initial release
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version {VERSION}**")
    st.sidebar.markdown(f"_Matching: {MATCHING_ENGINE_VERSION}_")
    
    # =====================
    # LOAD DATA BUTTON
    # =====================
    
    files_ready = sales_files and credit_file
    
    if st.sidebar.button("🚀 Load Data", type="primary", disabled=not files_ready):
        with st.spinner("Loading and processing data..."):
            
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
                    st.info(f"📋 Catalog columns: {list(catalog_df.columns)[:8]}...")
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
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Processed {len(processed_df):,} products successfully!")
            
            # Force rerun to show results
            st.rerun()
    
    # =====================
    # MAIN CONTENT
    # =====================
    
    # Welcome screen when no data processed
    if 'processed_data' not in st.session_state:
        st.info("👆 Upload files and click **Load Data** to get started")
        
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
            st.markdown(f"""
            ### v1.2.0 Features
            
            **🔧 Consistent Design**
            - Load Data button (like other Haven apps)
            - Progress indicators throughout
            
            **🧠 Smart Matching Engine**
            - Uses {MATCHING_ENGINE_VERSION} matching logic
            - Brand + Category + Weight matching
            - Placeholder patterns (COLOR, STRAIN, FLAVOR)
            
            **📁 Multi-File Upload**
            - Upload multiple monthly exports
            - Automatic deduplication
            - Chunked loading for large files
            """)
        
        return
    
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
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear Data"):
        if 'processed_data' in st.session_state:
            del st.session_state['processed_data']
        st.rerun()
    
    # Validation metrics
    st.subheader("📊 Data Summary")
    
    vendor_captured = filtered_df['Vendor_Pays'].sum()
    haven_captured = filtered_df['Haven_Pays'].sum()
    profile_matched = filtered_df['Profile_Template'].notna().sum() if 'Profile_Template' in filtered_df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", f"{len(filtered_df):,}")
    with col2:
        transactions = filtered_df['Trans No'].nunique() if 'Trans No' in filtered_df.columns else 0
        st.metric("Transactions", f"{transactions:,}")
    with col3:
        st.metric("Vendor Credits", format_currency(vendor_captured))
    with col4:
        match_pct = (profile_matched / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("SKU Type Match", f"{match_pct:.0f}%")
    
    # Tabs for different views
    tab_names = ["📊 Overview", "🏷️ Brands", "🏪 Shops", "📂 Categories", "📦 SKU Types", "📋 Products"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        render_network_view(filtered_df)
    
    with tabs[1]:
        render_brand_view(filtered_df)
    
    with tabs[2]:
        render_shop_view(filtered_df)
    
    with tabs[3]:
        render_category_view(filtered_df)
    
    with tabs[4]:
        render_sku_type_view(filtered_df)
    
    with tabs[5]:
        render_product_view(filtered_df)

if __name__ == "__main__":
    main()