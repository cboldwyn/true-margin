"""
True Margin Calculator v1.3.2
Calculate true product margins by incorporating vendor promo credits

Merges Blaze POS sales data with Haven Promo Performance vendor credit data
to show the actual margin after accounting for vendor-paid promotions.

MATCHING ENGINE: Adapted from Price Checker v4.3.22
- Uses same Brand + Category + Weight + Keyword matching logic
- Column mapping: Blaze 'Product' → 'Item', 'Product Category' → 'Category'
- Category-aware placeholder/wildcard matching (prevents cross-category mismatches)
- Apparel matching with size stripping
- Turn Up/Turn Down pattern handling

CHANGELOG:
v1.3.2 (2026-02-04)
- ENHANCED: Product Detail tab rebuilt with rich filtering controls
  Text search (comma-separated terms), Brand, Category, SKU Type (Template),
  Weight, Pack Size, Match Status filters
- ADDED: Selection Summary banner - recalculates combined True Margin % properly
  from filtered set (sums raw values, derives percentages from totals)
- ADDED: Filters act as ad-hoc grouping (e.g. filter to WCC 3pk to see combined margins)
- ADDED: Weight and Pack Size extracted for filter dropdowns
- ADDED: Filtered + Full download options
- FIX: Match_Keywords column check in Debug tab (conditional aggregation)
- FIX: Search help text clarifies space=AND, comma=OR behavior
- FIX: Broad apostrophe normalization - Phase 2 regex catches ALL orphaned " s " patterns
  beyond just brand prefixes (fixes "40 s" → "40's", "Mother s" → "Mother's",
  "Women s" → "Women's", and Juicy Jay products under wrong brand columns)

v1.3.1 (2026-02-04)
- FIX: Product name apostrophe normalization (Dr. Norm s → Dr. Norm's, etc.)
  Blaze POS strips apostrophes from some product names, breaking exact matching
  for brands like Dr. Norm's, Uncle Arnie's, Not Your Father's (44 products fixed)
- ENHANCED: Product Detail view - removed top 100 limit, added display controls
- ADDED: Debug/Diagnostics tab with unmatched product analysis + export
- ADDED: Match type breakdown and matching engine diagnostics

v1.3.0 (2026-02-04)
- CRITICAL FIX: Strategy 2 placeholder matching now category-aware (v4.3.21 fix)
  Was searching ALL brand templates, now searches only brand+category templates
- CRITICAL FIX: Added placeholder specificity scoring (picks most specific match)
- ADDED: Strategy 2.5 - Wildcard matching with category-awareness + specificity scoring
  match_wildcard_template() was defined but never called - now fully integrated
- CRITICAL FIX: Strategy 5 upgraded from bare-bones weight-only matching to full
  category-specific matching (flower/preroll/vape/extract/apparel)
- ADDED: _match_flower(), _match_preroll(), _match_vape_extract() helper functions
  Adapted from Price Checker with weight+keyword+pack size matching
- ADDED: skip_auto_matching check for Strategies 3/4 (EXACT_PRODUCT_MATCH_BRANDS)
- ADDED: Apparel to Strategy 5 categories

v1.2.1 (2026-01-28)
- UPDATED: Matching engine to Price Checker v4.3.22
- CRITICAL: Placeholder/wildcard matching now respects categories (v4.3.21)
- ADDED: Apparel matching with size stripping (v4.3.17)
- ADDED: Turn Up/Turn Down pattern handling (v4.3.18)
- CHANGED: Apparel removed from excluded categories (v4.3.16)

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
    page_title="True Margin Calculator v1.3.2",
    page_icon="💰",
    layout="wide"
)

VERSION = "1.3.2"
MATCHING_ENGINE_VERSION = "Price Checker v4.3.22"

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
    """
    Check if a product name matches a template with placeholder patterns (COLOR, STRAIN, FLAVOR)
    
    Handles placeholders like:
    - STRAIN (any strain name)
    - COLOR (any color)
    - FLAVOR (any flavor)
    - SIZE (any size)
    
    Special handling for Stiiizy patterns where STRAIN comes BEFORE bag type:
    - "Stiiizy - Black Bag Black Cherry 3.5g" matches "Stiiizy - STRAIN Black Bag 3.5g"
    
    Updated in v4.3.18 for Turn Up/Turn Down pattern handling
    """
    if pd.isna(product_name) or pd.isna(template_name):
        return False
    
    placeholders = ['STRAIN', 'COLOR', 'FLAVOR', 'SIZE', 'VARIANT']
    
    has_placeholder = any(placeholder in str(template_name).upper() for placeholder in placeholders)
    if not has_placeholder:
        return False
    
    product_upper = str(product_name).upper()
    template_upper = str(template_name).upper()
    
    # SPECIAL HANDLING FOR "Turn Up/Turn Down" PATTERN (v4.3.18)
    if "TURN UP/TURN DOWN" in template_upper and "TURN" in product_upper:
        if "TURN UP" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN UP")
        elif "TURN DOWN" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN DOWN")
    
    # SPECIAL HANDLING FOR STIIIZY STRAIN PATTERNS (v4.3.7)
    if 'STIIIZY' in product_upper and 'STIIIZY' in template_upper and 'STRAIN' in template_upper:
        bag_types = ['BLACK BAG', 'WHITE BAG', 'BLUE BAG', 'GOLD BAG', 'SILVER BAG', 'PURPLE BAG']
        
        for bag_type in bag_types:
            if bag_type in product_upper and bag_type in template_upper:
                product_after_brand = product_upper.split('STIIIZY -')[-1].strip()
                template_after_brand = template_upper.split('STIIIZY -')[-1].strip()
                
                if template_after_brand.startswith('STRAIN ' + bag_type):
                    weight_pattern = r'(\d+\.?\d*G)'
                    product_weight = re.search(weight_pattern, product_after_brand)
                    template_weight = re.search(weight_pattern, template_after_brand)
                    
                    if product_weight and template_weight:
                        if product_weight.group(1) == template_weight.group(1):
                            if bag_type in product_after_brand:
                                return True
    
    # STANDARD PLACEHOLDER MATCHING
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

def match_wildcard_template(item_text, template, wildcards=['COLOR', 'STRAIN', 'FLAVOR']):
    """
    Match item against template with wildcards (COLOR, STRAIN, FLAVOR)
    Returns (match_found, extracted_values) tuple
    
    Improved in v4.3.18 to handle complex patterns like "Turn Up/Turn Down"
    """
    if pd.isna(item_text) or pd.isna(template):
        return False, {}
    
    item_str = str(item_text).strip()
    template_str = str(template).strip()
    
    # Find all wildcard positions in template
    wildcard_positions = {}
    for wildcard in wildcards:
        if wildcard in template_str:
            wildcard_positions[wildcard] = template_str.find(wildcard)
    
    if not wildcard_positions:
        return False, {}
    
    # Create regex pattern from template
    pattern = re.escape(template_str)
    
    # Replace escaped wildcards with capture groups
    for wildcard in wildcards:
        escaped_wildcard = re.escape(wildcard)
        if escaped_wildcard in pattern:
            pattern = pattern.replace(escaped_wildcard, r'([^,]+?)', 1)
    
    # Try to match
    match = re.match(pattern + r'\s*$', item_str, re.IGNORECASE)
    
    if match:
        extracted_values = {}
        wildcard_list = sorted(wildcard_positions.items(), key=lambda x: x[1])
        for i, (wildcard, _) in enumerate(wildcard_list, 1):
            if i <= len(match.groups()):
                extracted_values[wildcard] = match.group(i).strip()
        return True, extracted_values
    
    return False, {}

def match_apparel_products(row, templates):
    """
    Advanced matching for apparel products by stripping size suffixes
    
    Apparel products in Company Products have sizes in parentheses:
    - "Haven - California Shirt (XS)" → matches "Haven - California Shirt"
    - "Haven - Hoodie (Large)" → matches "Haven - Hoodie"
    """
    company_item = row.get('Product', row.get('Item', ''))
    match_steps = []
    
    # Strip size from company item (anything in parentheses at the end)
    size_pattern = r'\s*\([^)]+\)\s*$'
    company_item_no_size = re.sub(size_pattern, '', str(company_item)).strip()
    
    if company_item_no_size != str(company_item).strip():
        size_match = re.search(r'\(([^)]+)\)$', str(company_item))
        if size_match:
            extracted_size = size_match.group(1)
            match_steps.append(f"size: {extracted_size}")
    
    matched_template = None
    for template in templates:
        if company_item_no_size.lower() == str(template).lower():
            matched_template = template
            match_steps.append("exact match (without size)")
            break
    
    if not matched_template and len(templates) == 1:
        template = templates[0]
        if company_item_no_size.lower() in str(template).lower() or str(template).lower() in company_item_no_size.lower():
            matched_template = template
            match_steps.append("partial match (single template)")
    
    return (matched_template, match_steps) if matched_template else (None, [])

# ============================================================================
# CATEGORY-SPECIFIC MATCHING - ADAPTED FROM PRICE CHECKER v4.3.3
# ============================================================================

def _match_flower(row, templates):
    """
    Advanced matching for flower products using weight and keywords
    Adapted from Price Checker match_flower_products() for TMC column naming
    """
    current_templates = list(templates)
    match_steps = []
    
    # Step 1: Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight and len(current_templates) > 1:
        weight_matched = [t for t in current_templates if extract_weight_from_item(t) == company_weight]
        if weight_matched:
            current_templates = weight_matched
            match_steps.append(f"weight: {company_weight}")
    
    # Step 2: Filter by keywords if still multiple options
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_kw_list = [kw.strip() for kw in str(company_keywords).split(',')]
        
        template_scores = []
        for template in current_templates:
            catalog_keywords = extract_category_keywords(template, 'Flower')
            if catalog_keywords:
                catalog_kw_list = [kw.strip() for kw in catalog_keywords.split(',')]
                matches = sum(1 for ck in company_kw_list if ck in catalog_kw_list)
                template_scores.append((template, matches, len(catalog_kw_list), catalog_kw_list))
            else:
                template_scores.append((template, 0, 0, []))
        
        max_score = max(score for _, score, _, _ in template_scores)
        if max_score > 0:
            best = [(t, s, tk, kl) for t, s, tk, kl in template_scores if s == max_score]
            
            if len(best) == 1:
                current_templates = [best[0][0]]
                matched_kws = [ck for ck in company_kw_list if ck in best[0][3]]
                match_steps.append(f"keywords: {', '.join(matched_kws)}")
            else:
                # Tiebreaker: prefer fewer total keywords (more specific template)
                min_total = min(tk for _, _, tk, _ in best)
                final = [t for t, _, tk, _ in best if tk == min_total]
                if len(final) == 1:
                    current_templates = final
                    winner_kws = [kl for t, _, _, kl in best if t == final[0]][0]
                    matched_kws = [ck for ck in company_kw_list if ck in winner_kws]
                    match_steps.append(f"keywords: {', '.join(matched_kws)} (tiebreaker)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

def _match_preroll(row, templates):
    """
    Advanced matching for preroll products using infused status, pack size, weight, and keywords
    Adapted from Price Checker match_preroll_products() for TMC column naming
    """
    # Step 1: Filter by infused status
    company_has_infused = 'infused' in str(row.get('Item', row.get('Product', ''))).lower()
    
    infused_filtered = [t for t in templates if ('infused' in str(t).lower()) == company_has_infused]
    current_templates = infused_filtered if infused_filtered else list(templates)
    match_steps = []
    if infused_filtered:
        match_steps.append(f"infused: {'yes' if company_has_infused else 'no'}")
    
    # Step 2: Filter by pack size FIRST (very distinctive for prerolls)
    company_pack = row.get('Extracted_Pack_Size')
    if company_pack and len(current_templates) > 1:
        pack_matched = [t for t in current_templates if extract_pack_size_from_item(t) == company_pack]
        if pack_matched:
            current_templates = pack_matched
            match_steps.append(f"pack: {company_pack}")
    
    # Step 3: Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight and len(current_templates) > 1:
        weight_matched = [t for t in current_templates if extract_weight_from_item(t) == company_weight]
        if weight_matched:
            current_templates = weight_matched
            match_steps.append(f"weight: {company_weight}")
    
    # Step 4: If no pack size in product but still multiple templates, prefer templates without pack
    if not company_pack and len(current_templates) > 1:
        no_pack = [t for t in current_templates if not extract_pack_size_from_item(t)]
        if len(no_pack) == 1:
            current_templates = no_pack
            match_steps.append("no pack (fallback)")
    
    # Step 5: Filter by type keywords (excluding 'infused')
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_kw_list = [kw.strip() for kw in str(company_keywords).split(',')]
        company_type_kws = [kw for kw in company_kw_list if kw != 'infused']
        
        if company_type_kws:
            template_scores = []
            for template in current_templates:
                catalog_keywords = extract_category_keywords(template, 'Preroll')
                if catalog_keywords:
                    catalog_kw_list = [kw.strip() for kw in catalog_keywords.split(',')]
                    catalog_type_kws = [kw for kw in catalog_kw_list if kw != 'infused']
                    matches = sum(1 for ck in company_type_kws if ck in catalog_type_kws)
                    template_scores.append((template, matches, len(catalog_type_kws), catalog_type_kws))
                else:
                    template_scores.append((template, 0, 0, []))
            
            max_score = max(score for _, score, _, _ in template_scores)
            if max_score > 0:
                best = [(t, s, tk, kl) for t, s, tk, kl in template_scores if s == max_score]
                
                if len(best) == 1:
                    current_templates = [best[0][0]]
                    matched_kws = [ck for ck in company_type_kws if ck in best[0][3]]
                    match_steps.append(f"type: {', '.join(matched_kws)}")
                else:
                    min_total = min(tk for _, _, tk, _ in best)
                    final = [t for t, _, tk, _ in best if tk == min_total]
                    if len(final) == 1:
                        current_templates = final
                        winner_kws = [kl for t, _, _, kl in best if t == final[0]][0]
                        matched_kws = [ck for ck in company_type_kws if ck in winner_kws]
                        match_steps.append(f"type: {', '.join(matched_kws)} (tiebreaker)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

def _match_vape_extract(row, templates, category):
    """
    Advanced matching for vape and extract products using weight and keywords
    Adapted from Price Checker match_vape_extract_products() for TMC column naming
    """
    current_templates = list(templates)
    match_steps = []
    
    # Step 1: Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight and len(current_templates) > 1:
        weight_matched = [t for t in current_templates if extract_weight_from_item(t) == company_weight]
        if weight_matched:
            current_templates = weight_matched
            match_steps.append(f"weight: {company_weight}")
    
    # Step 2: Filter by keywords
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_kw_list = [kw.strip() for kw in str(company_keywords).split(',')]
        
        template_scores = []
        for template in current_templates:
            catalog_keywords = extract_category_keywords(template, category)
            if catalog_keywords:
                catalog_kw_list = [kw.strip() for kw in catalog_keywords.split(',')]
                matches = sum(1 for ck in company_kw_list if ck in catalog_kw_list)
                template_scores.append((template, matches, len(catalog_kw_list), catalog_kw_list))
            else:
                template_scores.append((template, 0, 0, []))
        
        max_score = max(score for _, score, _, _ in template_scores)
        if max_score > 0:
            best = [(t, s, tk, kl) for t, s, tk, kl in template_scores if s == max_score]
            
            if len(best) == 1:
                current_templates = [best[0][0]]
                matched_kws = [ck for ck in company_kw_list if ck in best[0][3]]
                match_steps.append(f"keywords: {', '.join(matched_kws)}")
            else:
                min_total = min(tk for _, _, tk, _ in best)
                final = [t for t, _, tk, _ in best if tk == min_total]
                if len(final) == 1:
                    current_templates = final
                    winner_kws = [kl for t, _, _, kl in best if t == final[0]][0]
                    matched_kws = [ck for ck in company_kw_list if ck in winner_kws]
                    match_steps.append(f"keywords: {', '.join(matched_kws)} (tiebreaker)")
    elif not company_keywords and len(current_templates) > 1:
        # Fallback: prefer templates without keywords (generic template)
        no_kw = [t for t in current_templates if not extract_category_keywords(t, category)]
        if len(no_kw) == 1:
            current_templates = no_kw
            match_steps.append("no keywords (fallback)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

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
# PROFILE TEMPLATE MATCHING - ADAPTED FROM PRICE CHECKER v4.3.22
# ============================================================================

def normalize_category(category):
    """Normalize category names to match Product Catalog"""
    if pd.isna(category):
        return 'Unknown'
    
    category_str = str(category).strip()
    
    flower_mapping = {
        'Flower (Indica)': 'Flower',
        'Flower (Sativa)': 'Flower', 
        'Flower (Hybrid)': 'Flower'
    }
    
    return flower_mapping.get(category_str, category_str)

def normalize_product_names(sales_df, product_col='Product'):
    """
    Normalize product names to fix apostrophe stripping from Blaze POS exports.
    
    Some Blaze POS data strips apostrophes from product names but NOT from brand names:
      Product: "Dr. Norm s - Hybrid THC Max Tablets 20ct 1000mg"  
      Brand:   "Dr. Norm's"
    
    This causes exact matching to fail because "Dr. Norm s" != "Dr. Norm's".
    
    Fix: For each product where the Brand contains an apostrophe, check if the
    Product name has the brand name with the apostrophe stripped. If so, replace
    the broken prefix with the correct brand name.
    
    Affected brands typically include: Dr. Norm's, Uncle Arnie's, Not Your Father's
    
    Args:
        sales_df: DataFrame with Product and Brand columns
        product_col: Name of the product column (default: 'Product')
    
    Returns:
        DataFrame with normalized product names
    """
    if 'Brand' not in sales_df.columns or product_col not in sales_df.columns:
        return sales_df
    
    # Find brands that contain apostrophes
    all_brands = sales_df['Brand'].dropna().unique()
    apostrophe_brands = [b for b in all_brands if "'" in str(b)]
    
    if not apostrophe_brands:
        return sales_df
    
    fixed_count = 0
    fixed_brands = {}
    
    for brand in apostrophe_brands:
        # Create the "broken" version (apostrophe stripped)
        broken_brand = str(brand).replace("'", " ").replace("  ", " ")
        
        # Also try version where apostrophe just disappears (no space)
        broken_brand_nospace = str(brand).replace("'", "")
        
        # Find products with this brand that have the broken name
        brand_mask = sales_df['Brand'] == brand
        
        for idx in sales_df[brand_mask].index:
            product = str(sales_df.at[idx, product_col])
            
            # Check if product starts with the broken brand name
            if product.startswith(broken_brand) and not product.startswith(str(brand)):
                # Replace the broken prefix with the correct brand name
                fixed_product = str(brand) + product[len(broken_brand):]
                sales_df.at[idx, product_col] = fixed_product
                fixed_count += 1
                fixed_brands[brand] = fixed_brands.get(brand, 0) + 1
            elif product.startswith(broken_brand_nospace) and not product.startswith(str(brand)):
                fixed_product = str(brand) + product[len(broken_brand_nospace):]
                sales_df.at[idx, product_col] = fixed_product
                fixed_count += 1
                fixed_brands[brand] = fixed_brands.get(brand, 0) + 1
    
    if fixed_count > 0:
        st.info(f"🔤 Fixed {fixed_count} product names with stripped apostrophes (brand prefix):")
        for brand, count in sorted(fixed_brands.items()):
            st.write(f"  • **{brand}**: {count} products normalized")
    
    # ---- Phase 2: Broad regex fix for ALL remaining orphaned " s " patterns ----
    # Catches non-brand apostrophes like "40 s Mini" → "40's Mini",
    # "Mother s Milk" → "Mother's Milk", "Women s Tee" → "Women's Tee",
    # and brand-prefix cases missed due to wrong Brand column
    # (e.g. Brand="Juicy Jay Papers" but Product="Juicy Jay s - ...")
    orphan_pattern = re.compile(r'(\w+) s\b')
    phase2_count = 0
    
    for idx in sales_df.index:
        product = str(sales_df.at[idx, product_col])
        if orphan_pattern.search(product):
            fixed_product = orphan_pattern.sub(r"\1's", product)
            if fixed_product != product:
                sales_df.at[idx, product_col] = fixed_product
                phase2_count += 1
    
    if phase2_count > 0:
        st.info(f"🔤 Fixed {phase2_count} additional product names with orphaned apostrophes (general pattern)")
    
    return sales_df

def add_profile_template_matching(sales_df, catalog_df, progress_callback=None):
    """
    Match sales products to Profile Templates using Price Checker's smart matching engine
    
    v1.3.0 FIXES:
    - Strategy 2: Now category-aware with specificity scoring (v4.3.21 fix)
    - Strategy 2.5: NEW - Wildcard matching with category-awareness
    - Strategy 3/4: Now respect EXACT_PRODUCT_MATCH_BRANDS (skip_auto_matching)
    - Strategy 5: Full category-specific matching (flower/preroll/vape/extract/apparel)
    
    Column mapping:
    - Blaze 'Product' → Price Checker 'Item'
    - Blaze 'Product Category' → Price Checker 'Category'
    """
    if catalog_df is None or catalog_df.empty:
        sales_df['Profile_Template'] = None
        sales_df['Match_Type'] = None
        return sales_df
    
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
    product_col = 'Product'
    category_col = 'Product Category' if 'Product Category' in sales_df.columns else 'Category'
    
    # Normalize categories in sales data
    sales_df['_Normalized_Category'] = sales_df[category_col].apply(normalize_category)
    
    # Build brand and category mappings from catalog (with deduplication)
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
    
    # Filter out exact match brands from auto-matching
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
    wildcard_matches = 0
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
        
        # =====================================================================
        # Strategy 1: Exact match (searches all brand templates)
        # =====================================================================
        if brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if str(product).lower() == str(template).lower():
                    sales_df.at[idx, 'Profile_Template'] = template
                    sales_df.at[idx, 'Match_Type'] = 'exact'
                    exact_matches += 1
                    match_found = True
                    break
        
        # =====================================================================
        # Strategy 2: Placeholder pattern match - CATEGORY-AWARE + specificity scoring
        # v1.3.0 FIX: Was using brand_catalog_map (all templates), now uses
        # brand_category_catalog_map (category-specific) to prevent cross-category mismatches
        # Also picks MOST SPECIFIC match instead of first match
        # =====================================================================
        if not match_found:
            brand_category_key = f"{brand}|{category}"
            search_templates = brand_category_catalog_map.get(brand_category_key, [])
            
            if search_templates:
                placeholder_candidates = []
                
                for template in search_templates:
                    if match_placeholder_pattern(product, template):
                        # Score by specificity: more matching non-placeholder words = higher score
                        product_words = set(str(product).lower().split())
                        template_words = set(str(template).lower().split())
                        placeholder_keywords = {'strain', 'color', 'flavor', 'size', 'variant'}
                        template_meaningful_words = template_words - placeholder_keywords
                        specificity_score = len(template_meaningful_words & product_words)
                        
                        placeholder_candidates.append({
                            'template': template,
                            'score': specificity_score
                        })
                
                if placeholder_candidates:
                    # Pick the MOST SPECIFIC match
                    placeholder_candidates.sort(key=lambda x: x['score'], reverse=True)
                    best_match = placeholder_candidates[0]
                    
                    sales_df.at[idx, 'Profile_Template'] = best_match['template']
                    sales_df.at[idx, 'Match_Type'] = 'placeholder'
                    placeholder_matches += 1
                    match_found = True
        
        # =====================================================================
        # Strategy 2.5: Wildcard matching - CATEGORY-AWARE + specificity scoring
        # v1.3.0 NEW: match_wildcard_template() was defined but never called
        # =====================================================================
        if not match_found:
            brand_category_key = f"{brand}|{category}"
            search_templates = brand_category_catalog_map.get(brand_category_key, [])
            
            if search_templates:
                wildcard_candidates = []
                
                for template in search_templates:
                    is_match, extracted_vals = match_wildcard_template(product, template)
                    if is_match:
                        product_words = set(str(product).lower().split())
                        template_words = set(str(template).lower().split())
                        placeholder_keywords = {'strain', 'color', 'flavor', 'size', 'variant'}
                        template_meaningful_words = template_words - placeholder_keywords
                        specificity_score = len(template_meaningful_words & product_words)
                        
                        wildcard_candidates.append({
                            'template': template,
                            'wildcards': extracted_vals,
                            'score': specificity_score
                        })
                
                if wildcard_candidates:
                    wildcard_candidates.sort(key=lambda x: x['score'], reverse=True)
                    best_match = wildcard_candidates[0]
                    
                    sales_df.at[idx, 'Profile_Template'] = best_match['template']
                    sales_df.at[idx, 'Match_Type'] = 'wildcard'
                    sales_df.at[idx, 'Match_Keywords'] = ', '.join(
                        [f"{k}={v}" for k, v in best_match['wildcards'].items()])
                    wildcard_matches += 1
                    match_found = True
        
        # Skip auto-matching for exact product match brands
        skip_auto_matching = brand in EXACT_PRODUCT_MATCH_BRANDS
        
        # =====================================================================
        # Strategy 3: Single entry brand auto-match
        # v1.3.0 FIX: Now respects skip_auto_matching
        # =====================================================================
        if not match_found and not skip_auto_matching and brand in filtered_single_entry_brands:
            template = filtered_single_entry_brands[brand]
            sales_df.at[idx, 'Profile_Template'] = template
            sales_df.at[idx, 'Match_Type'] = 'brand_auto'
            single_entry_matches += 1
            match_found = True
        
        # =====================================================================
        # Strategy 4: Brand+category auto-match
        # v1.3.0 FIX: Now respects skip_auto_matching
        # =====================================================================
        if not match_found and not skip_auto_matching:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in filtered_single_entry_brand_categories:
                template = filtered_single_entry_brand_categories[brand_category_key]
                sales_df.at[idx, 'Profile_Template'] = template
                sales_df.at[idx, 'Match_Type'] = 'brand_category_auto'
                brand_category_matches += 1
                match_found = True
        
        # =====================================================================
        # Strategy 5: Advanced category-specific matching
        # v1.3.0 FIX: Was bare-bones weight-only, now uses full Price Checker logic:
        # - Flower: weight + quality tier keywords with tiebreakers
        # - Preroll: infused + pack size + weight + type keywords
        # - Vape/Extract: weight + keywords with no-keyword fallback
        # - Apparel: size stripping + exact/partial matching
        # =====================================================================
        if not match_found and category in ['Flower', 'Preroll', 'Vape', 'Extract', 'Apparel']:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in brand_category_catalog_map:
                templates = brand_category_catalog_map[brand_category_key]
                
                matched_template = None
                match_steps = []
                
                # Build a row dict with column names expected by matching functions
                match_row = {
                    'Item': product,
                    'Product': product,
                    'Category': category,
                    'Extracted_Weight': row.get('_Extracted_Weight'),
                    'Extracted_Pack_Size': row.get('_Extracted_Pack_Size'),
                    'Extracted_Category_Keywords': row.get('_Extracted_Keywords'),
                }
                
                if category == 'Flower':
                    matched_template, match_steps = _match_flower(match_row, templates)
                elif category == 'Preroll':
                    matched_template, match_steps = _match_preroll(match_row, templates)
                elif category in ['Vape', 'Extract']:
                    matched_template, match_steps = _match_vape_extract(match_row, templates, category)
                elif category == 'Apparel':
                    matched_template, match_steps = match_apparel_products(row, templates)
                
                if matched_template:
                    sales_df.at[idx, 'Profile_Template'] = matched_template
                    sales_df.at[idx, 'Match_Type'] = f'{category.lower()}_advanced'
                    sales_df.at[idx, 'Match_Keywords'] = ', '.join(match_steps)
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
    total_matched = exact_matches + placeholder_matches + wildcard_matches + single_entry_matches + brand_category_matches + advanced_matches
    
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
    
    # Step 3: Normalize product names (fix apostrophe stripping)
    update_progress(0.52, "Normalizing product names...")
    sales_df = normalize_product_names(sales_df, product_col='Product')
    
    # Step 4: Match Profile Templates
    update_progress(0.55, "Matching Profile Templates...")
    
    sales_df = add_profile_template_matching(
        sales_df, catalog_df,
        lambda p, t: update_progress(0.55 + p * 0.3, t)
    )
    
    # Step 5: Calculate margins
    update_progress(0.9, "Calculating margins...")
    
    sales_df = calculate_margins(sales_df, private_labels)
    
    # Step 6: Final cleanup
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
    """
    Product-level detail with rich filtering and selection summary.
    
    Filters act as an ad-hoc grouping mechanism: narrow down to any set of
    products and the Selection Summary recalculates combined metrics properly
    (summing raw values, then deriving percentages from totals).
    """
    st.subheader("📋 Product Detail")
    
    # ---- Build product summary ----
    group_cols = ['Product', 'Brand']
    has_category = 'Product Category' in df.columns
    if has_category:
        group_cols.append('Product Category')
    
    agg_dict = {
        'Net Sales': 'sum',
        'Standard_Margin': 'sum',
        'True_Margin': 'sum',
        'Vendor_Pays': 'sum',
        'Haven_Pays': 'sum',
        'Margin_Lift': 'sum',
        'Quantity Sold': 'sum',
        'Profile_Template': 'first'
    }
    
    product_summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Calculate per-product percentages
    product_summary['Std %'] = (product_summary['Standard_Margin'] / product_summary['Net Sales'] * 100).round(1)
    product_summary['True %'] = (product_summary['True_Margin'] / product_summary['Net Sales'] * 100).round(1)
    
    # Add derived columns for filtering
    product_summary['_Weight'] = product_summary['Product'].apply(extract_weight_from_item)
    product_summary['_Pack'] = product_summary['Product'].apply(extract_pack_size_from_item)
    
    product_summary = product_summary.sort_values('Net Sales', ascending=False)
    
    # ---- Filter Controls ----
    st.markdown("**🔍 Filters** — _use these to build ad-hoc product groups; summary recalculates from filtered set_")
    
    # Row 1: Text search + Brand + Category
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    
    with f_col1:
        text_search = st.text_input(
            "Search products:",
            placeholder="e.g. Jefferey 3pk — comma for OR: Jefferey, CUREjoint",
            help="**Space = AND** (all words must match)\n**Comma = OR** (any term matches)\n\nExamples:\n- `Jefferey 3pk` → products with BOTH words\n- `Jefferey, CUREjoint` → products with EITHER word",
            key="pv_text_search"
        )
    
    with f_col2:
        brand_options = sorted(product_summary['Brand'].dropna().unique())
        brand_filter = st.multiselect(
            "Brand:",
            options=brand_options,
            default=[],
            key="pv_brand_filter"
        )
    
    with f_col3:
        if has_category:
            cat_options = sorted(product_summary['Product Category'].dropna().unique())
            category_filter = st.multiselect(
                "Category:",
                options=cat_options,
                default=[],
                key="pv_category_filter"
            )
        else:
            category_filter = []
    
    # Row 2: SKU Type (Template) + Weight + Pack + Match status
    f_col4, f_col5, f_col6, f_col7 = st.columns([3, 1, 1, 1])
    
    with f_col4:
        template_options = sorted(product_summary['Profile_Template'].dropna().unique())
        template_filter = st.multiselect(
            "SKU Type (Template):",
            options=template_options,
            default=[],
            key="pv_template_filter"
        )
    
    with f_col5:
        weight_options = sorted(
            [w for w in product_summary['_Weight'].dropna().unique()],
            key=lambda w: float(re.sub(r'[^\d.]', '', w)) if re.search(r'[\d.]', w) else 0
        )
        weight_filter = st.multiselect(
            "Weight:",
            options=weight_options,
            default=[],
            key="pv_weight_filter"
        )
    
    with f_col6:
        pack_options = sorted(
            [p for p in product_summary['_Pack'].dropna().unique()],
            key=lambda p: int(re.sub(r'[^\d]', '', p)) if re.search(r'\d', p) else 0
        )
        pack_filter = st.multiselect(
            "Pack:",
            options=pack_options,
            default=[],
            key="pv_pack_filter"
        )
    
    with f_col7:
        match_filter = st.selectbox(
            "Match:",
            options=["All", "Matched", "Unmatched"],
            key="pv_match_filter"
        )
    
    # ---- Apply Filters ----
    filtered = product_summary.copy()
    active_filters = []
    
    if text_search:
        # Comma-separated terms = OR logic between groups
        # Space-separated words within a term = AND logic (all words must be present)
        terms = [t.strip() for t in text_search.split(',') if t.strip()]
        if terms:
            mask = pd.Series(False, index=filtered.index)
            for term in terms:
                words = term.split()
                if len(words) > 1:
                    # Multiple words: AND logic - product must contain ALL words
                    word_mask = pd.Series(True, index=filtered.index)
                    for word in words:
                        word_mask = word_mask & filtered['Product'].str.contains(
                            re.escape(word), case=False, na=False)
                    mask = mask | word_mask
                else:
                    # Single word: simple contains
                    mask = mask | filtered['Product'].str.contains(
                        re.escape(term), case=False, na=False)
            filtered = filtered[mask]
            active_filters.append(f"search: '{text_search}'")
    
    if brand_filter:
        filtered = filtered[filtered['Brand'].isin(brand_filter)]
        active_filters.append(f"{len(brand_filter)} brand(s)")
    
    if category_filter:
        filtered = filtered[filtered['Product Category'].isin(category_filter)]
        active_filters.append(f"{len(category_filter)} category(ies)")
    
    if template_filter:
        filtered = filtered[filtered['Profile_Template'].isin(template_filter)]
        active_filters.append(f"{len(template_filter)} template(s)")
    
    if weight_filter:
        filtered = filtered[filtered['_Weight'].isin(weight_filter)]
        active_filters.append(f"weight: {', '.join(weight_filter)}")
    
    if pack_filter:
        filtered = filtered[filtered['_Pack'].isin(pack_filter)]
        active_filters.append(f"pack: {', '.join(pack_filter)}")
    
    if match_filter == "Matched":
        filtered = filtered[filtered['Profile_Template'].notna()]
        active_filters.append("matched only")
    elif match_filter == "Unmatched":
        filtered = filtered[filtered['Profile_Template'].isna()]
        active_filters.append("unmatched only")
    
    # ---- Selection Summary Banner ----
    # Calculate combined metrics from filtered set (proper margin math)
    total_sales = filtered['Net Sales'].sum()
    total_std_margin = filtered['Standard_Margin'].sum()
    total_true_margin = filtered['True_Margin'].sum()
    total_vendor = filtered['Vendor_Pays'].sum()
    total_haven = filtered['Haven_Pays'].sum()
    total_lift = filtered['Margin_Lift'].sum()
    total_qty = filtered['Quantity Sold'].sum()
    
    combined_std_pct = (total_std_margin / total_sales * 100) if total_sales > 0 else 0
    combined_true_pct = (total_true_margin / total_sales * 100) if total_sales > 0 else 0
    
    matched_count = filtered['Profile_Template'].notna().sum()
    unmatched_count = filtered['Profile_Template'].isna().sum()
    
    # Show filter status
    if active_filters:
        st.caption(f"Active filters: {' · '.join(active_filters)}")
    
    # Summary metrics
    s_col1, s_col2, s_col3, s_col4, s_col5, s_col6, s_col7 = st.columns(7)
    with s_col1:
        st.metric("Products", f"{len(filtered):,}")
    with s_col2:
        st.metric("Net Sales", format_currency(total_sales))
    with s_col3:
        st.metric("Std Margin", f"{combined_std_pct:.1f}%", help=format_currency(total_std_margin))
    with s_col4:
        st.metric("True Margin", f"{combined_true_pct:.1f}%", help=format_currency(total_true_margin))
    with s_col5:
        st.metric("Vendor Pays", format_currency(total_vendor))
    with s_col6:
        st.metric("Margin Lift", format_currency(total_lift))
    with s_col7:
        st.metric("Qty Sold", f"{total_qty:,.0f}")
    
    # ---- Display Controls ----
    d_col1, d_col2 = st.columns([1, 5])
    with d_col1:
        all_count = len(filtered)
        count_options = [x for x in [50, 100, 250, 500] if x < all_count] + [all_count] if all_count > 0 else [0]
        display_count = st.selectbox(
            "Show:",
            options=count_options,
            index=min(1, len(count_options) - 1),
            format_func=lambda x: f"All ({x:,})" if x == all_count else f"Top {x}",
            key="pv_display_count"
        )
    with d_col2:
        st.caption(f"{matched_count:,} matched · {unmatched_count:,} unmatched")
    
    # ---- Data Table ----
    if len(filtered) > 0:
        display_cols = ['Brand', 'Product']
        if has_category:
            display_cols.append('Product Category')
        display_cols += ['Net Sales', 'Quantity Sold', 'Std %', 'True %',
                         'Vendor_Pays', 'Haven_Pays', 'Margin_Lift', 'Profile_Template']
        
        existing_cols = [c for c in display_cols if c in filtered.columns]
        display_df = filtered[existing_cols].head(display_count).copy()
        
        for col in ['Net Sales', 'Vendor_Pays', 'Haven_Pays', 'Margin_Lift']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_currency)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No products match the current filters.")
    
    # ---- Downloads ----
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        # Download filtered data
        if len(filtered) > 0:
            filt_csv = io.StringIO()
            filtered.drop(columns=['_Weight', '_Pack'], errors='ignore').to_csv(filt_csv, index=False)
            label = f"📥 Download Filtered ({len(filtered):,} products)" if active_filters else "📥 Download All Product Data"
            st.download_button(label, filt_csv.getvalue(),
                               "product_true_margins_filtered.csv", "text/csv",
                               key="pv_download_filtered")
    
    with dl_col2:
        # Download full data (always available)
        if active_filters and len(product_summary) > 0:
            full_csv = io.StringIO()
            product_summary.drop(columns=['_Weight', '_Pack'], errors='ignore').to_csv(full_csv, index=False)
            st.download_button(f"📥 Download All ({len(product_summary):,} products)", full_csv.getvalue(),
                               "product_true_margins_all.csv", "text/csv",
                               key="pv_download_all")

def render_debug_view(df):
    """Debug/Diagnostics tab for analyzing matching quality and data issues"""
    st.subheader("🔧 Matching Diagnostics")
    
    # ---- Match Type Breakdown ----
    st.write("**📊 Match Type Distribution**")
    
    if 'Match_Type' in df.columns:
        total_products = df['Product'].nunique()
        
        # Get unique products with their match info
        agg_dict = {
            'Brand': 'first',
            'Profile_Template': 'first',
            'Match_Type': 'first',
            'Net Sales': 'sum',
            'Quantity Sold': 'sum'
        }
        if 'Match_Keywords' in df.columns:
            agg_dict['Match_Keywords'] = 'first'
        product_matches = df.groupby('Product').agg(agg_dict).reset_index()
        
        matched = product_matches[product_matches['Profile_Template'].notna()]
        unmatched = product_matches[product_matches['Profile_Template'].isna()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Products", f"{total_products:,}")
        with col2:
            st.metric("Matched", f"{len(matched):,}", 
                      delta=f"{len(matched)/total_products*100:.1f}%")
        with col3:
            st.metric("Unmatched", f"{len(unmatched):,}",
                      delta=f"-{len(unmatched)/total_products*100:.1f}%")
        
        # Match type breakdown
        if len(matched) > 0:
            type_counts = matched['Match_Type'].value_counts()
            
            type_display = {
                'exact': '🎯 Exact Match',
                'placeholder_pattern': '🔤 Placeholder (STRAIN/COLOR/FLAVOR)',
                'wildcard': '🃏 Wildcard Match',
                'brand_auto': '1️⃣ Single Entry Auto',
                'brand_category_auto': '📂 Brand+Category Auto',
                'flower_advanced': '🌸 Flower (Weight+Keywords)',
                'preroll_advanced': '🚬 Preroll (Pack+Weight+Type)',
                'vape_advanced': '💨 Vape (Weight+Keywords)',
                'extract_advanced': '🧪 Extract (Weight+Keywords)',
                'apparel_advanced': '👕 Apparel (Size Strip)',
            }
            
            st.write("**Strategy Performance:**")
            for match_type, count in type_counts.items():
                display_name = type_display.get(match_type, match_type)
                pct = count / total_products * 100
                st.write(f"  • {display_name}: **{count:,}** ({pct:.1f}%)")
    
    st.markdown("---")
    
    # ---- Unmatched Products Analysis ----
    st.write("**❌ Unmatched Products Analysis**")
    
    product_matches = df.groupby('Product').agg({
        'Brand': 'first',
        'Profile_Template': 'first',
        'Net Sales': 'sum',
        'Quantity Sold': 'sum'
    }).reset_index()
    
    if 'Product Category' in df.columns:
        cat_agg = df.groupby('Product')['Product Category'].first().reset_index()
        product_matches = product_matches.merge(cat_agg, on='Product', how='left')
    
    unmatched = product_matches[product_matches['Profile_Template'].isna()].sort_values(
        'Net Sales', ascending=False
    )
    
    if len(unmatched) > 0:
        # Unmatched by brand
        unmatched_by_brand = unmatched.groupby('Brand').agg({
            'Product': 'count',
            'Net Sales': 'sum'
        }).rename(columns={'Product': 'Unmatched_Count'}).sort_values('Net Sales', ascending=False)
        
        st.write(f"**Top unmatched brands** ({len(unmatched_by_brand)} brands with unmatched products):")
        
        display_brand_df = unmatched_by_brand.head(20).copy()
        display_brand_df['Net Sales'] = display_brand_df['Net Sales'].apply(format_currency)
        st.dataframe(display_brand_df, use_container_width=True)
        
        # Top unmatched products by revenue
        st.write("**Top unmatched products by revenue:**")
        
        unmatched_display_cols = ['Brand', 'Product', 'Net Sales', 'Quantity Sold']
        if 'Product Category' in unmatched.columns:
            unmatched_display_cols.insert(2, 'Product Category')
        
        top_unmatched = unmatched[unmatched_display_cols].head(50).copy()
        top_unmatched['Net Sales'] = top_unmatched['Net Sales'].apply(format_currency)
        st.dataframe(top_unmatched, use_container_width=True, hide_index=True)
        
        # Download unmatched products
        st.write("**📥 Export Unmatched Products:**")
        
        unmatched_export = unmatched.copy()
        unmatched_csv = io.StringIO()
        unmatched_export.to_csv(unmatched_csv, index=False)
        st.download_button(
            f"📥 Download {len(unmatched):,} Unmatched Products",
            unmatched_csv.getvalue(),
            "unmatched_products.csv",
            "text/csv"
        )
    else:
        st.success("🎉 All products matched!")
    
    st.markdown("---")
    
    # ---- Data Quality Checks ----
    st.write("**🔍 Data Quality Checks**")
    
    # Check for apostrophe issues
    if 'Brand' in df.columns and 'Product' in df.columns:
        apostrophe_brands = [b for b in df['Brand'].dropna().unique() if "'" in str(b)]
        if apostrophe_brands:
            st.write(f"Brands with apostrophes: {', '.join(sorted(apostrophe_brands))}")
            
            for brand in apostrophe_brands:
                broken_brand = str(brand).replace("'", " ").replace("  ", " ")
                brand_products = df[df['Brand'] == brand]['Product'].unique()
                still_broken = [p for p in brand_products if str(p).startswith(broken_brand) and not str(p).startswith(str(brand))]
                if still_broken:
                    st.warning(f"⚠️ {brand}: {len(still_broken)} products still have stripped apostrophes")
                    for p in still_broken[:5]:
                        st.write(f"  • `{p}`")
    
    # Matching engine info
    st.markdown("---")
    st.write("**ℹ️ Matching Engine Info**")
    st.write(f"• Engine: Price Checker v4.3.22 (adapted)")
    st.write(f"• TMC Version: v{VERSION}")
    st.write(f"• Strategies: Exact → Placeholder → Wildcard → Single Entry → Brand+Category → Advanced (Flower/Preroll/Vape/Extract/Apparel)")
    st.write(f"• EXACT_PRODUCT_MATCH_BRANDS: {len(EXACT_PRODUCT_MATCH_BRANDS)} brands (exact-only matching)")

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
        **v1.3.2** (Current - 2026-02-04)
        - 🔍 Product Detail: rich filtering (search, brand, category, template, weight, pack)
        - 📊 Selection Summary: combined True Margin % from filtered set
        - 🔤 Broad apostrophe fix: catches all orphaned " s " patterns (40's, Mother's, etc.)
        - 💡 Search: help text clarifies space=AND, comma=OR
        - 📥 Filtered + full download options
        
        **v1.3.1** (2026-02-04)
        - 🔤 Apostrophe normalization (Dr. Norm's, Uncle Arnie's, etc.)
        - 📋 Product Detail: configurable display + match/brand filters
        - 🔧 Debug tab: unmatched analysis + match diagnostics + export
        
        **v1.3.0** (Current - 2026-02-04)
        - 🔧 CRITICAL: Placeholder matching now category-aware
        - 🔧 CRITICAL: Added placeholder specificity scoring
        - ✨ NEW: Wildcard matching (Strategy 2.5)
        - 🔧 CRITICAL: Full advanced matching (flower/preroll/vape/extract/apparel)
        - 🔧 FIX: skip_auto_matching for EXACT_PRODUCT_MATCH_BRANDS
        
        **v1.2.1** (2026-01-28)
        - 🧠 Matching engine to {MATCHING_ENGINE_VERSION}
        - Apparel matching, Turn Up/Turn Down handling
        
        **v1.2.0** (2026-01-14)
        - 🔧 Load Data button, Profile Template matching
        - 📁 Multi-file upload for Sales Detail
        
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
            
            st.session_state['processed_data'] = processed_df
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Processed {len(processed_df):,} products successfully!")
            
            st.rerun()
    
    # =====================
    # MAIN CONTENT
    # =====================
    
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
            ### v1.3.0 Features
            
            **🧠 Full Matching Engine (v4.3.22)**
            - Category-aware placeholder matching
            - Specificity scoring (picks best match)
            - Wildcard matching (COLOR, STRAIN, FLAVOR)
            - Full advanced matching:
              - Flower: weight + quality keywords
              - Preroll: infused + pack + weight + type
              - Vape/Extract: weight + keywords
              - Apparel: size stripping
            
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
    
    # Tabs
    tab_names = ["📊 Overview", "🏷️ Brands", "🏪 Shops", "📂 Categories", "📦 SKU Types", "📋 Products", "🔧 Debug"]
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
    
    with tabs[6]:
        render_debug_view(filtered_df)

if __name__ == "__main__":
    main()