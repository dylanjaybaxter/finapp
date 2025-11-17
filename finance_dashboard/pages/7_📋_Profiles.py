"""Profiles Page - Inspect and view profile information."""

from __future__ import annotations

import streamlit as st
import json
import pandas as pd
from pathlib import Path

# Import handling for Streamlit pages
import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
parent_dir = PathLib(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from shared_sidebar import render_shared_sidebar
from profile_manager import get_registry
import db


def main():
    """Render the Profiles inspection page."""
    st.set_page_config(page_title="Profiles", page_icon="📋", layout="wide")
    
    # Render shared sidebar
    render_shared_sidebar()
    
    st.header("📋 Profile Inspection")
    st.markdown("Inspect profile definitions and see which files were processed with each profile.")
    
    registry = get_registry()
    all_profiles = list(registry._profiles.keys())
    
    # Also include profiles from database that might not be in registry
    db_profiles = db.get_all_profiles()
    all_profiles = sorted(list(set(all_profiles + db_profiles)))
    
    if not all_profiles:
        st.warning("No profiles found.")
        return
    
    # Profile selection
    selected_profile_name = st.selectbox(
        "Select Profile to Inspect",
        options=all_profiles,
        help="Choose a profile to view its properties and associated files"
    )
    
    if selected_profile_name:
        # Try to get profile from registry, otherwise show database info only
        if selected_profile_name in registry._profiles:
            profile = registry._profiles[selected_profile_name]
            _render_profile_details(profile, selected_profile_name)
        else:
            # Profile exists in database but not in registry (maybe deleted from profiles dir)
            st.warning(f"Profile '{selected_profile_name}' found in database but not in profile definitions.")
            _render_profile_from_db_only(selected_profile_name)


def _render_profile_details(profile, profile_name: str):
    """Render detailed information about a profile."""
    st.subheader(f"Profile: {profile_name}")
    
    # Get files processed with this profile
    files = db.get_files_by_profile(profile_name)
    
    # Files section at the top
    st.markdown("### 📁 Files Processed with This Profile")
    if files:
        st.info(f"**{len(files)} file(s)** have been processed using this profile:")
        for filename in sorted(files):
            st.write(f"  • `{filename}`")
    else:
        st.info("No files have been processed with this profile yet.")
    
    st.divider()
    
    # Profile properties
    st.markdown("### 📝 Profile Properties")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        st.write(f"**Name:** {profile.name}")
        st.write(f"**Description:** {profile.description}")
        st.write(f"**Source Path:** `{profile.source_path}`")
    
    with col2:
        st.markdown("#### Requirements")
        required_fields = profile.required_fields(['Transaction Date', 'Description', 'Amount'])
        st.write("**Required Fields:**")
        for field in required_fields:
            st.write(f"  • {field}")
    
    # Match criteria
    st.markdown("#### Match Criteria")
    match_info = profile.match
    
    if 'filename_patterns' in match_info:
        st.write("**Filename Patterns:**")
        for pattern in match_info['filename_patterns']:
            st.code(pattern)
    
    if 'header_contains' in match_info:
        st.write("**Required Headers:**")
        for header in match_info['header_contains']:
            st.write(f"  • {header}")
    
    # Column mappings
    st.markdown("#### Column Mappings")
    if profile.fields and 'column_mappings' in profile.fields:
        mappings = profile.fields['column_mappings']
        mapping_df = pd.DataFrame([
            {"Normalized Column": norm, "Original Columns": ", ".join(orig)}
            for norm, orig in mappings.items()
        ])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    
    # Category mappings
    st.markdown("#### Category Mappings")
    if profile.fields and 'category_mappings' in profile.fields:
        cat_mappings = profile.fields['category_mappings']
        for field_name, mappings in cat_mappings.items():
            st.write(f"**{field_name} → Category:**")
            for key, value in mappings.items():
                st.write(f"  • `{key}` → `{value}`")
    
    # Category groups
    st.markdown("#### Category Groups")
    if profile.fields and 'category_groups' in profile.fields:
        cat_groups = profile.fields['category_groups']
        for field_name, groups in cat_groups.items():
            st.write(f"**{field_name} Groupings:**")
            for category, variants in groups.items():
                st.write(f"  • **{category}:** {', '.join(variants)}")
    
    # Description keywords
    st.markdown("#### Description Keywords")
    if profile.fields and 'description_keywords' in profile.fields:
        keywords = profile.fields['description_keywords']
        for category, keyword_list in keywords.items():
            st.write(f"**{category}:**")
            st.write(f"  {', '.join(keyword_list)}")
    
    # Transformations
    st.markdown("#### Transformations")
    if profile.transformations:
        st.json(profile.transformations)
    else:
        st.write("No transformations defined.")
    
    # Quality settings
    st.markdown("#### Quality Settings")
    if profile.quality:
        st.json(profile.quality)
    else:
        st.write("No quality settings defined.")
    
    # Raw JSON view
    with st.expander("🔍 View Raw JSON"):
        st.json(profile.raw)


def _render_profile_from_db_only(profile_name: str):
    """Render profile information when profile definition is not available."""
    st.subheader(f"Profile: {profile_name}")
    
    # Get files processed with this profile
    files = db.get_files_by_profile(profile_name)
    
    # Files section at the top
    st.markdown("### 📁 Files Processed with This Profile")
    if files:
        st.info(f"**{len(files)} file(s)** have been processed using this profile:")
        for filename in sorted(files):
            st.write(f"  • `{filename}`")
    else:
        st.info("No files have been processed with this profile yet.")
    
    st.warning("⚠️ Profile definition file not found. This profile was used to process files but the definition is no longer available.")


# Streamlit will execute this file directly
if __name__ == "__main__":
    main()
else:
    main()

