import streamlit as st
import pandas as pd
import re
from datetime import datetime
import base64
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="Skills Analyzer",
    page_icon="📊",
    layout="wide"
)

# URL to your trainers data in GitHub
TRAINERS_DATA_URL = "https://raw.githubusercontent.com/ritwikchakradhar/Skills_Analyzer/refs/heads/main/Current%20delivery%20workforce%20-%20Raw%20Data.csv"

# Initialize session state for skill variations
if 'skill_variations' not in st.session_state:
    st.session_state.skill_variations = {
        'nodejs': ['node', 'nodejs', 'node.js', 'node-js', 'node js'],
        'python': ['python', 'python3', 'python 3'],
        'java': ['java', 'core java', 'java se'],
        'kotlin': ['kotlin', 'kotlin-android', 'kotlin android'],
        'react': ['react', 'reactjs', 'react.js', 'react js'],
        'angular': ['angular', 'angularjs', 'angular.js', 'angular js']
    }

@st.cache_data
def load_trainers_data():
    """Load trainers data from GitHub."""
    try:
        df = pd.read_csv(TRAINERS_DATA_URL)
        return df
    except Exception as e:
        st.error(f"Error loading trainers data: {str(e)}")
        return None

def validate_email(email: str) -> bool:
    """Validate email format."""
    if pd.isna(email) or email == 'N/A' or email == '':
        return False
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(email)))

# [Rest of your helper functions remain the same]

def main():
    st.title("Skills Analyzer")
    
    st.write("""
    Upload the managers data file to analyze skills and find qualified trainers.
    Trainers must meet the minimum score threshold in ALL selected skills to qualify.
    """)
    
    # Load trainers data
    trainers_df = load_trainers_data()
    if trainers_df is None:
        st.error("Failed to load trainers data. Please try again later.")
        return
    
    # File upload section - only for managers file
    st.header("1. Upload Managers Data")
    
    st.info("""
    **Required File Format:**
    - Managers CSV must have: 'Developer turing email', 'Manager Turing Email' columns
    """)
    
    managers_file = st.file_uploader("Upload Managers CSV", type='csv')
    
    if managers_file:
        try:
            # Read managers file
            managers_df = pd.read_csv(managers_file)
            
            # Validate required columns
            required_manager_cols = ['Developer turing email', 'Manager Turing Email']
            
            if not all(col in managers_df.columns for col in required_manager_cols):
                st.error("Managers file missing required columns!")
                return
            
            st.success("Managers file uploaded successfully!")
            
            # Configuration section
            st.header("2. Configure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                business_lines = sorted(trainers_df['Business line'].unique().tolist())
                selected_lines = st.multiselect(
                    "Select Business Lines",
                    options=business_lines,
                    default=['LLM', 'Services'] if 'LLM' in business_lines else None
                )
                
                minimum_score = st.slider(
                    "Minimum Skill Score Required",
                    min_value=0,
                    max_value=100,
                    value=70,
                    step=5
                )
            
            with col2:
                available_skills = sorted(st.session_state.skill_variations.keys())
                selected_skills = st.multiselect(
                    "Select Required Skills",
                    options=available_skills,
                    default=['python', 'nodejs'] if 'python' in available_skills else None,
                    help="Trainers must have ALL these skills"
                )
            
            # [Rest of your main function remains the same]

if __name__ == "__main__":
    main()
