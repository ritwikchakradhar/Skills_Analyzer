import streamlit as st
import pandas as pd
import re
from datetime import datetime
import base64
from typing import Dict, List, Optional
from fuzzywuzzy import process

# Page config
st.set_page_config(
    page_title="Skills Analyzer",
    page_icon="📊",
    layout="wide"
)

# Skill variations for matching
st.session_state.skill_variations = {
    'nodejs': ['node', 'nodejs', 'node.js', 'node-js', 'node js'],
    'python': ['python', 'python3', 'python 3', 'python(django)', 'python automation', 'python security automation'],
    'java': ['java', 'core java', 'java se'],
    'react': ['react', 'reactjs', 'react.js', 'react js', 'react hooks'],
    'typescript': ['typescript', 'ts'],
    'javascript': ['javascript', 'js', 'es6'],
}

# Helper Functions
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters, spaces, and formatting."""
    df.columns = (
        df.columns
        .str.replace(r'^="', '', regex=True)  # Remove leading ="
        .str.replace(r'"$', '', regex=True)  # Remove trailing "
        .str.strip()
        .str.lower()
        .str.replace(' ', '')
        .str.replace('_', '')
    )
    return df

def map_columns(df: pd.DataFrame, required_columns: Dict[str, List[str]]) -> Dict[str, str]:
    """Map required columns dynamically based on variations."""
    cleaned_columns = clean_column_names(df).columns
    column_mapping = {}

    for standard_name, variations in required_columns.items():
        for col in cleaned_columns:
            if any(variation == col for variation in variations):
                column_mapping[standard_name] = col
                break
        if standard_name not in column_mapping:
            st.error(f"❌ Missing required column for '{standard_name}'. Please check your data.")
            st.stop()

    return column_mapping

def extract_skill_score(skills_str: str, variations: List[str]) -> Optional[float]:
    """Extract skill score for variations."""
    if not skills_str:
        return None
    skills_str = str(skills_str).lower()
    max_score = None
    for variation in variations:
        matches = re.findall(fr'{variation}\s*-\s*(\d+\.?\d*)%', skills_str, re.IGNORECASE)
        for score in matches:
            score = float(score)
            if max_score is None or score > max_score:
                max_score = score
    return max_score

def analyze_skills(trainers_df, managers_df, selected_skills, min_score):
    """Analyze trainers against skills and scores."""
    primary_col = 'primary_skills'
    secondary_col = 'secondary_skills'

    trainers_df[primary_col] = trainers_df[primary_col].fillna('')
    trainers_df[secondary_col] = trainers_df[secondary_col].fillna('')

    skill_scores = {}
    qualified_mask = pd.Series(True, index=trainers_df.index)
    for skill in selected_skills:
        variations = st.session_state.skill_variations.get(skill, [skill])
        primary_scores = trainers_df[primary_col].apply(lambda x: extract_skill_score(x, variations))
        secondary_scores = trainers_df[secondary_col].apply(lambda x: extract_skill_score(x, variations))
        max_scores = pd.DataFrame({'primary': primary_scores, 'secondary': secondary_scores}).max(axis=1)
        skill_scores[f'{skill}_score'] = max_scores
        qualified_mask &= (max_scores >= min_score)

    qualified_trainers = trainers_df[qualified_mask].copy()
    qualified_trainers = qualified_trainers.assign(**skill_scores)

    # Add manager mapping
    if 'developerturingemail' in managers_df.columns:
        manager_map = managers_df.set_index('developerturingemail')['managerturingemail'].to_dict()
        qualified_trainers['manager turing email'] = qualified_trainers['developerturingemail'].map(manager_map)

    return qualified_trainers

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results</a>'

# Main Function
def main():
    st.title("🎯 Skills Analyzer")

    # Load Static Trainer Data
    try:
        trainers_df = pd.read_csv("Delivery Workforce - Data.csv")  # Ensure this file is in the same directory
        required_columns = {
            'business_line': ['businessline', 'business_line'],
            'primary_skills': ['primaryskills', 'primary_skills'],
            'secondary_skills': ['secondaryskills', 'secondary_skills'],
            'developer_turing_email': ['developerturingemail', 'developer_email'],
        }
        column_mapping = map_columns(trainers_df, required_columns)
        trainers_df = trainers_df.rename(columns=column_mapping)
        st.success("✅ Trainer data loaded and columns mapped successfully!")
    except Exception as e:
        st.error(f"Error loading trainer file: {e}")
        return

    # Upload Managers Data
    st.header("1. Upload Managers Data")
    managers_file = st.file_uploader("Upload Managers CSV", type='csv')
    if managers_file is not None:
        managers_df = pd.read_csv(managers_file)
        managers_df = clean_column_names(managers_df)
        st.success("✅ Managers file loaded successfully!")

    # Configuration
    st.header("2. Configure Analysis")
    if managers_file:
        skills = list(st.session_state.skill_variations.keys())
        selected_skills = st.multiselect("Select Skills", options=skills, default=['python', 'nodejs'])
        min_score = st.slider("Minimum Skill Score (%)", 0, 100, 70, step=5)

        # Run Analysis
        if st.button("🔍 Analyze Skills"):
            with st.spinner("Analyzing data..."):
                results = analyze_skills(trainers_df, managers_df, selected_skills, min_score)

            # Display Results
            st.header("3. Results")
            if not results.empty:
                st.dataframe(results)
                filename = f"qualified_trainers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(get_download_link(results, filename), unsafe_allow_html=True)
            else:
                st.warning("No trainers matched the criteria.")

if __name__ == "__main__":
    main()
