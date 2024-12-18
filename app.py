import streamlit as st
import pandas as pd
import re
from datetime import datetime
import base64
from fuzzywuzzy import process  # For fuzzy matching
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="Skills Analyzer",
    page_icon="📊",
    layout="wide"
)

# Predefined Skill Variations
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

def validate_managers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate managers DataFrame and map columns dynamically."""
    required_columns = {
        'developer turing email': ['developerturingemail'],
        'manager turing email': ['managerturingemail']
    }

    df = clean_column_names(df)

    column_mapping = {}
    for standard_col, variations in required_columns.items():
        for col in df.columns:
            if col in variations:
                column_mapping[col] = standard_col
                break

    if len(column_mapping) < len(required_columns):
        st.error("❌ Missing required columns in the managers file. Please check your upload.")
        st.stop()

    return df.rename(columns=column_mapping)[list(required_columns.keys())]

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

def match_user_skills(user_skill: str, primary_skills: pd.Series, secondary_skills: pd.Series) -> List[str]:
    """Match user-provided skill dynamically in primary and secondary skills."""
    matches = []
    for col in [primary_skills, secondary_skills]:
        col_matches = col.str.contains(user_skill, case=False, na=False)
        matches.extend(col[col_matches].tolist())
    return matches

def analyze_skills(trainers_df, managers_df, selected_skills, user_skill, min_score):
    """Analyze trainers against skills and scores."""
    trainers_df = clean_column_names(trainers_df)
    skill_columns = {
        'primary skills': ['primaryskills', 'primary skills'],
        'secondary skills': ['secondaryskills', 'secondary skills']
    }

    # Match skill-related columns dynamically
    column_mapping = {}
    for standard_name, variations in skill_columns.items():
        for col in trainers_df.columns:
            if col in variations:
                column_mapping[standard_name] = col
                break

    if len(column_mapping) < 2:
        st.error("❌ Missing required skill columns ('primary skills' or 'secondary skills') in the trainer file.")
        st.stop()

    primary_col, secondary_col = column_mapping['primary skills'], column_mapping['secondary skills']
    trainers_df[primary_col] = trainers_df[primary_col].fillna('')
    trainers_df[secondary_col] = trainers_df[secondary_col].fillna('')

    # Dynamically search for user-provided skill if entered
    if user_skill:
        user_matches = match_user_skills(user_skill, trainers_df[primary_col], trainers_df[secondary_col])
        st.write(f"Matches for '{user_skill}' in skills data: {user_matches}")
        st.session_state.skill_variations[user_skill] = [user_skill]  # Add the user skill to session state

    skill_scores = {}
    qualified_mask = pd.Series(True, index=trainers_df.index)
    for skill in selected_skills + ([user_skill] if user_skill else []):
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
        trainers_df = clean_column_names(pd.read_csv("Current delivery workforce - Raw Data.csv"))
        st.success("✅ Static trainer data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading static trainer file: {e}")
        return

    # Upload Managers Data
    st.header("1. Upload Managers Data")
    managers_file = st.file_uploader("Upload Managers CSV", type='csv')
    if managers_file is not None:
        managers_df = pd.read_csv(managers_file)
        managers_df = validate_managers_df(managers_df)

        # Configuration
        st.header("2. Configure Analysis")
        skills = list(st.session_state.skill_variations.keys())
        selected_skills = st.multiselect("Select Skills", options=skills, default=['python', 'nodejs'])
        user_skill = st.text_input("Enter a custom skill name for analysis (e.g., 'Go' for Golang):")
        min_score = st.slider("Minimum Skill Score (%)", 0, 100, 70, step=5)

        # Run Analysis
        if st.button("🔍 Analyze Skills"):
            with st.spinner("Analyzing data..."):
                results = analyze_skills(trainers_df, managers_df, selected_skills, user_skill, min_score)

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
