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

def validate_and_map_columns(df: pd.DataFrame, required_columns: Dict[str, List[str]]) -> Dict[str, str]:
    """Validate and map required columns dynamically."""
    cleaned_columns = clean_column_names(df).columns
    column_mapping = {}

    for standard_name, variations in required_columns.items():
        for col in cleaned_columns:
            if any(variation == col for variation in variations):
                column_mapping[standard_name] = col
                break

    # Check if all required columns are found
    missing_columns = [name for name in required_columns if name not in column_mapping]
    if missing_columns:
        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}. Please check your file.")
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
    # Ensure required columns are present
    if 'primary_skills' not in trainers_df.columns or 'secondary_skills' not in trainers_df.columns:
        st.error("❌ Required columns ('primary_skills' or 'secondary_skills') are missing in the trainer data.")
        st.stop()

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
    if 'developer_turing_email' in managers_df.columns:
        manager_map = managers_df.set_index('developer_turing_email')['manager_turing_email'].to_dict()
        qualified_trainers['manager_turing_email'] = qualified_trainers['developer_turing_email'].map(manager_map)

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
        st.write("Available columns in Trainer Data:", trainers_df.columns.tolist())  # Debugging: Show all columns
        required_columns = {
            'business_line': ['businessline', 'business_line'],
            'primary_skills': ['primaryskills', 'primary_skills', 'primary skills', 'primary-skill'],
            'secondary_skills': ['secondaryskills', 'secondary_skills', 'secondary skills', 'secondary-skill'],
            'developer_turing_email': ['developerturingemail', 'developer_email'],
        }
        column_mapping = validate_and_map_columns(trainers_df, required_columns)
        trainers_df = trainers_df.rename(columns=column_mapping)
        st.write("Mapped Columns (Trainers):", column_mapping)  # Debugging: Show mapped columns
        st.success("✅ Trainer data loaded and columns mapped successfully!")
    except Exception as e:
        st.error(f"Error loading trainer file: {e}")
        return

    # Upload Managers Data
    st.header("1. Upload Managers Data")
    managers_file = st.file_uploader("Upload Managers CSV", type='csv')
    if managers_file is not None:
        try:
            managers_df = pd.read_csv(managers_file)
            st.write("Available columns in Managers Data:", managers_df.columns.tolist())  # Debugging: Show all columns
            required_columns = {
                'developer_turing_email': ['developerturingemail', 'developer_email'],
                'manager_turing_email': ['managerturingemail', 'manager_email']
            }
            column_mapping = validate_and_map_columns(managers_df, required_columns)
            managers_df = managers_df.rename(columns=column_mapping)
            st.write("Mapped Columns (Managers):", column_mapping)  # Debugging: Show mapped columns
            st.success("✅ Managers file loaded and columns mapped successfully!")
        except Exception as e:
            st.error(f"Error processing managers file: {e}")
            return

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
