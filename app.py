import streamlit as st
import pandas as pd
import re
from datetime import datetime
import base64
from fuzzywuzzy import process
from typing import Dict, List, Optional
import json
import csv
import os

# Page config
st.set_page_config(
    page_title="Skills Analyzer",
    page_icon="📊",
    layout="wide"
)

# Cache the CSV conversion
@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV string with caching"""
    return df.to_csv(index=False).encode('utf-8')

def save_log_to_csv(log_data: dict):
    """Save the log data to the logs CSV file"""
    log_df = pd.DataFrame([log_data])
    try:
        # Try to read existing logs
        existing_logs = pd.read_csv('data/logs.csv')
        updated_logs = pd.concat([existing_logs, log_df], ignore_index=True)
    except FileNotFoundError:
        updated_logs = log_df
    # Save updated logs
    updated_logs.to_csv('data/logs.csv', index=False)

# Load Skills State
def load_skill_variations() -> Dict[str, List[str]]:
    """Load skill variations from the skills_state.txt file."""
    try:
        with open("skills_state.txt", "r") as file:
            skill_variations = json.load(file)
        return skill_variations
    except Exception as e:
        st.error(f"Error loading skill variations: {e}")
        return {}

# Initialize skill variations in session state
if "skill_variations" not in st.session_state:
    st.session_state.skill_variations = load_skill_variations()

# Helper Functions
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters, spaces, and formatting."""
    df.columns = (
        df.columns
        .str.replace(r'^="', '', regex=True)
        .str.replace(r'"$', '', regex=True)
        .str.strip()
        .str.lower()
        .str.replace(' ', '')
        .str.replace('_', '')
    )
    return df

def validate_managers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate managers DataFrame and map columns dynamically with detailed feedback."""
    required_columns = {
        'developer turing email': ['developerturingemail', 'developer_turing_email', 'developer turing email'],
        'manager turing email': ['managerturingemail', 'manager_turing_email', 'manager turing email']
    }

    df = clean_column_names(df)
    
    # Display original columns for debugging
    st.write("📋 Detected columns in your file:", df.columns.tolist())
    
    column_mapping = {}
    detected_columns = []
    missing_columns = []
    
    for standard_col, variations in required_columns.items():
        found = False
        for col in df.columns:
            if col in variations:
                column_mapping[col] = standard_col
                detected_columns.append(f"✅ Found '{standard_col}' as '{col}'")
                found = True
                break
        if not found:
            missing_columns.append(f"❌ Missing: '{standard_col}'")
    
    # Show detailed feedback
    st.write("### Column Validation Results")
    
    # Show detected columns
    if detected_columns:
        st.success("**Successfully detected columns:**")
        for msg in detected_columns:
            st.write(msg)
    
    # Show missing columns
    if missing_columns:
        st.error("**Missing required columns:**")
        for msg in missing_columns:
            st.write(msg)
            
    # Show column mapping hint
    st.info("**Expected column variations:**")
    for standard_col, variations in required_columns.items():
        st.write(f"- '{standard_col}' can be any of: {', '.join(variations)}")

    if len(column_mapping) < len(required_columns):
        st.error("❌ Please check your file and ensure it contains the required columns.")
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

def main():
    st.title("🎯 Skills Analyzer")

    # Load Static Trainer Data
    try:
        trainers_df = clean_column_names(pd.read_csv("Current delivery workforce - Raw Data.csv"))
        st.success("✅ Static trainer data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading static trainer file: {e}")
        return

    # Upload Managers Data with enhanced feedback
    st.header("1. Upload Managers Data")
    with st.expander("ℹ️ Column Requirements", expanded=False):
        st.write("""
        Your CSV file should contain the following columns:
        - Developer Turing Email
        - Manager Turing Email
        
        The columns can be named in various formats (with/without spaces, with underscores, etc.)
        """)
    
    managers_file = st.file_uploader("Upload Managers CSV", type='csv')
    if managers_file is not None:
        try:
            managers_df = pd.read_csv(managers_file)
            st.success("📁 File uploaded successfully!")
            managers_df = validate_managers_df(managers_df)
            st.success("✅ Managers data validated and processed successfully!")
            
            # Configuration
            st.header("2. Configure Analysis")
            skills = list(st.session_state.skill_variations.keys())
            selected_skills = st.multiselect("Select Skills", options=skills, default=['python', 'node.js'])
            user_skill = st.text_input("Enter a custom skill name for analysis (e.g., 'Go' for Golang):")
            min_score = st.slider("Minimum Skill Score (%)", 0, 100, 70, step=5)

            # Run Analysis
            if st.button("🔍 Analyze Skills"):
                with st.spinner("Analyzing data..."):
                    results = analyze_skills(trainers_df, managers_df, selected_skills, user_skill, min_score)

                # Display Results Summary
                st.header("3. Results Summary")
                rows_found = results.shape[0]
                
                # Only show the count and analysis info
                st.info(f"""
                📊 Analysis Results:
                - Total trainers found: {rows_found}
                - Skills analyzed: {', '.join(selected_skills + ([user_skill] if user_skill else []))}
                - Minimum score requirement: {min_score}%
                """)
                
                if rows_found > 0:
                    st.markdown("---")
                    st.subheader("📝 Download Information")
                    st.write("Please fill in the following details to download the results:")

                    # Using st.form to prevent rerun on every input change
                    with st.form(key="download_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            turing_email = st.text_input("Turing Email ID")
                            project_name = st.text_input("Project Name")
                            
                        with col2:
                            client_name = st.text_input("Client Name")
                            opportunity_type = st.selectbox(
                                "Opportunity Type",
                                ["Fulltime", "Part Time"]
                            )
                        
                        submitted = st.form_submit_button("📥 Download Results", type="primary")
                        
                        if submitted:
                            if not all([turing_email, project_name, client_name]):
                                st.error("⚠️ Please fill in all required fields")
                            else:
                                try:
                                    # Generate filename
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    filename = f"qualified_trainers_{timestamp}.csv"
                                    
                                    # Convert DataFrame to CSV
                                    csv_data = convert_df_to_csv(results)
                                    
                                    # Create download button
                                    st.success("✅ Details saved successfully! Click below to download.")
                                    st.download_button(
                                        label="📥 Download CSV File",
                                        data=csv_data,
                                        file_name=filename,
                                        mime="text/csv",
                                        key='download-csv'
                                    )
                                    
                                except Exception as e:
                                    st.error(f"❌ Error preparing download: {str(e)}")
                                    st.exception(e)  # This will show the full error trace
                        
        except Exception as e:
            st.error(f"Error processing managers file: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
