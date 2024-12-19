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

def save_log_to_csv(log_data, filename="trainer_data_logs.csv"):
    """Save the log data to a CSV file."""
    file_exists = os.path.exists(filename)

    # Ensure the file has headers only if it doesn't exist
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())

        if not file_exists:
            writer.writeheader()  # Write headers if file doesn't exist

        writer.writerow(log_data)  # Append the new log entry

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results</a>'

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
            
            # Show preview of processed data
            st.write("### Preview of Processed Managers Data")
            st.dataframe(managers_df.head())

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
                st.success(f"Total trainers found meeting criteria: {rows_found}")

                # Download Section
                if rows_found > 0:
                    filename = f"qualified_trainers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    # Initialize session state for form visibility
                    if 'show_form' not in st.session_state:
                        st.session_state.show_form = False
                    
                    # Download button to trigger form
                    if st.button("📥 Download Trainer Data"):
                        st.session_state.show_form = True
                    
                    # Show form when button is clicked
                    if st.session_state.show_form:
                        with st.form(key="download_form"):
                            st.subheader("Please provide additional information")
                            turing_email = st.text_input("Enter your Turing email ID")
                            project_name = st.text_input("Enter the Project Name")
                            client_name = st.text_input("Enter the Client Name")
                            opportunity_type = st.selectbox(
                                "Select Opportunity Type", 
                                ["Fulltime", "Part Time"]
                            )
                            submit_button = st.form_submit_button("Submit and Download")
                            
                            if submit_button:
                                if not all([turing_email, project_name, client_name]):
                                    st.error("Please fill in all fields")
                                else:
                                    # Save results to CSV
                                    results.to_csv(filename, index=False)
                                    
                                    # Log the download
                                    log_data = {
                                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        "email": turing_email,
                                        "project": project_name,
                                        "client": client_name,
                                        "opportunity": opportunity_type,
                                        "rows_found": rows_found,
                                        "file_path": filename
                                    }
                                    save_log_to_csv(log_data)
                                    
                                    # Create download link
                                    st.success("Details saved successfully!")
                                    st.markdown(get_download_link(results, filename), unsafe_allow_html=True)
                                    
                                    # Reset form visibility
                                    st.session_state.show_form = False
                                    
        except Exception as e:
            st.error(f"Error processing managers file: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
