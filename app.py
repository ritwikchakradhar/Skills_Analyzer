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

# Initialize session state for skill variations
if 'skill_variations' not in st.session_state:
    st.session_state.skill_variations = {
        'nodejs': ['node', 'nodejs', 'node.js', 'node-js', 'node js'],
        'python': ['python', 'python3', 'python 3', 'python(django)', 'python automation', 'python security automation'],
        'java': ['java', 'core java', 'java se'],
        'kotlin': ['kotlin', 'kotlin-android', 'kotlin android'],
        'react': ['react', 'reactjs', 'react.js', 'react js', 'react hooks'],
        'angular': ['angular', 'angularjs', 'angular.js', 'angular js'],
        'typescript': ['typescript', 'ts'],
        'javascript': ['javascript', 'js', 'es6'],
        'golang': ['golang', 'go'],
        'rust': ['rust', 'rustlang'],
        'scala': ['scala'],
        'ruby': ['ruby', 'ruby on rails', 'rails'],
        'php': ['php', 'php/mysql'],
        'csharp': ['c#', 'csharp', 'c-sharp'],
        'cplusplus': ['c++', 'cpp', 'visual c++']
    }

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names removing special characters and standardizing format."""
    df = df.copy()
    df.columns = df.columns.str.replace('="', '')\
                         .str.replace('"', '')\
                         .str.replace('\n', ' ')\
                         .str.strip()\
                         .str.lower()
    return df

def validate_managers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean managers DataFrame."""
    # Show original columns for debugging
    st.write("Original columns in managers file:", list(df.columns))
    
    # Clean column names
    df = clean_column_names(df)
    st.write("Cleaned columns:", list(df.columns))
    
    # Expected column names and their variations
    email_columns = {
        'developer turing email': [
            'developer turing email',
            'developerturingemail',
            'developer_turing_email',
            'developer email',
            'turing email'
        ],
        'manager turing email': [
            'manager turing email',
            'managerturingemail',
            'manager_turing_email',
            'manager email'
        ]
    }
    
    # Find matching columns
    column_mapping = {}
    for standard_name, variations in email_columns.items():
        found = False
        for variant in variations:
            matching_cols = [col for col in df.columns if variant in col.lower().replace(' ', '')]
            if matching_cols:
                column_mapping[matching_cols[0]] = standard_name
                found = True
                break
        if not found:
            raise ValueError(f"Could not find a column matching {standard_name}")
    
    # Rename and clean columns
    df = df.rename(columns=column_mapping)
    
    # Clean email values
    for col in ['developer turing email', 'manager turing email']:
        df[col] = df[col].astype(str)\
                        .str.replace('="', '')\
                        .str.replace('"', '')\
                        .str.strip()
    
    # Remove invalid rows
    df = df[
        df['manager turing email'].notna() & 
        (df['manager turing email'] != 'N/A') &
        (df['manager turing email'] != 'nan') &
        (df['manager turing email'] != '')
    ]
    
    st.write(f"Found {len(df)} valid manager assignments")
    st.write("Sample of processed data:")
    st.write(df[['developer turing email', 'manager turing email']].head())
    
    return df

def extract_skill_score(skills_str: str, variations: List[str]) -> Optional[float]:
    """Extract the highest score for a skill from skills string."""
    if pd.isna(skills_str) or skills_str == 'nan' or not skills_str:
        return None
    
    try:
        skills_str = str(skills_str).lower()
        max_score = None
        
        # Split by lines if multiline
        skills_list = [s.strip() for s in skills_str.split('\n') if s.strip()]
        if not skills_list:
            skills_list = [skills_str]
        
        for skill_item in skills_list:
            for variation in variations:
                pattern = fr'{variation}\s*-\s*(\d+\.?\d*)%'
                matches = re.finditer(pattern, skill_item, re.IGNORECASE)
                
                for match in matches:
                    try:
                        score = float(match.group(1))
                        if max_score is None or score > max_score:
                            max_score = score
                    except (ValueError, TypeError):
                        continue
        
        return max_score
    except Exception:
        return None

def analyze_skills(
    trainers_df: pd.DataFrame,
    managers_df: pd.DataFrame,
    selected_skills: List[str],
    minimum_score: float,
    business_lines: Optional[List[str]] = None
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Analyze skills and return qualified trainers."""
    try:
        # Create working copy and clean data
        trainers_df = trainers_df.copy()
        trainers_df = clean_column_names(trainers_df)
        minimum_score = float(minimum_score)
        
        # Clean skills columns
        trainers_df['primary skills'] = trainers_df['primary skills'].fillna('').astype(str)
        trainers_df['secondary skills'] = trainers_df['secondary skills'].fillna('').astype(str)
        
        # Filter business lines
        if business_lines:
            trainers_df = trainers_df[trainers_df['business line'].isin(business_lines)].copy()
        
        skill_scores = {}
        qualified_mask = pd.Series(True, index=trainers_df.index)
        skill_stats = {}
        
        # Process each skill
        for skill in selected_skills:
            variations = st.session_state.skill_variations.get(skill, [skill])
            
            # Calculate scores
            primary_scores = pd.Series([
                extract_skill_score(str(x), variations) for x in trainers_df['primary skills']
            ])
            secondary_scores = pd.Series([
                extract_skill_score(str(x), variations) for x in trainers_df['secondary skills']
            ])
            
            # Convert to numeric
            primary_scores = pd.to_numeric(primary_scores, errors='coerce')
            secondary_scores = pd.to_numeric(secondary_scores, errors='coerce')
            
            # Get maximum score
            max_scores = pd.DataFrame({
                'primary': primary_scores,
                'secondary': secondary_scores
            }).max(axis=1, skipna=True)
            
            skill_scores[f'{skill}_Max_Score'] = max_scores
            skill_mask = max_scores.fillna(0) >= minimum_score
            qualified_mask &= skill_mask
            skill_stats[skill] = skill_mask.sum()
        
        # Get qualified trainers
        qualified_trainers = trainers_df[qualified_mask].copy()
        
        if len(qualified_trainers) > 0:
            # Add skill scores
            for col, scores in skill_scores.items():
                qualified_trainers[col] = scores.round(1)
            
            # Add manager information
            manager_mapping = managers_df.set_index('developer turing email')['manager turing email'].to_dict()
            qualified_trainers['Manager_Turing_Email'] = qualified_trainers['developer turing email'].map(manager_mapping)
            
            # Calculate average score
            score_columns = [f'{skill}_Max_Score' for skill in selected_skills]
            qualified_trainers['Average_Skill_Score'] = qualified_trainers[score_columns].mean(axis=1).round(1)
            
            # Sort results
            qualified_trainers = qualified_trainers.sort_values(
                by='Average_Skill_Score',
                ascending=False,
                na_position='last'
            )
        
        return qualified_trainers, skill_stats
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return pd.DataFrame(), {}

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link for DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download Results CSV</a>'

def main():
    st.title("🎯 Skills Analyzer")
    
    st.markdown("""
        <style>
        .download-button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    try:
        trainers_df = pd.read_csv('Current delivery workforce - Raw Data.csv')
        trainers_df = clean_column_names(trainers_df)
        
        st.write("""
        Upload managers data file to analyze skills and find qualified trainers.
        Trainers must meet the minimum score threshold in ALL selected skills to qualify.
        """)
        
        st.header("1. Upload Managers Data")
        managers_file = st.file_uploader(
            "Upload Managers CSV",
            type='csv',
            help="File must contain 'Developer Turing Email' and 'Manager Turing Email' columns"
        )
        
        if managers_file is not None:
            try:
                managers_df = pd.read_csv(managers_file)
                managers_df = validate_managers_df(managers_df)
                
                if len(managers_df) == 0:
                    st.warning("No valid manager assignments found in the file.")
                    st.stop()
                
                st.success("✅ Managers file processed successfully!")
                
                st.header("2. Configure Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    business_lines = sorted(trainers_df['business line'].unique().tolist())
                    selected_lines = st.multiselect(
                        "Select Business Lines",
                        options=business_lines,
                        default=['LLM', 'Services'] if 'LLM' in business_lines else None,
                        help="Filter trainers by business line"
                    )
                    
                    minimum_score = st.slider(
                        "Minimum Skill Score Required",
                        min_value=0,
                        max_value=100,
                        value=70,
                        step=5,
                        help="Trainers must have at least this score in ALL selected skills"
                    )
                
                with col2:
                    available_skills = sorted(st.session_state.skill_variations.keys())
                    selected_skills = st.multiselect(
                        "Select Required Skills",
                        options=available_skills,
                        default=['python', 'nodejs'] if 'python' in available_skills else None,
                        help="Trainers must have ALL these skills at or above the minimum score"
                    )
                
                with st.expander("Advanced: Edit Skill Name Variations"):
                    for skill in selected_skills:
                        current_variations = ', '.join(st.session_state.skill_variations.get(skill, []))
                        new_variations = st.text_input(
                            f"Variations for {skill}",
                            value=current_variations,
                            help=f"Different ways {skill} might appear in the data"
                        )
                        if new_variations:
                            st.session_state.skill_variations[skill] = [
                                v.strip() for v in new_variations.split(',')
                            ]
                
                if st.button("🔍 Run Analysis", type="primary"):
                    if not selected_skills:
                        st.warning("⚠️ Please select at least one skill to analyze.")
                        st.stop()
                    
                    with st.spinner("🔄 Analyzing data..."):
                        qualified_trainers, skill_stats = analyze_skills(
                            trainers_df=trainers_df,
                            managers_df=managers_df,
                            selected_skills=selected_skills,
                            minimum_score=minimum_score,
                            business_lines=selected_lines
                        )
                    
                    st.header("3. Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Trainers", len(trainers_df))
                    with col2:
                        st.metric("Qualified Trainers", len(qualified_trainers))
                    with col3:
                        rate = (len(qualified_trainers) / len(trainers_df) * 100) if len(trainers_df) > 0 else 0
                        st.metric("Qualification Rate", f"{rate:.1f}%")
                    
                    st.subheader("Skill-wise Qualified Trainers")
                    skill_cols = st.columns(len(skill_stats))
                    for i, (skill, count) in enumerate(skill_stats.items()):
                        with skill_cols[i]:
                            st.metric(f"{skill.title()}", count)
                    
                    if not qualified_trainers.empty:
                        st.subheader("Qualified Trainers")
                        
                        display_columns = [
                            'developer', 'developer turing email', 'Manager_Turing_Email',
                            'business line', 'Average_Skill_Score'
                        ] + [f'{skill}_Max_Score' for skill in selected_skills]
                        
                        st.dataframe(
                            qualified_trainers[display_columns],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'qualified_trainers_{timestamp}.csv'
                        st.markdown(get_download_link(qualified_trainers, filename), unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No trainers found matching all criteria.")
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.error("Please check your file format and try again.")
                
    except Exception as e:
        st.error(f"Error loading trainers data: {str(e)}")
        st.error("Please ensure 'Current delivery workforce - Raw Data.csv' is present in the repository.")

if __name__ == "__main__":
    main()
