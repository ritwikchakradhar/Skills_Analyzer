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

# Cache data loading
@st.cache_data
def load_trainers_data():
    """Load trainers data from the repository"""
    try:
        df = pd.read_csv('Current delivery workforce - Raw Data.csv')
        # Ensure string type for skills columns
        df['Primary Skills'] = df['Primary Skills'].fillna('').astype(str)
        df['Secondary Skills'] = df['Secondary Skills'].fillna('').astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading trainers data: {str(e)}")
        return None

# Initialize session state for skill variations
if 'skill_variations' not in st.session_state:
    st.session_state.skill_variations = {
        'nodejs': ['node', 'nodejs', 'node.js', 'node-js', 'node js'],
        'python': ['python', 'python3', 'python 3'],
        'java': ['java', 'core java', 'java se'],
        'kotlin': ['kotlin', 'kotlin-android', 'kotlin android'],
        'react': ['react', 'reactjs', 'react.js', 'react js'],
        'angular': ['angular', 'angularjs', 'angular.js', 'angular js'],
        'typescript': ['typescript', 'ts'],
        'javascript': ['javascript', 'js', 'es6'],
        'golang': ['golang', 'go'],
        'rust': ['rust', 'rustlang'],
        'scala': ['scala'],
        'ruby': ['ruby', 'ruby on rails', 'rails'],
        'php': ['php'],
        'csharp': ['c#', 'csharp', 'c-sharp'],
        'cplusplus': ['c++', 'cpp']
    }

def validate_managers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean managers DataFrame."""
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Validate required columns
    required_cols = ['manager turing email', 'developer turing email']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Clean email columns
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace('="', '').str.replace('"', '')
    
    # Remove invalid rows
    df = df[
        df['manager turing email'].notna() & 
        (df['manager turing email'] != 'N/A') &
        (df['manager turing email'] != 'nan') &
        (df['manager turing email'] != '')
    ]
    
    return df

def extract_skill_score(skills_str: str, variations: List[str]) -> Optional[float]:
    """Extract skill score from skills string."""
    if pd.isna(skills_str) or skills_str == 'nan' or not skills_str:
        return None
    
    skills_str = str(skills_str).lower()
    for variation in variations:
        pattern = fr'{variation}\s*-\s*(\d+\.?\d*)%'
        match = re.search(pattern, skills_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
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
        # Filter business lines
        if business_lines:
            trainers_df = trainers_df[trainers_df['Business line'].isin(business_lines)].copy()
        
        skill_scores = {}
        qualified_mask = pd.Series(True, index=trainers_df.index)
        skill_stats = {}
        
        for skill in selected_skills:
            variations = st.session_state.skill_variations.get(skill, [skill])
            
            # Calculate scores
            primary_scores = trainers_df['Primary Skills'].apply(
                lambda x: extract_skill_score(str(x), variations)
            )
            secondary_scores = trainers_df['Secondary Skills'].apply(
                lambda x: extract_skill_score(str(x), variations)
            )
            
            # Convert to numeric
            primary_scores = pd.to_numeric(primary_scores, errors='coerce')
            secondary_scores = pd.to_numeric(secondary_scores, errors='coerce')
            
            # Get maximum score
            max_scores = pd.DataFrame({
                'primary': primary_scores,
                'secondary': secondary_scores
            }).max(axis=1)
            
            skill_scores[f'{skill}_Max_Score'] = max_scores
            skill_mask = max_scores >= minimum_score
            qualified_mask &= skill_mask
            skill_stats[skill] = skill_mask.sum()
        
        # Add skill scores to DataFrame
        for col, scores in skill_scores.items():
            trainers_df[col] = scores
        
        # Get qualified trainers
        qualified_trainers = trainers_df[qualified_mask].copy()
        
        # Add manager information
        manager_mapping = managers_df.groupby('developer turing email')['manager turing email'].first().to_dict()
        qualified_trainers['Manager_Turing_Email'] = qualified_trainers['developer turing email'].map(manager_mapping)
        
        # Calculate average score
        score_columns = [f'{skill}_Max_Score' for skill in selected_skills]
        qualified_trainers['Average_Skill_Score'] = qualified_trainers[score_columns].mean(axis=1)
        
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
    
    # Add custom CSS
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
    
    # Load trainers data
    trainers_df = load_trainers_data()
    if trainers_df is None:
        st.error("Failed to load trainers data. Please check the file in the repository.")
        st.stop()
    
    st.write("""
    Upload managers data file to analyze skills and find qualified trainers.
    Trainers must meet the minimum score threshold in ALL selected skills to qualify.
    """)
    
    # File upload
    st.header("1. Upload Managers Data")
    managers_file = st.file_uploader(
        "Upload Managers CSV",
        type='csv',
        help="File must contain 'Developer Turing Email' and 'Manager Turing Email' columns"
    )
    
    if managers_file is not None:
        try:
            # Read and validate managers file
            managers_df = pd.read_csv(managers_file)
            managers_df = validate_managers_df(managers_df)
            
            if len(managers_df) == 0:
                st.warning("No valid manager assignments found in the file.")
                st.stop()
            
            st.success("✅ Managers file processed successfully!")
            
            # Configuration section
            st.header("2. Configure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                business_lines = sorted(trainers_df['Business line'].unique().tolist())
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
            
            # Skill variations editor
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
            
            # Analysis button
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
                
                # Results section
                st.header("3. Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trainers", len(trainers_df))
                with col2:
                    st.metric("Qualified Trainers", len(qualified_trainers))
                with col3:
                    rate = (len(qualified_trainers) / len(trainers_df) * 100) if len(trainers_df) > 0 else 0
                    st.metric("Qualification Rate", f"{rate:.1f}%")
                
                # Skill statistics
                st.subheader("Skill-wise Qualified Trainers")
                skill_cols = st.columns(len(skill_stats))
                for i, (skill, count) in enumerate(skill_stats.items()):
                    with skill_cols[i]:
                        st.metric(f"{skill.title()}", count)
                
                # Results table
                if not qualified_trainers.empty:
                    st.subheader("Qualified Trainers")
                    
                    display_columns = [
                        'Developer', 'Developer turing email', 'Manager_Turing_Email',
                        'Business line', 'Average_Skill_Score'
                    ] + [f'{skill}_Max_Score' for skill in selected_skills]
                    
                    st.dataframe(
                        qualified_trainers[display_columns],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download link
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'qualified_trainers_{timestamp}.csv'
                    st.markdown(get_download_link(qualified_trainers, filename), unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No trainers found matching all criteria.")
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.error("Please check your file format and try again.")

if __name__ == "__main__":
    main()
