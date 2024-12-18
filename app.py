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
    '.net core': ['.net core', '.netcore', 'dotnet core', '.net-core'],
    'airflow': ['airflow'],
    'algorithms and data structures': ['algorithms and data structures', 'data structures', 'algorithms'],
    'amazon redshift': ['amazon redshift', 'redshift', 'aws redshift'],
    'android': ['android', 'android development'],
    'android hybrid app development': ['android hybrid app development', 'android hybrid apps'],
    'android sdk': ['android sdk', 'android software development kit'],
    'android studio': ['android studio'],
    'android testing': ['android testing', 'android test'],
    'angular': ['angular', 'angular.js', 'angularjs'],
    'angularjs': ['angularjs', 'angular.js'],
    'apache kafka': ['apache kafka', 'kafka'],
    'apache spark': ['apache spark', 'spark'],
    'apollo': ['apollo'],
    'asp.net': ['asp.net', 'aspnet', 'asp.net framework'],
    'asp.net boilerplate': ['asp.net boilerplate', 'boilerplate asp.net'],
    'asp.net core': ['asp.net core', 'aspnet core', 'dotnet core'],
    'asp.net web forms': ['asp.net web forms', 'aspnet web forms'],
    'asp.net/c#': ['asp.net/c#', 'aspnet c#', 'asp.net with c#'],
    'automation testing': ['automation testing', 'automated testing'],
    'aws': ['aws', 'amazon web services'],
    'aws administration': ['aws administration', 'aws admin'],
    'aws amplify': ['aws amplify'],
    'aws aurora': ['aws aurora', 'aurora aws'],
    'aws cli': ['aws cli', 'amazon cli'],
    'aws cognito': ['aws cognito', 'cognito aws'],
    'aws devops': ['aws devops', 'devops on aws'],
    'aws emr': ['aws emr', 'emr aws'],
    'aws glue': ['aws glue'],
    'aws iam': ['aws iam', 'iam aws'],
    'aws lambda': ['aws lambda', 'lambda functions'],
    'aws solutions architecture': ['aws solutions architecture', 'aws architecture'],
    'azure': ['azure', 'microsoft azure'],
    'azure cloud sql': ['azure cloud sql', 'cloud sql azure'],
    'azure cosmos db': ['azure cosmos db', 'cosmos db azure'],
    'azure data factory': ['azure data factory'],
    'azure data lake': ['azure data lake'],
    'azure databricks': ['azure databricks', 'databricks azure'],
    'azure devops': ['azure devops'],
    'azure eventhub': ['azure eventhub', 'eventhub azure'],
    'azure function app': ['azure function app', 'function app azure'],
    'bash': ['bash', 'bash scripting'],
    'bigquery': ['bigquery', 'google bigquery'],
    'blockchain': ['blockchain'],
    'business analysis': ['business analysis', 'business analyst'],
    'c': ['c', 'c programming'],
    'c#': ['c#', 'c sharp'],
    'c++': ['c++', 'cplusplus'],
    'cakephp': ['cakephp'],
    'circleci': ['circleci'],
    'classic asp': ['classic asp', 'asp classic'],
    'cloud': ['cloud', 'cloud computing'],
    'css3': ['css3', 'css'],
    'd3.js': ['d3.js', 'd3js'],
    'data analysis': ['data analysis', 'data analyst'],
    'data engineering': ['data engineering', 'data engineer'],
    'devops': ['devops'],
    'django': ['django', 'python django'],
    'docker': ['docker'],
    'elixir': ['elixir'],
    'embedded systems': ['embedded systems'],
    'express.js': ['express.js', 'expressjs', 'express'],
    'firebase': ['firebase'],
    'flask': ['flask', 'python flask'],
    'flutter': ['flutter'],
    'git': ['git'],
    'github': ['github'],
    'gitlab': ['gitlab'],
    'go': ['go', 'golang'],
    'graphql': ['graphql'],
    'heroku': ['heroku'],
    'html': ['html', 'html5'],
    'integration testing': ['integration testing'],
    'ionic': ['ionic'],
    'ios - swift': ['ios - swift', 'swift for ios'],
    'java': ['java', 'core java', 'java se', 'java programming'],
    'javascript': ['javascript', 'js', 'es6'],
    'jenkins': ['jenkins'],
    'junit': ['junit'],
    'kafka': ['kafka', 'apache kafka'],
    'kotlin': ['kotlin'],
    'kubernetes': ['kubernetes'],
    'laravel': ['laravel'],
    'machine learning': ['machine learning', 'ml'],
    'matlab': ['matlab'],
    'meteor.js': ['meteor.js', 'meteorjs'],
    'microservices': ['microservices'],
    'mobile app development': ['mobile app development'],
    'mongodb': ['mongodb', 'mongo'],
    'ms sql server': ['ms sql server', 'mssql', 'sql server'],
    'mysql': ['mysql'],
    'next.js': ['next.js', 'nextjs'],
    'node.js': ['node.js', 'node', 'nodejs'],
    'nuxt.js': ['nuxt.js', 'nuxtjs'],
    'php': ['php'],
    'postgresql': ['postgresql', 'postgres'],
    'product design': ['product design'],
    'project management': ['project management'],
    'python': ['python', 'python3', 'python 3', 'python(django)', 'python automation', 'python security automation'],
    'react': ['react', 'reactjs', 'react.js', 'react hooks'],
    'redux': ['redux'],
    'rest/restful apis': ['rest', 'restful apis', 'rest apis'],
    'ruby': ['ruby', 'ruby on rails'],
    'scala': ['scala'],
    'scss': ['scss'],
    'selenium': ['selenium'],
    'sql': ['sql', 'structured query language'],
    'typescript': ['typescript', 'ts'],
    'vue.js': ['vue.js', 'vuejs'],
    'wordpress': ['wordpress']
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
        selected_skills = st.multiselect("Select Skills", options=skills, default=['python', 'node.js'])
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
