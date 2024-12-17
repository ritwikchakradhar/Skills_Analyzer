import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Set
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Skill variations mapping - add more skills and variations as needed
SKILL_VARIATIONS = {
    'nodejs': ['node', 'nodejs', 'node.js', 'node-js', 'node js'],
    'python': ['python', 'python3', 'python 3'],
    'java': ['java', 'core java', 'java se'],
    'kotlin': ['kotlin', 'kotlin-android', 'kotlin android'],
    'react': ['react', 'reactjs', 'react.js', 'react js'],
    'angular': ['angular', 'angularjs', 'angular.js', 'angular js'],
    # Add more skills as needed
}

class EmailValidator:
    """Handles email validation logic."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        if pd.isna(email) or email == 'N/A' or email == '':
            return False
        return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(email)))

class SkillsAnalyzer:
    """Generic skills analyzer that requires all specified skills to meet threshold."""
    
    def __init__(
        self,
        trainers_path: str,
        managers_path: str,
        minimum_score: float = 70,
        business_lines: Optional[List[str]] = None,
        skills_to_analyze: Optional[List[str]] = None,
        skill_variations: Optional[Dict[str, List[str]]] = None
    ):
        """Initialize analyzer with paths and configuration."""
        self.trainers_path = Path(trainers_path)
        self.managers_path = Path(managers_path)
        self.minimum_score = minimum_score
        self.business_lines = business_lines
        self.skills_to_analyze = skills_to_analyze or ['python', 'nodejs', 'java']
        self.skill_variations = skill_variations or SKILL_VARIATIONS
        self.validator = EmailValidator()
        
        # Create reverse mapping for skill variations
        self.skill_mapping = {}
        for main_skill, variations in self.skill_variations.items():
            for variation in variations:
                self.skill_mapping[variation.lower()] = main_skill

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean quotes and special characters from DataFrame."""
        df.columns = df.columns.str.replace('="', '').str.replace('"', '')
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('="', '').str.replace('"', '')
        return df
    
    def _extract_skill_score(self, skills_str: str, skill_variations: Set[str]) -> Optional[float]:
        """Extract skill score from skills string for any variation of the skill."""
        if pd.isna(skills_str):
            return None
            
        skills_str = skills_str.lower()
        for variation in skill_variations:
            pattern = fr'{variation}\s*-\s*(\d+\.?\d*)%'
            match = re.search(pattern, skills_str)
            if match:
                return float(match.group(1))
        return None

    def _calculate_skill_scores(self, row: pd.Series) -> pd.Series:
        """Calculate scores for all specified skills."""
        results = {}
        
        for skill in self.skills_to_analyze:
            variations = set(v.lower() for v in self.skill_variations.get(skill, [skill]))
            
            primary_score = self._extract_skill_score(str(row['Primary Skills']), variations)
            secondary_score = self._extract_skill_score(str(row['Secondary Skills']), variations)
            
            results[f'{skill}_Primary_Score'] = primary_score
            results[f'{skill}_Secondary_Score'] = secondary_score
            results[f'{skill}_Max_Score'] = max(
                filter(None, [primary_score, secondary_score]), 
                default=None
            )
            
        return pd.Series(results)

    def analyze_and_export(self) -> str:
        """Perform the analysis of trainers and export results."""
        try:
            # Read and validate input files
            if not self.trainers_path.exists():
                raise FileNotFoundError(f"Trainers file not found: {self.trainers_path}")
            if not self.managers_path.exists():
                raise FileNotFoundError(f"Managers file not found: {self.managers_path}")
            
            trainers_df = pd.read_csv(self.trainers_path)
            managers_df = pd.read_csv(self.managers_path)
            
            # Clean managers data
            managers_df = self._clean_dataframe(managers_df)
            
            # Create manager mapping
            valid_managers_df = managers_df[
                managers_df['Manager Turing Email'].apply(self.validator.is_valid_email)
            ].copy()
            manager_mapping = valid_managers_df.groupby('Developer turing email')[
                'Manager Turing Email'
            ].first().to_dict()
            
            # Filter for relevant business lines if specified
            df_filtered = trainers_df
            if self.business_lines:
                df_filtered = trainers_df[
                    trainers_df['Business line'].isin(self.business_lines)
                ].copy()
            
            # Calculate skill scores
            skill_scores = df_filtered.apply(self._calculate_skill_scores, axis=1)
            df_filtered = pd.concat([df_filtered, skill_scores], axis=1)
            
            # Filter qualified trainers (must meet minimum score for ALL skills)
            qualified_mask = pd.Series(True, index=df_filtered.index)
            for skill in self.skills_to_analyze:
                skill_mask = df_filtered[f'{skill}_Max_Score'] >= self.minimum_score
                qualified_mask &= skill_mask
                
                # Log how many trainers have this skill at required level
                logger.info(f"Trainers with {skill} >= {self.minimum_score}%: {skill_mask.sum()}")
            
            qualified_trainers = df_filtered[qualified_mask].copy()
            
            # Add manager information
            qualified_trainers['Manager_Turing_Email'] = qualified_trainers[
                'Developer turing email'
            ].map(manager_mapping)
            
            # Sort by average of maximum scores across all skills
            score_columns = [f'{skill}_Max_Score' for skill in self.skills_to_analyze]
            qualified_trainers['Average_Skill_Score'] = qualified_trainers[score_columns].mean(axis=1)
            qualified_trainers = qualified_trainers.sort_values(
                by='Average_Skill_Score',
                ascending=False,
                na_position='last'
            )
            
            # Export results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            skills_str = '_'.join(self.skills_to_analyze)
            export_filename = f'qualified_trainers_all_{skills_str}_{timestamp}.csv'
            qualified_trainers.to_csv(export_filename, index=False, encoding='utf-8')
            
            # Log results
            logger.info("\nAnalysis Summary:")
            logger.info(f"Total trainers analyzed: {len(trainers_df)}")
            logger.info(f"Total qualified trainers (with ALL required skills): {len(qualified_trainers)}")
            logger.info(f"Required skills: {', '.join(self.skills_to_analyze)}")
            logger.info(f"Minimum score threshold: {self.minimum_score}%")
            logger.info(f"Results exported to: {export_filename}")
            
            if len(qualified_trainers) > 0:
                logger.info("\nTop qualified trainer scores:")
                for _, row in qualified_trainers.head(1).iterrows():
                    logger.info(f"Developer: {row['Developer']}")
                    for skill in self.skills_to_analyze:
                        logger.info(f"{skill}: {row[f'{skill}_Max_Score']}%")
            
            return export_filename
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

def main():
    """Main function to run the analysis."""
    analyzer = SkillsAnalyzer(
        trainers_path='/Users/ritwikchakradhar/Documents/Dev Mobility - Data/Current delivery workforce - Raw Data(10th Nov).csv',
        managers_path='/Users/ritwikchakradhar/Documents/Dev Mobility - Data/Engagements-01-Dec-2024-to-31-Dec-2024_Updated.csv',
        minimum_score=70,                        # Minimum skill score threshold
        business_lines=['LLM', 'Services'],      # Business lines to include (optional)
        skills_to_analyze=['python', 'nodejs'],  # Skills to analyze - trainers must have ALL these
        skill_variations=SKILL_VARIATIONS        # Skill name variations mapping
    )
    
    try:
        export_path = analyzer.analyze_and_export()
        print(f"\nAnalysis completed successfully. Results saved to: {export_path}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()