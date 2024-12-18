import pandas as pd
import re

def extract_unique_skills(file_path: str, primary_col: str, secondary_col: str) -> set:
    """Extract unique skills from primary and secondary skills columns."""
    try:
        # Read the file
        df = pd.read_csv(file_path)

        # Combine primary and secondary skills into one series
        all_skills = pd.concat([df[primary_col], df[secondary_col]], axis=0, ignore_index=True)

        # Split skills by common delimiters (commas, newlines, etc.) and normalize
        skill_set = set()
        for entry in all_skills.dropna():  # Drop NaN values
            # Normalize the skills: split by common delimiters and strip spaces
            skills = re.split(r'[,\n;]+', entry)  # Split by commas, newlines, or semicolons
            normalized_skills = [skill.strip().lower() for skill in skills if skill.strip()]
            skill_set.update(normalized_skills)

        return skill_set
    except Exception as e:
        print(f"Error processing file: {e}")
        return set()

# Example Usage
file_path = "Current delivery workforce - Raw Data.csv"  # Path to your file
primary_col = "Primary_Skills"  # Column name for primary skills
secondary_col = "Secondary_Skills"  # Column name for secondary skills

unique_skills = extract_unique_skills(file_path, primary_col, secondary_col)

# Output the exhaustive list of skills
print(f"Exhaustive list of skills ({len(unique_skills)}):")
print(sorted(unique_skills))
