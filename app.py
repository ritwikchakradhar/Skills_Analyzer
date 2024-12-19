import streamlit as st
import json

# Function to load skills from the file
def load_skills():
    try:
        with open("skills_state.txt", "r") as file:
            skills_content = file.read()
            # Extract the dictionary from the file
            skills_variations = json.loads(skills_content)
        return skills_variations
    except Exception as e:
        st.error(f"Error loading skills: {e}")
        return {}

# Function to handle exact matching for custom skills
def match_custom_skill(custom_skill, skills_variations, trainer_data):
    for trainer in trainer_data:
        primary_skills = trainer.get("Primary Skills", [])
        secondary_skills = trainer.get("Secondary Skills", [])
        if custom_skill in primary_skills or custom_skill in secondary_skills:
            return trainer
    return None

def main():
    st.title("Skills Analyzer")

    # Load skills from the file
    skills_variations = load_skills()

    # Mock trainer data for demonstration
    trainer_data = [
        {
            "Name": "Trainer A",
            "Primary Skills": ["python", "javascript", "aws"],
            "Secondary Skills": ["docker", "kubernetes"]
        },
        {
            "Name": "Trainer B",
            "Primary Skills": ["java", "angular"],
            "Secondary Skills": ["spring", "react"]
        }
    ]

    # Skill selection dropdown
    skill_options = list(skills_variations.keys())
    selected_skills = st.multiselect("Select Skills", options=skill_options, default=[])

    # Custom skill input
    custom_skill = st.text_input("Enter a Custom Skill")

    # Match custom skill
    if custom_skill:
        matched_trainer = match_custom_skill(custom_skill, skills_variations, trainer_data)
        if matched_trainer:
            st.success(f"Matched Trainer: {matched_trainer['Name']}")
        else:
            st.warning("No trainer found with the custom skill.")

    # Display trainers with selected skills
    if selected_skills:
        st.write("Trainers with selected skills:")
        for trainer in trainer_data:
            primary_skills = trainer.get("Primary Skills", [])
            secondary_skills = trainer.get("Secondary Skills", [])
            combined_skills = primary_skills + secondary_skills

            # Match any variation of the selected skills
            for skill in selected_skills:
                if any(var in combined_skills for var in skills_variations.get(skill, [])):
                    st.write(f"- {trainer['Name']} (Primary: {primary_skills}, Secondary: {secondary_skills})")
                    break

if __name__ == "__main__":
    main()
