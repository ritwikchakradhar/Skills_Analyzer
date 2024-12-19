import streamlit as st
import json

def load_skills():
    """Load skills from the skills_state.txt file."""
    try:
        with open("skills_state.txt", "r") as file:
            skills_content = file.read()
            # Parse the dictionary from the file
            skills_variations = json.loads(skills_content)
        return skills_variations
    except Exception as e:
        st.error(f"Error loading skills: {e}")
        return {}

def match_custom_skill(custom_skill, trainer_data):
    """Find a trainer matching the exact custom skill."""
    for trainer in trainer_data:
        primary_skills = trainer.get("Primary Skills", [])
        secondary_skills = trainer.get("Secondary Skills", [])
        if custom_skill in primary_skills or custom_skill in secondary_skills:
            return trainer
    return None

def filter_trainers_by_skill(selected_skills, skills_variations, trainer_data, skill_threshold):
    """Filter trainers based on selected skills and skill threshold."""
    matched_trainers = []
    for trainer in trainer_data:
        primary_skills = trainer.get("Primary Skills", [])
        secondary_skills = trainer.get("Secondary Skills", [])
        combined_skills = primary_skills + secondary_skills

        # Count matches for selected skills
        match_count = 0
        for skill in selected_skills:
            if any(var in combined_skills for var in skills_variations.get(skill, [])):
                match_count += 1

        # Add trainer if matches exceed threshold
        if match_count >= skill_threshold:
            matched_trainers.append(trainer)

    return matched_trainers

def main():
    st.title("Skills Analyzer")

    # Load skills variations from the file
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

    # Skill match threshold
    skill_threshold = st.slider("Skill Match Threshold", min_value=1, max_value=len(skill_options), value=1)

    # Custom skill input
    custom_skill = st.text_input("Enter a Custom Skill")

    # Match custom skill
    if custom_skill:
        matched_trainer = match_custom_skill(custom_skill, trainer_data)
        if matched_trainer:
            st.success(f"Matched Trainer: {matched_trainer['Name']}")
        else:
            st.warning("No trainer found with the custom skill.")

    # Display trainers with selected skills
    if selected_skills:
        st.write("Trainers matching the selected skills:")
        filtered_trainers = filter_trainers_by_skill(selected_skills, skills_variations, trainer_data, skill_threshold)
        if filtered_trainers:
            for trainer in filtered_trainers:
                st.write(f"- {trainer['Name']} (Primary: {trainer['Primary Skills']}, Secondary: {trainer['Secondary Skills']})")
        else:
            st.warning("No trainers match the selected criteria.")

if __name__ == "__main__":
    main()
