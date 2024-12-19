import streamlit as st
import json
import os

# File path for the skills state
SKILLS_STATE_FILE = "skills_state.txt"

def load_skills():
    """Load skills from the skills_state.txt file."""
    if os.path.exists(SKILLS_STATE_FILE):
        with open(SKILLS_STATE_FILE, "r") as file:
            skills_content = file.read().strip()
            # Remove the prefix `st.session_state.skill_variations =`
            if skills_content.startswith("st.session_state.skill_variations ="):
                skills_content = skills_content.replace("st.session_state.skill_variations =", "").strip()
            try:
                # Parse the cleaned JSON content
                skills_variations = json.loads(skills_content)
                return skills_variations
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON from skills file: {e}")
                return {}
    else:
        st.error("Skills file not found. Please ensure skills_state.txt is available.")
        return {}


def get_primary_secondary_matches(skills_variations, trainer_data, selected_skills):
    """Fetch matches for selected skills from primary and secondary skills columns."""
    matches = []
    for skill in selected_skills:
        variations = skills_variations.get(skill, [skill])
        for index, row in trainer_data.iterrows():
            if any(variation in row['Primary Skills'] or variation in row['Secondary Skills'] for variation in variations):
                matches.append(row.to_dict())
    return matches

def get_custom_skill_matches(custom_skill, trainer_data):
    """Match an exact custom skill in the trainer data."""
    matches = []
    for index, row in trainer_data.iterrows():
        if custom_skill in row['Primary Skills'] or custom_skill in row['Secondary Skills']:
            matches.append(row.to_dict())
    return matches

def main():
    # Load skills into session state
    skills_variations = load_skills()
    st.session_state.skill_variations = skills_variations

    st.title("Skill Variations App")

    # Simulated trainer data
    trainer_data = st.session_state.get("trainer_data", None)
    if trainer_data is None:
        st.session_state.trainer_data = [
            {"Trainer": "Trainer A", "Primary Skills": "python, flask", "Secondary Skills": "django, sql"},
            {"Trainer": "Trainer B", "Primary Skills": "javascript, react", "Secondary Skills": "node.js, vue.js"},
            # Add more mock trainer rows as needed
        ]

    # Display available skills for selection
    skills = list(st.session_state.skill_variations.keys())
    selected_skills = st.multiselect("Select Skills", options=skills)

    if selected_skills:
        matches = get_primary_secondary_matches(st.session_state.skill_variations, st.session_state.trainer_data, selected_skills)
        st.write("Matching Trainers for Selected Skills:")
        st.write(matches)

    # Input for custom skill
    custom_skill = st.text_input("Enter a Custom Skill")
    if custom_skill:
        matches = get_custom_skill_matches(custom_skill, st.session_state.trainer_data)
        st.write(f"Matching Trainers for Custom Skill '{custom_skill}':")
        st.write(matches)

if __name__ == "__main__":
    main()
