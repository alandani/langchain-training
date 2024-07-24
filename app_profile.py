import streamlit as st
from chat_profile import get_profile

# Streamlit App
st.title("Profile Search")

# Input text for name
name_input = st.text_input("Enter a name:")

# Button to submit the name
if st.button("Search Profile"):
    summary, profile_picture = get_profile(name_input)
    if summary or profile_picture:
        # Display name
        st.header(name_input)

        # Display profile picture
        st.image(profile_picture, width=150)

        # Display summary
        st.subheader("Summary")
        st.write(summary.summary)

        # Display education list
        st.subheader("Interesting Facts")
        facts = summary.facts
        for x in facts:
            st.write(f"- {x}")
    else:
        st.error("Profile not found.")