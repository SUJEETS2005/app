import streamlit as st
st.title("Movie Rating App")
st.header("Welcome to Movie Review System")
st.write("Rate your favorite movie here")
st.text("Fill the details below")
st.markdown("### Movie Feedback Form")
st.markdown("*Your review helps others choose good movies*")
st.image("movie.png", caption="Movie Time")
name = st.text_input("Enter your Name")
movie = st.text_input("Enter Movie Name")
age = st.number_input("Enter your Age")
rating = st.slider("Rate the Movie", 1, 10)
language = st.selectbox(
    "Movie Language",
    ["Tamil", "Telugu", "Hindi", "English", "Malayalam"]
)
if st.button("Submit Review"):
    st.write("### Review Submitted ")
    st.write("Name:", name)
    st.write("Movie:", movie)
    st.write("Age:", age)
    st.write("Rating:", rating)
    st.write("Language:", language)
    st.write("Thank you for your review!")