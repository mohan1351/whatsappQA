import streamlit as st

st.set_page_config(page_title="Hello World App", layout="centered")

st.title("Hello World! 🎉")
st.write("This is a simple test Streamlit application.")
st.success("If you see this, Streamlit is working!")

st.write("---")
if st.button("Click me!"):
    st.balloons()
    st.write("✨ Button clicked! You're all set!")
