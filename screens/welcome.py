import streamlit as st

#Welcome Screen
# This screen provides a brief introduction to the app and its functionalities.
def show():
    st.subheader('Quickstart')
    st.write("Use the navigation tab on the left hand side to visit different links.")

    st.subheader("Introduction")
    st.write("""
    This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
    """)