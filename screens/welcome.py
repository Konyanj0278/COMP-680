import streamlit as st

# Welcome Screen
# This screen provides a brief introduction to the app and its functionalities.
def show():
    st.markdown("""
        <style>
        .welcome-title {
            font-size: 4rem;
            font-weight: bold;
            color: white;
            text-align: center;
            animation: glowFade 2s ease-in-out infinite alternate;
            margin-bottom: 1rem;
        }

        .fade-in-section {
            animation: fadeIn 1.5s ease-in-out;
            margin-top: 20px;
            font-size: 1.1rem;
        }

        @keyframes glowFade {
            from {
                text-shadow: 0 0 10px #0ff, 0 0 20px #0ff;
            }
            to {
                text-shadow: 0 0 20px #0ff, 0 0 40px #0ff;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="welcome-title"> Welcome to Deep Net</div>', unsafe_allow_html=True)
    st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)

    st.subheader('Quickstart')
    st.write("Use the navigation tab on the left hand side to visit different links.")

    st.subheader("Introduction")
    st.markdown("""
        <div class="intro-text">
        This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. 
        It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements ensure a smoother user experience.
        </div>
    """, unsafe_allow_html=True)