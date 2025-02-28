# COMP-680
Class Project for COMP 680

# Log In issue solution:
Go to file COMP-680\venv\Lib\site-packages\streamlit_login_auth_ui\widgets.py
around line (111-113) 
Change "st.experimental_rerun()" to "st.rerun()"