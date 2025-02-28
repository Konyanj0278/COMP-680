# COMP-680
Class Project for COMP 680

# Log In issue solution:
Go to file COMP-680\venv\Lib\site-packages\streamlit_login_auth_ui\widgets.py
<br>
Around line (111-113)
<br>
Change "st.experimental_rerun()" to "st.rerun()"