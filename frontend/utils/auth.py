"""
Authentication Utility for Streamlit Dashboard.
"""

import streamlit as st

def check_auth():
    """Simple password-based authentication for the dashboard."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 80vh;
            }
            .login-box {
                background-color: #1a1a2e;
                padding: 3rem;
                border-radius: 15px;
                border: 1px solid #4ECDC4;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                max-width: 400px;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("  Intelligence Suite Login")
        
        with st.form("login_form"):
            st.markdown("Please enter the administrator password to access the analytics suite.")
            password = st.text_input("Password", type="password", placeholder="Enter Password")
            
            submit = st.form_submit_button("Enter Dashboard", use_container_width=True)
            
            if submit:
                # In a real app, this would check against a DB or env var
                if password == "admin123":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

def logout():
    """Log the user out and reset session state."""
    st.session_state.authenticated = False
    st.rerun()
