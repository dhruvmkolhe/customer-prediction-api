import streamlit as st

def inject_material_icons():
    """Inject Material Symbols stylesheet into the app."""
    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
        <style>
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 48;
            vertical-align: middle;
            margin-right: 10px;
        }
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            color: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def material_header(icon_name, title, subtitle=None):
    """Render a header with a Material Symbol."""
    inject_material_icons()
    st.markdown(
        f"""
        <div class="header-container">
            <span class="material-symbols-outlined" style="font-size: 40px; color: #3b82f6;">{icon_name}</span>
            <h1 class="header-title">{title}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(f"*{subtitle}*")
