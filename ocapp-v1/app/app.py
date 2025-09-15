import streamlit as st
from ui.app_main import render_app

st.set_page_config(page_title="Option Chain Dashboard", layout="wide")

def main():
    render_app()

if __name__ == "__main__":
    main()
