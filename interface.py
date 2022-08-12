import streamlit as st
from co import CoHere
import numpy as np

st.header("Your personal text classifier - Co:here application")

api_key = st.text_input("API Key:", type="password")

description = [st.text_input("Description:")]

cohere = CoHere(api_key)
cohere.list_of_examples()

if st.button("Classify"):
    here = cohere.classify(description)[0]
    col1, col2 = st.columns(2)
    for no, con in enumerate(here.confidence):
        if no % 2 == 0:
            col1.write(f"{con.label}: {np.round(con.confidence*100, 2)}%")
            col1.progress(con.confidence)
        else:
            col2.write(f"{con.label}: {np.round(con.confidence * 100, 2)}%")
            col2.progress(con.confidence)

