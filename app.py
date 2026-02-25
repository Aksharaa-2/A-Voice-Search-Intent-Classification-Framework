import streamlit as st
from webpage_analyzer import analyze_webpage, compute_score, recommendations

st.title("Voice Search SEO Analyzer")

url = st.text_input("Enter webpage URL")

if st.button("Analyze"):
    features = analyze_webpage(url)
    score = compute_score(features)
    recs = recommendations(features)

    st.subheader("Voice Compatibility Score")
    st.write(score)

    st.subheader("Recommendations")
    for r in recs:
        st.write("-", r)