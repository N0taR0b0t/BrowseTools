import streamlit as st
from agent import WebBrowsingAgent

st.set_page_config(page_title="Web Browsing Agent", layout="wide")

st.title("🧠 Web Browsing Agent")
search_term = st.text_input("Enter your search term:")

if search_term:
    agent = WebBrowsingAgent()
    with st.spinner("Processing..."):
        agent.run_conversation(search_term)
        st.success("Done!")

    st.subheader("Visited Articles")
    for article in agent.visited_articles:
        st.markdown(f"**{article['url']}**")
        st.write(article['content'])

    st.subheader("Search History")
    for entry in agent.search_results_history:
        st.markdown(f"**Query:** {entry['query']}")
        st.text(entry['results'])