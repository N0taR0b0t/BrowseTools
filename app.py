import streamlit as st
from agent import WebBrowsingAgent

st.set_page_config(page_title="Web Browsing Agent", layout="wide")
st.title("🧠 Web Browsing Agent")

# Initialize session memory
if "agent" not in st.session_state:
    st.session_state.agent = WebBrowsingAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Enter your search term:")

# When user submits a query
if user_input:
    with st.spinner("Processing..."):
        messages = st.session_state.chat_history + [
            {"role": "user", "content": user_input}
        ]
        response = st.session_state.agent.run_conversation(messages)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.success("Done!")

# Show assistant's last response
if st.session_state.chat_history:
    last_response = st.session_state.chat_history[-1]
    if last_response["role"] == "assistant":
        st.subheader("Assistant's Response")
        st.write(last_response["content"])

# Show visited articles
st.subheader("Visited Articles")
if st.session_state.agent.visited_articles:
    for article in st.session_state.agent.visited_articles:
        st.markdown(f"**{article['url']}**")
        st.write(article['content'])
else:
    st.write("None")

# Show search history
st.subheader("Search History")
if st.session_state.agent.search_results_history:
    for entry in st.session_state.agent.search_results_history:
        st.markdown(f"**Query:** {entry['query']}")
        st.text(entry['results'])
else:
    st.write("None")