import streamlit as st
from src.brain import get_rag_chain

st.set_page_config(page_title="AstroGuide: Space Law Assistant", page_icon="🚀")

st.title("🚀 AstroGuide: Space Law Assistant")
st.markdown(
    "Get expert answers on space law, regulations, and rules for startups and individuals launching into space."
)

# Initialize chain
try:
    chain = get_rag_chain()
    st.success("System ready! Ask your questions below.")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about space law..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = chain.invoke({prompt})
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
