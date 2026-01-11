import streamlit as st
from src.brain import get_astro_answer, get_rag_chain

st.set_page_config(page_title="AstroGuide: Space Law Assistant", page_icon="🚀")

st.title("🚀 AstroGuide: Space Law Assistant")
st.markdown("Get expert answers on space law, regulations, and rules for startups and individuals launching into space.")

# (opcjonalnie) tylko żeby złapać błąd braku bazy na starcie
try:
    get_rag_chain()
    st.success("System ready! Ask your questions below.")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

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
            result = get_astro_answer(prompt)
            answer = result["answer"].replace("\\n", "\n")

            st.markdown(answer)

            with st.expander("Sources"):
                if result["sources"]:
                    st.markdown("\n".join(f"- {s}" for s in result["sources"]))
                else:
                    st.write("No sources returned.")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")