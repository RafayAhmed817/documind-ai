import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL","http://localhost:8000")
st.set_page_config(page_title="PDF AI",page_icon="📄")
st.title("📄 PDF AI")
st.caption("Ask questions about your pdf")

with st.sidebar:
    st.header("Upload PDF")
    uploaded=st.file_uploader("Choose a text based pdf ",type=["pdf"])
    if st.button("Index Document",disabled=uploaded is None):
        try:
            response=requests.post(
                f"{API_URL}/documents",
                files={"file":(uploaded.name,uploaded.getvalue(),"application/pdf")},
                timeout=300
            )
            response.raise_for_status()
            result=response.json()
            st.session_state.document_id=result["document_id"]
            st.session_state.messages=[]
            st.success(f"{result['status']}:{result['chunks']} new chunks")
        except requests.RequestException as exc:
            st.error(f"Upload failed:{exc}")
    if st.session_state.get("document_id"):
        st.caption(f"Document ID:{st.session_state.document_id[:12]}...")

question=st.chat_input("Ask about your pdf",disabled=not st.session_state.get("document_id"))
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            response=requests.post(
                f"{API_URL}/ask",
                json={"question":question,"document_id":st.session_state.document_id},
                timeout=300
            )
            response.raise_for_status()
            result=response.json()
            st.markdown(result["answer"])
            with st.expander("Retrieved Sources"):
                for src in result["sources"]:
                    st.write(f"Page{src['page']}-{src['excerpt']}")
            st.session_state.messages.append({"role":"assistant",**result,"content":result["answer"]})
        except requests.RequestException as exc:
            st.error(f"Question failed:{exc}")