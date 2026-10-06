import streamlit as st

from pdf_processor import extract_text_from_pdf
from rag_pipeline import process_document, answer_question


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="College Notes Q&A Chatbot",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 College Notes Q&A Chatbot")

st.write(
    "Upload your college notes PDF and ask questions "
    "based on the content of your notes."
)

st.divider()


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your notes PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"PDF uploaded: {uploaded_file.name}"
    )

    if st.button("Process PDF"):

        with st.spinner(
            "Reading and processing your PDF..."
        ):

            try:
                extracted_text = extract_text_from_pdf(
                    uploaded_file
                )

                if not extracted_text.strip():

                    st.error(
                        "Could not extract text from this PDF."
                    )

                else:

                    st.success(
                        "PDF text extracted successfully!"
                    )

                    st.write(
                        f"Characters extracted: "
                        f"{len(extracted_text):,}"
                    )

                    with st.spinner(
                        "Creating embeddings and storing notes..."
                    ):

                        number_of_chunks = process_document(
                            extracted_text
                        )

                    st.success(
                        f"PDF processed successfully! "
                        f"Created {number_of_chunks} text chunks."
                    )

                    # Store a flag so the user can ask questions
                    st.session_state["pdf_processed"] = True

            except Exception as e:

                st.error(
                    f"Error while processing PDF: {e}"
                )


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions About Your Notes")

question = st.text_input(
    "Ask a question about your notes:",
    placeholder="Example: What are DDL and DML commands?"
)


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.get(
        "pdf_processed",
        False
    ):

        st.warning(
            "Please upload and process a PDF before "
            "asking a question."
        )

    else:

        with st.spinner(
            "Searching your notes and generating an answer..."
        ):

            try:

                answer = answer_question(
                    question
                )

                st.success("🤖 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error while generating answer: {e}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "College Notes Q&A Chatbot • "
    "RAG + Sentence Transformers + ChromaDB + Gemini"
)