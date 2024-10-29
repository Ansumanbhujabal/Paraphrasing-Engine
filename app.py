

import streamlit as st
import os
from Paraphrasing_test import paraphrase_pg
from Back_Translation import backtranslate
from NLI_Validation import nli_for_corpus
from Document_Processing import extract_text
from Summarizer import summarize_text
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")
import tempfile  # for temporary file storage

# Streamlit app
st.title("Paraphrasing Engine")

# Text or PDF input method selection
input_method = st.radio("Choose input method", ("Enter Text", "Upload PDF"))

if input_method == "Enter Text":
    text = st.text_area("Enter your paragraph here:")
elif input_method == "Upload PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        # Extract text from the temporary file
        text = extract_text(tmp_file_path)
        st.subheader("Extracted Text from PDF:")
        st.code(text, language="text")
    else:
        text = ""

# Processing text after button click
if st.button("Process Text"):
    if text:
        st.subheader("Original Text")
        st.code(text, language="text")  # Makes the text output copiable
        def format_nli_score(nli_score):
            label, probabilities = nli_score
            color = {
                'Contradiction': 'red',
                'Neutral': 'blue',
                'Entailment': 'green'
            }.get(label, 'black')  # default to black if label is unexpected
            return f"<span style='color:{color};'>{label}</span> - Probabilities: {probabilities}"        

        # Paraphrase
        paraphrased_text = paraphrase_pg(text)
        st.subheader("Paraphrased Text")
        st.code(paraphrased_text, language="text")  # Makes the paraphrased output copiable
        nli_score1 = nli_for_corpus(text, paraphrased_text)
        nli_score1_label = format_nli_score(nli_score1)
        st.markdown(f"NLI Score for Original vs Paraphrased: {nli_score1_label}", unsafe_allow_html=True)


        # Back-Translation
        back_translated_text = backtranslate(paraphrased_text)
        st.subheader("Back-Translated Paraphrased Text")
        st.code(back_translated_text, language="text")  # Makes the back-translated text copiable
        nli_score2 = nli_for_corpus(text, back_translated_text)
        nli_score2_label = format_nli_score(nli_score2)
        st.markdown(f"NLI Score for Original vs Back-Translated: {nli_score2_label}", unsafe_allow_html=True)        

        # Summarization
        summarized_text1 = summarize_text(back_translated_text)
        summarized_text2 = summarize_text(paraphrased_text)
        
        st.subheader("Summarized Back-Translated Text")
        st.code(summarized_text1, language="text")  # Summarized back-translated text copiable

        st.subheader("Summarized Paraphrased Text")
        st.code(summarized_text2, language="text")  # Summarized paraphrased text copiable

        # Function to format NLI scores with colors


        # Display NLI Scores with color formatting
        st.subheader("NLI Scores")

        # Display NLI Score for Original vs Paraphrased
        nli_score1_label = format_nli_score(nli_score1)
        st.markdown(f"NLI Score for Original vs Paraphrased: {nli_score1_label}", unsafe_allow_html=True)

        # Display NLI Score for Original vs Back-Translated
        nli_score2_label = format_nli_score(nli_score2)
        st.markdown(f"NLI Score for Original vs Back-Translated: {nli_score2_label}", unsafe_allow_html=True)
    else:
        st.error("Please enter text or upload a PDF file.")

