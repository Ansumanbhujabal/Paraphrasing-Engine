
from transformers import pipeline

def summarize_text(text):
    summarizer =pipeline("summarization")
    text_length=len(text.split())
    max_length=int(text_length//1.5)
    min_length=max_length
    summary=summarizer(text,max_length=max_length,min_length= min_length,do_sample=False)
    output = summary[0]['summary_text']
    # return summarizer(text,max_length=max_length,min_length= min_length,do_sample=False)
    return output
