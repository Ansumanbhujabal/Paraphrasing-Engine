# import re
# import mlflow

# # Function to parse the log file and extract relevant information
# def parse_nli_log(log_file_path):
#     nli_entries = []
    
#     with open(log_file_path, 'r') as log_file:
#         log_lines = log_file.readlines()

#     # Regular expression patterns to capture premise, hypothesis, label, and probabilities
#     premise_pattern = re.compile(r"Performing NLI inference between premise: '(.*?)' and hypothesis: '(.*?)'")
#     label_pattern = re.compile(r"NLI Label: (\w+), Probabilities: \[(.*?)\]")

#     current_premise = None
#     current_hypothesis = None
    
#     # Loop over the log lines to find relevant data
#     for line in log_lines:
#         premise_match = premise_pattern.search(line)
#         label_match = label_pattern.search(line)

#         if premise_match:
#             current_premise = premise_match.group(1)
#             current_hypothesis = premise_match.group(2)
        
#         if label_match:
#             label = label_match.group(1)
#             probabilities = [float(x.strip()) for x in label_match.group(2).split()]
            
#             # Save the parsed data
#             if current_premise and current_hypothesis:
#                 nli_entries.append({
#                     "premise": current_premise,
#                     "hypothesis": current_hypothesis,
#                     "label": label,
#                     "probabilities": probabilities
#                 })
                
#                 # Reset premise and hypothesis for the next entry
#                 current_premise = None
#                 current_hypothesis = None
                
#     return nli_entries
# # Function to log NLI inference results to MLflow
# def log_nli_to_mlflow(nli_entries):
#     for entry in nli_entries:
#         with mlflow.start_run():
#             # Log the premise and hypothesis as parameters
#             mlflow.log_param("premise", entry['premise'])
#             mlflow.log_param("hypothesis", entry['hypothesis'])

#             # Log the NLI label
#             mlflow.log_param("NLI_label", entry['label'])

#             # Log the probabilities
#             mlflow.log_metric("entailment_probability", entry['probabilities'][2])
#             mlflow.log_metric("neutral_probability", entry['probabilities'][1])
#             mlflow.log_metric("contradiction_probability", entry['probabilities'][0])
# def process_nli_log(log_file_path):
#     # Parse the log file
#     nli_entries = parse_nli_log(log_file_path)
    
#     # Log the parsed entries to MLflow
#     log_nli_to_mlflow(nli_entries)
    
# # Specify the log file path
# log_file_path = "/teamspace/studios/this_studio/Paraphrasing-Engine/log/nli_logs.log"

# # Process the log and log to MLflow
# process_nli_log(log_file_path)



import re
import streamlit as st
import matplotlib.pyplot as plt

# Function to parse the log file and extract relevant information
def parse_nli_log(log_file_path):
    nli_entries = []
    
    with open(log_file_path, 'r') as log_file:
        log_lines = log_file.readlines()

    # Regular expression patterns to capture premise, hypothesis, label, and probabilities
    premise_pattern = re.compile(r"Performing NLI inference between premise: '(.*?)' and hypothesis: '(.*?)'")
    label_pattern = re.compile(r"NLI Label: (\w+), Probabilities: \[(.*?)\]")

    current_premise = None
    current_hypothesis = None
    
    # Loop over the log lines to find relevant data
    for line in log_lines:
        premise_match = premise_pattern.search(line)
        label_match = label_pattern.search(line)

        if premise_match:
            current_premise = premise_match.group(1)
            current_hypothesis = premise_match.group(2)
        
        if label_match:
            label = label_match.group(1)
            probabilities = [float(x.strip()) for x in label_match.group(2).split()]
            
            # Save the parsed data
            if current_premise and current_hypothesis:
                nli_entries.append({
                    "premise": current_premise,
                    "hypothesis": current_hypothesis,
                    "label": label,
                    "probabilities": probabilities
                })
                
                # Reset premise and hypothesis for the next entry
                current_premise = None
                current_hypothesis = None
                
    return nli_entries

# Function to display NLI results using Streamlit
def display_nli_results(nli_entries):
    st.title("NLI Inference Results")
    
    # Loop through the entries and visualize each result
    for entry in nli_entries:
        st.header("Premise: " + entry["premise"])
        st.subheader("Hypothesis: " + entry["hypothesis"])
        st.text("NLI Label: " + entry["label"])

        # Create a bar chart for the probabilities
        labels = ['Contradiction', 'Neutral', 'Entailment']
        probabilities = entry["probabilities"]

        fig, ax = plt.subplots()
        ax.bar(labels, probabilities, color=['red', 'orange', 'green'])
        ax.set_ylabel('Probability')
        ax.set_title('NLI Probabilities')

        # Display the bar chart in Streamlit
        st.pyplot(fig)

def process_nli_log(log_file_path):
    # Parse the log file
    nli_entries = parse_nli_log(log_file_path)
    
    # Display the parsed entries using Streamlit
    display_nli_results(nli_entries)
    
# Specify the log file path
log_file_path = "/teamspace/studios/this_studio/Paraphrasing-Engine/log/nli_logs.log"

# Process the log and display it in Streamlit
if __name__ == "__main__":
    process_nli_log(log_file_path)
