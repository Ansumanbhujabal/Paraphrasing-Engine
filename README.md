# Paraphrasing-Engine
This project is a streamlined paraphrasing and summarization engine that applies Natural Language Inference (NLI) to ensure meaning retention between the original and paraphrased text, followed by summarization. Logs are visualized to monitor model performance, and the application offers a user-friendly interface built with Streamlit. The entire application is containerized for easy deployment.

---
# Architecture

![chatuml-diagram](https://github.com/user-attachments/assets/6f57c625-2730-416b-a1ef-8f6245c5b0a9)
![diagram-export-10-30-2024-1_57_29-AM](https://github.com/user-attachments/assets/04eed2a7-a385-493d-ab45-6125504cb149)

## Features

1. **Paraphrasing and Translation:**
   - Text is first paraphrased using Pegasus paraphraser.
   - The paraphrased text is translated to French and then back-translated to English for added linguistic variation.

2. **NLI Validation:**
   - Both the paraphrased and back-translated texts are verified against the original text using NLI, ensuring meaning preservation.

3. **Summarization:**
   - The validated paraphrased and back-translated texts are summarized for concise content output.

4. **Logging and Visualization:**
   - All logs are visualized for a clear view of model performance, enabling insights into paraphrasing, translation, and summarization effectiveness.

5. **Streamlit Interface:**
   - A user-friendly UI with Streamlit allows easy interaction with the tool.

6. **Containerized Application:**
   - Docker support is available for streamlined deployment and scalability.

---

## Setup & Installation

### Prerequisites

- **Python** (version 3.7 or above)
- **Docker** (for containerized deployment)

### Option 1: Running Locally

1. **Create a Virtual Environment:**

    ```bash
    python3 -m venv paraphrase_env
    source paraphrase_env/bin/activate  # On Windows, use paraphrase_env\Scripts\activate
    ```

2. **Clone the Repository:**

    ```bash
    git clone https://github.com/Ansumanbhujabal/Paraphrasing-Engine.git
    cd Paraphrasing-Engine
    ```

3. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

### Option 2: Running with Docker

1. **Build the Docker Image:**

    ```bash
    docker build -t paraphrase_engine .
    ```

2. **Run the Docker Container:**

    ```bash
    docker run -d --name paraphrase_engine -p 8501:8501 paraphrase_engine
    ```

---

## Usage

- Once running, access the application by navigating to `http://localhost:8501` in your browser.
- Input text directly or upload a PDF file for processing.
- View logs to monitor and analyze model performance on paraphrasing, translation, and summarization.

---

## Future Scope

- **Additional Model Options:** Support for more models in paraphrasing, NLI, and summarization for enhanced flexibility.
- **Grafana Dashboard:** Integration of a Grafana-based dashboard for in-depth log visualization and insights.

---

## Contributing

Contributions are welcome! Please clone the repository, create a branch, and submit a pull request for any improvements or feature additions.

---

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.
