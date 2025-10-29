# AgriAssist: AI-Powered Plant Disease Detection

**AgriAssist** is a comprehensive, full-stack application designed to provide farmers with an advanced tool for identifying plant diseases. It uses a sophisticated AI backend to analyze leaf images and features a voice-enabled Q&A system to answer follow-up questions.

---

## ✨ Project Overview & Technology Stack

This project was built with a modern, robust technology stack, separating the AI logic (backend) from the user interface (frontend).

| Category      | Technology / Framework                                       | Purpose                                                 |
| :------------ | :----------------------------------------------------------- | :------------------------------------------------------ |
| **Backend** | **Python 3.11**, **FastAPI** | For building a high-performance, asynchronous API server. |
| **AI/ML** | **PyTorch**, **Ultralytics (YOLOv8)**, **SentenceTransformers**, **FAISS** | For image classification, object detection, and the RAG Q&A system. |
| **Frontend** | **HTML**, **CSS**, **JavaScript** (with Web Speech API)        | To create a lightweight, interactive, and universally accessible user interface. |
| **Tooling** | **Git & GitHub** | For version control and code management.                  |

---

## 🤖 How the AI System Works

The intelligence of AgriAssist is powered by two distinct AI systems working in tandem.

### ### 1. Image Classification: The Intelligent Ensemble Approach

Instead of relying on a single model, the system uses an **ensemble of three different methods** to ensure the highest possible accuracy for disease detection. When an image is uploaded, it's analyzed by a "team of experts":

1.  **The Basic Classifier:** An EfficientNet model that quickly analyzes the entire image.
2.  **The TTA Specialist:** This method uses **Test-Time Augmentation (TTA)**, showing the same image to the classifier multiple times with slight variations (flipped, rotated, different brightness) and averaging the results for a more robust prediction.
3.  **The YOLOv8 Expert:** This is the most advanced method. It first uses a **YOLOv8 object detection model** to precisely locate the leaf within the image, cropping out any distracting background. It then applies TTA on just the isolated leaf for a highly focused and accurate analysis.

An intelligent scoring algorithm then evaluates the results from all three methods, considering factors like confidence and method reliability, to select the best possible diagnosis.

### ### 2. Voice Q&A: The RAG System

To answer user questions, the app uses a **Retrieval-Augmented Generation (RAG)** system. Think of this as a "smart librarian" for our private knowledge base.

* **The Knowledge Base (`kb_data.jsonl`):** This is a curated collection of documents containing detailed information about plant diseases.
* **The Language Brain (`SentenceTransformer`):** This model understands the *meaning* of the user's spoken question and converts it into a vector (a series of numbers).
* **The Super-Fast Index (`FAISS`):** The entire knowledge base is pre-indexed into a FAISS vector database. This allows for lightning-fast similarity searches.

When a user asks a question, the system finds the most relevant document in the knowledge base and presents that information as the answer, providing a highly accurate, context-specific response without needing an internet connection to a large LLM.

---

## 🔄 Development Process

The project was developed iteratively:
1.  **Model Training:** The core EfficientNet classification model was trained on a labeled dataset of plant leaf images.
2.  **Backend API Development:** A robust backend was built using FastAPI to serve the model. This was later upgraded to include the full ensemble and RAG systems.
3.  **Frontend Prototyping:** A clean web interface was built using HTML, CSS, and JavaScript, ensuring it could communicate with the backend API.
4.  **Advanced Feature Integration:** The YOLOv8 model, TTA logic, and the complete RAG Q&A pipeline (including voice input/output) were integrated into the backend and frontend.
5.  **Local Packaging:** The project was packaged for local execution with a detailed `README.md` to ensure a smooth evaluation process.

---

## 🚀 How to Run This Project

This project is submitted as a local-first package. You will run the backend server on your machine and then open the frontend file in your browser.

### ### Step 1: Set Up and Run the Backend Server (Terminal 1)

1.  **Navigate to the Backend Folder:**
    After cloning or downloading this repository, open a terminal and navigate into the backend folder.
    ```bash
    cd pulp_function_backend
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the environment
    py -m venv venv

    # Activate it (on Windows)
    .\venv\Scripts\activate
    ```

3.  **Install All Dependencies:**
    This command installs all required packages from the `requirements.txt` file. This may take a few minutes.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will now be running at `http://127.0.0.1:8000`. The first startup is slow as it loads all the AI models into memory. **Keep this terminal window open.**

### ### Step 2: Launch the Frontend

1.  Navigate to the `puppy_frontend` folder in your file explorer.
2.  Find the **`index.html`** file.
3.  **Double-click** the file to open it in your default web browser.

The application is now running. You can upload a leaf image to get an instant diagnosis and use the microphone to ask follow-up questions.
