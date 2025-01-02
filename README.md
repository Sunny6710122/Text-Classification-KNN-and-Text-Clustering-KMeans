# Text Classification (KNN) and Text Clustering (KMeans) 📚🔍

This repository contains the implementation of **Text Classification** using **K-Nearest Neighbors (KNN)** and **Text Clustering** using **K-Means**, integrated with a **Vector Space Model (VSM)** for efficient text processing and retrieval. The project also includes a user-friendly **Streamlit GUI** for easy interaction.

---

## Running the GUI Application 🚀

Follow these steps to run the Streamlit GUI application:

### 1. Navigate to the Application Directory
- Open a command prompt or terminal.
- Change the directory to where your python file is located.

### 2. Run the Application
- Execute the following command:
  ```bash
  streamlit run app.py
  ```

### 3. Access the Application
- Open a web browser and go to the URL provided by Streamlit.

### 4. Interacting with the Application
- **Run KNN**: Executes the K-Nearest Neighbors (KNN) algorithm for document classification and retrieval. Displays:
  - Train and test documents after splitting.
  - Predicted labels and evaluation metrics.
- **Run K-Means**: Executes the K-Means clustering algorithm for document clustering. Shows:
  - Random seeds used.
  - Final clusters and evaluation metrics (including a silhouette score graph).

---

## VSM Model Overview 🧠

This project implements a **basic Vector Space Model (VSM)** for text processing and retrieval using **natural language processing (NLP)** techniques. It computes **Term Frequency-Inverse Document Frequency (TF-IDF)** scores, critical for ranking and retrieving documents based on their relevance to a query.

### Features 🌟

- **Text Preprocessing**: 🛠️ Cleans text, stems using the PorterStemmer, and removes stopwords.
- **TF-IDF Calculation**: 📊 Computes TF-IDF scores for terms in the corpus.
- **Query Processing**: 🔍 Processes text queries and computes their vector representation based on indexed terms.
- **Relevance Scoring**: 📈 Scores and ranks documents using cosine similarity to the query.

---

## Modules Required 🛠️

- **`numpy`**: For handling large arrays and matrices.
- **`json`**: For storing and retrieving TF-IDF vectors in JSON format.
- **`nltk`**: For stemming words using the PorterStemmer.
- **`re`**: For regular expression operations during text processing.
- **`streamlit`**: To run the `vsm_gui.py` GUI application.

---

## File Structure 📂

- **`Stopword-List.txt`**: Contains a list of stopwords.
- **`ResearchPapers/`**: Directory containing text documents in `.txt` format, each with a unique identifier

---

Explore and experiment with **Text Classification and Clustering** using this powerful toolkit! 🚀✨
