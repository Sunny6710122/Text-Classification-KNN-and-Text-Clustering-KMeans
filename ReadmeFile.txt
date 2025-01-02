# Information Retrieval System

This project consists of two parts: a document classification system using k-NN and a text clustering system using K-means.

## Part 1: Document Classification using k-NN

### Description
This part of the project implements a document classification system using the k-nearest neighbors (k-NN) algorithm. It loads TF-IDF data from files, splits documents into training and testing sets, calculates precision, recall, F1-score, and accuracy, and evaluates the classifier's performance. The user can input the value of k to evaluate the classifier.

### Usage
1. Make sure you have the necessary data files: "IvertedIndexofVSM.txt" and "IDF.txt".
2. Run the script.
3. Enter the value of k when prompted.
4. Click the "Evaluate" button to see the classification results.

## Part 2: Text Clustering using K-means

### Description
This part of the project implements a text clustering system using the K-means algorithm. It loads TF-IDF data from a file, performs K-means clustering, calculates purity, Rand Index, and silhouette score to evaluate the clustering results. The user can input the value of k to perform clustering.

### Usage
1. Make sure you have the necessary data file: "tfidf_results.txt".
2. Run the script.
3. Enter the value of k when prompted.
4. Click the "Evaluate" button to see the clustering results.

## Requirements
- Python 3.x
- NumPy
- scikit-learn
- tkinter (for GUI)

## Additional Notes
- Ensure that the data files are in the same directory as the scripts.
- The performance of the algorithms may vary depending on the input data and parameters.
