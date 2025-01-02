import math
import json
import random
from collections import defaultdict
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox

# Load the inverted index and IDF data
with open("IvertedIndexofVSM.txt", "r") as file:
    Inverted_Index = json.load(file)
with open("IDF.txt", "r") as file1:
    IDF = json.load(file1)

# Define document classes and their corresponding documents
classes = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}
# Function to calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vector1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vector2))
    return dot_product / (magnitude1 * magnitude2)

# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))

# Modified function to classify a document using k-NN with Euclidean distance
def classify_document(document_vector, train_documents, k):
    nearest_neighbors = []
    for doc_id, doc_class in train_documents:
        class_vector = [Inverted_Index.get(term, {}).get(str(doc_id), 0) * IDF.get(term, 0) for term in Inverted_Index.keys()]
        distance = euclidean_distance(document_vector, class_vector)
        nearest_neighbors.append((doc_id, doc_class, distance))
    nearest_neighbors.sort(key=lambda x: x[2])
    top_k_neighbors = nearest_neighbors[:int(k)]

    # Assign label/class based on cosine similarity
    class_votes = defaultdict(int)
    for doc_id, doc_class, _ in top_k_neighbors:
        class_vector = [Inverted_Index.get(term, {}).get(str(doc_id), 0) * IDF.get(term, 0) for term in Inverted_Index.keys()]
        similarity = cosine_similarity(document_vector, class_vector)
        class_votes[doc_class] += similarity
    return max(class_votes, key=class_votes.get)


# Function to split documents into training and testing sets
def split_train_test_documents(classes, train_ratio=0.7):
    train_documents = []
    test_documents = []
    for class_name, class_documents in classes.items():
        random.shuffle(class_documents)
        split_index = int(train_ratio * len(class_documents))
        train_documents.extend((doc_id, class_name) for doc_id in class_documents[:split_index])
        test_documents.extend((doc_id, class_name) for doc_id in class_documents[split_index:])
    return train_documents, test_documents

# Function to calculate precision
def precision(true_labels, predicted_labels):
    true_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    predicted_positive = len(predicted_labels)
    return true_positive / predicted_positive if predicted_positive != 0 else 0

# Function to calculate recall
def recall(true_labels, predicted_labels):
    true_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    actual_positive = len(true_labels)
    return true_positive / actual_positive if actual_positive != 0 else 0

# Function to calculate F1-score
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Function to calculate accuracy
def accuracy(true_labels, predicted_labels):
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total = len(true_labels)
    return correct / total if total != 0 else 0

# Main function for evaluation
def evaluate(train_documents, test_documents,k):
    true_labels = [label for _, label in test_documents]
    predicted_labels = []
    for doc_id, _ in test_documents:
        document_vector = [Inverted_Index.get(term, {}).get(str(doc_id), 0) * IDF.get(term, 0) for term in Inverted_Index.keys()]
        predicted_label = classify_document(document_vector, train_documents,k)
        predicted_labels.append(predicted_label)

    prec = precision(true_labels, predicted_labels)
    rec = recall(true_labels, predicted_labels)
    f1 = f1_score(prec, rec)
    acc = accuracy(true_labels, predicted_labels)

    return [prec, rec, f1, acc],predicted_labels

# Split documents into training and testing sets

train_documents, test_documents = split_train_test_documents(classes)

# Evaluate the classifier
# precision, recall, f1_score, accuracy = evaluate(train_documents, test_documents,3)

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1_score)
# print("Accuracy:", accuracy)
evaluation = ["Precision","Recall","F1-score","Accuracy"]





# from your_script_file_name import Simple_queryy, Proximity_Queryy
def perform_simple_query():
    query = kValue.get()
    results,predicted = evaluate(train_documents, test_documents,query)
    display_results(results,predicted)

def display_results(results,predicted):
    result_text.delete(1.0, tk.END)
    if results:
        count = 0
        for result in results:
            result_text.insert(tk.END, f"{evaluation[count]}: ")
            result_text.insert(tk.END, f"{result}\n")
            count+= 1
        result_text.insert(tk.END,f"Test_Documents with Predicted Labels: \n")
        count = 0
        for test_d in test_documents:
            result_text.insert(tk.END, f"{test_d[0]}: ")
            result_text.insert(tk.END, f"{predicted[count]}\n")
            count +=1
    else:
        result_text.insert(tk.END, "No results found.")

# Create main window
root = tk.Tk()

root.title("Information Retrieval System")

# Create input fields and buttons
simple_query_label = tk.Label(root, text="Enter the Value of K:")
simple_query_label.grid(row=0, column=0, padx=5, pady=5)

kValue = tk.Entry(root, width=50)
kValue.grid(row=0, column=1, padx=5, pady=5)

simple_query_button = tk.Button(root, text="Evaluate", command=perform_simple_query)
simple_query_button.grid(row=0, column=2, padx=5, pady=5)

# Create text area to display results
result_text = scrolledtext.ScrolledText(root, width=80, height=20)
result_text.grid(row=2, columnspan=3, padx=5, pady=5)

# Run the main event loop
root.mainloop()



