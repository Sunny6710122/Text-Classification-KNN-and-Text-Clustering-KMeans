import os  # Importing the operating system module
import numpy as np  # Importing NumPy for numerical computing
import math  # Importing the math module for mathematical operations
import re  # Importing the re module for regular expressions
import time  # Importing the time module for time-related functions
from sklearn.metrics import adjusted_rand_score, silhouette_score  # Importing functions for evaluation metrics
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import tkinter as tk  # Importing tkinter for creating GUI
from tkinter import scrolledtext  # Importing scrolledtext for creating scrollable text areas
from tkinter import messagebox  # Importing messagebox for displaying message boxes


def load_document_tfidf(filename):
    # Function to load TF-IDF data from a file
    doc_tfidf = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            doc_line = lines[i].strip()
            if doc_line.startswith('Document:'):
                doc_id = int(doc_line.split(':')[1].strip())
                tfidf_line = lines[i+1].strip()
                tfidf_vector = [float(x) for x in tfidf_line.split(':')[1].split(',')]
                doc_tfidf[doc_id] = tfidf_vector
                i += 2
            else:
                i += 1
    return doc_tfidf

def calculate_euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.around(np.sqrt(np.sum(np.square(vec1 - vec2))), 3)

def calculate_cluster_centroid(cluster):
    # Function to calculate the centroid of a cluster
    cluster_size = len(cluster)
    centroid_vector = np.zeros(len(next(iter(doc_tfidf_data.values()))))
    for doc_id in cluster:
        centroid_vector += doc_tfidf_data[doc_id]
    try:
        centroid_vector *= (1 / cluster_size)
    except:
        centroid_vector *= (1)
    return centroid_vector

def initialize_random_centroids(data, k):
    # Function to randomly initialize K centroids
    np.random.seed(int(time.time()))  
    random_indices = np.random.choice(len(data), size=k, replace=False)
    centroids = np.array([doc_tfidf_data[data[index]] for index in random_indices])
    return centroids

def assign_documents_to_clusters(docs, centroids, k=5):
    # Function to assign documents to clusters based on centroids
    clusters = {i: [] for i in range(k)}
    for doc_id in docs:
        distances = [calculate_euclidean_distance(doc_tfidf_data[doc_id], centroid) for centroid in centroids]
        min_distance_centroid_id = np.argmin(distances)
        clusters[min_distance_centroid_id].append(doc_id)
    return clusters

def update_centroids_position(clusters, centroids):
    # Function to update centroids' positions based on cluster assignments
    for cluster_id, cluster_docs in clusters.items():
        centroids[cluster_id] = calculate_cluster_centroid(cluster_docs)
    return centroids

def calculate_residual_sum_of_squares(clusters, centroids):
    # Function to calculate Residual Sum of Squares (RSS)
    rss = 0
    for cluster_id in range(len(clusters)):
        for doc_id in clusters[cluster_id]:
            rss += np.around(np.square(calculate_euclidean_distance(doc_tfidf_data[doc_id], centroids[cluster_id])), 3)
    return rss

def perform_kmeans_clustering(docs, k):
    # Function to perform K-means clustering
    clusters = []
    counter = 0
    centroids = initialize_random_centroids(docs, k)
    rss = float("inf")
    new_rss = 1000000
    print("Initial Data: \nRSS: {}\nClusters: {}\nCentroids: {}".format(rss, clusters, centroids))
    while new_rss > 0 and new_rss < rss:
        counter += 1
        rss = new_rss
        clusters = assign_documents_to_clusters(docs, centroids, k)
        centroids = update_centroids_position(clusters, centroids)
        new_rss = calculate_residual_sum_of_squares(clusters, centroids)
        print("Iteration # {}: \nRSS: {}\nClusters: {}\nCentroids: {}".format(counter, new_rss, clusters, centroids))
    return clusters

def perform_kmeans_clustering1(docs, k):
    # Function to perform K-means clustering
    clusters = []
    counter = 0
    centroids = initialize_random_centroids(docs, k)
    rss = float("inf")
    new_rss = 1000000
    while new_rss > 0 and new_rss < rss:
        counter += 1
        rss = new_rss
        clusters = assign_documents_to_clusters(docs, centroids, k)
        centroids = update_centroids_position(clusters, centroids)
        new_rss = calculate_residual_sum_of_squares(clusters, centroids)
    return clusters

def calculate_purity(golden_clusters, test_clusters):
    # Function to calculate the purity of clusters
    total_instances = sum(len(cluster) for cluster in test_clusters.values())
    total_correct = 0
    
    for test_cluster in test_clusters.values():
        max_common = 0
        for golden_cluster in golden_clusters.values():
            common = len(set(test_cluster).intersection(golden_cluster))
            if common > max_common:
                max_common = common
        total_correct += max_common
        
    purity = total_correct / total_instances
    return total_correct, purity

def calculate_rand_index(golden_clusters, test_clusters):
    # Function to calculate the Rand Index
    golden_labels = []
    test_labels = []
    
    for golden_cluster_id, golden_cluster in golden_clusters.items():
        golden_labels.extend([golden_cluster_id] * len(golden_cluster))
        
    for test_cluster_id, test_cluster in test_clusters.items():
        test_labels.extend([test_cluster_id] * len(test_cluster))
    
    rand_index = adjusted_rand_score(golden_labels, test_labels)
    return rand_index

def calculate_silhouette_score(test_clusters):
    # Function to calculate the Silhouette Score
    labels = []
    for cluster_id, cluster in test_clusters.items():
        labels.extend([cluster_id] * len(cluster))
    
    labels = np.array(labels)
    data = np.array(list(doc_tfidf_data.values()))
    
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg

def plot_silhouette_score(k_values, silhouette_scores):
    # Function to plot Silhouette Score against the number of clusters
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.show()


doc_tfidf_data = load_document_tfidf("tfidf_results.txt")
research_papers = [1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
docs_array = np.array(research_papers)  
golden_clusters = {
    0: [1, 2, 3, 7],
    1: [8, 9, 11],
    2: [12, 13, 14, 15, 16],
    3: [17, 18, 21],
    4: [22, 23, 24, 25, 26]
}

def perform_simple_query():
    # Function to perform the simple query
    query = kValue.get()
    clusters_result = perform_kmeans_clustering(docs_array,int(query))
    total_correct, purity = calculate_purity(golden_clusters, clusters_result)
    rand_index = calculate_rand_index(golden_clusters, clusters_result)
    silhouette_avg = calculate_silhouette_score(clusters_result)
    k_values = range(2, 11)
    silhouette_scores = []
    for k in k_values:
        clusters = perform_kmeans_clustering1(docs_array, k)
        silhouette_avg = calculate_silhouette_score(clusters)
        silhouette_scores.append(silhouette_avg)
    display_results(clusters_result,total_correct,purity,rand_index,silhouette_avg,k_values,silhouette_scores)

def display_results(clusters_result,total_correct,purity,rand_index,silhouette_avg,k_values,silhouette_scores):
    # Function to display the clustering results
    result_text.delete(1.0, tk.END)
    if clusters_result:
        count = 0
        result_text.insert(tk.END, "Clusters: \n")
        result_text.insert(tk.END, f"{clusters_result}\n\n\n")
        result_text.insert(tk.END, f"Total Correct Assignments: {total_correct}\n")
        result_text.insert(tk.END, f"Purity: {purity}\n")
        result_text.insert(tk.END, f"Rand Index: {rand_index}\n")
        result_text.insert(tk.END, f"Silhouette Score: {silhouette_avg}\n")
        plot_silhouette_score(k_values, silhouette_scores)
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
