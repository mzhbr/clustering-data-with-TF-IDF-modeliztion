import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the first Excel file with Business codes and descriptions
df1 = pd.read_excel("E:/eslahi.xlsx")

# Read the second Excel file with Business licenses
df2 = pd.read_excel("E:/11.xlsx")

# Fill NaN values in both dataframes with an empty string
df1['companyName'] = df1['companyName'].fillna('')
df2['RECEIVERADDRESS'] = df2['RECEIVERADDRESS'].fillna('')

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Combine the business descriptions and licenses into a single list
combined_text = list(df1['companyName']) + list(df2['RECEIVERADDRESS'])

# Fit and transform the TF-IDF Vectorizer on the combined text
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

# Calculate the cosine similarity between the TF-IDF vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get the indices of the business descriptions and licenses in the cosine similarity matrix
desc_indices = range(len(df1))
lic_indices = range(len(df1), len(df1) + len(df2))

# Set a higher cosine similarity threshold to consider more common words
cosine_threshold = 0.5  # Adjust this threshold as needed

# Create a dictionary to store the mappings between business descriptions and licenses
mapping_dict = {}

# Iterate over the business descriptions and find the best matching license based on cosine similarity
for desc_idx in desc_indices:
    matches = [(lic_idx, score) for lic_idx, score in zip(lic_indices, cosine_sim[desc_idx][lic_indices]) if score >= cosine_threshold]
    if matches:
        best_match_idx, best_score = max(matches, key=lambda x: x[1])
        mapping_dict[df1.loc[desc_idx, 'BC']] = (df1.loc[desc_idx, 'companyName'], df2.loc[best_match_idx - len(df1), 'RECEIVERADDRESS'], best_score)

# Create a new dataframe from the mapping dictionary
mapped_data = [(k, v[0], v[1], v[2]) for k, v in mapping_dict.items()]
mapped_df = pd.DataFrame(mapped_data, columns=['nationalCode', 'companyName', 'Mapped Barname', 'Cosine Similarity'])

# Save the mapped data to a new Excel file
mapped_df.to_excel('mapped_data_with_adress.xlsx', index=False)