🔍 Text Clustering with TF-IDF
Unsupervised text clustering using TF-IDF vectorization and cosine similarity — grouping similar text entries from Excel datasets without labeled data.
What It Does
Reads two Excel files, converts text columns into TF-IDF vectors, and clusters entries based on cosine similarity scores. Useful for finding patterns in unstructured text data.
Use Cases
Grouping similar customer feedback or survey responses
Detecting duplicate or near-duplicate records
Organizing research notes or document collections

How to Run
git clone https://github.com/mzhbr/clustering-data-with-TF-IDF-modeliztion
cd clustering-data-with-TF-IDF-modeliztion

pip install pandas scikit-learn numpy openpyxl

python clustering.py

How It Works
1.Load Excel files with pandas
2.Vectorize text columns using TfidfVectorizer
3.Compute pairwise cosine similarity
4.Group entries that exceed the similarity threshold

Key Concept: Why TF-IDF?
TF-IDF (Term Frequency–Inverse Document Frequency) weights words by how unique they are to a document. Common words like "the" get low scores; distinctive words get high scores — making it effective for finding semantically similar texts.

What I Learned
How TF-IDF works under the hood
Unsupervised similarity-based clustering (no labels needed)
Handling real-world Excel data with pandas
