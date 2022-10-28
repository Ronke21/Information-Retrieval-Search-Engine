# Information-Retrieval-Search-Engine

The project is a search engine in Hebrew and English. Engine operation stages:
  • Depending on the query entered, the engine diagnoses the language and chooses which file system to use (text files in Hebrew or English).
  • The file system is pre-processed - all documents undergo initial processing to remove special characters, punctuation, and stop words and finally transition to a base form (lemmatization).
  • Also, each file is assigned a BOW vector according to the dictionary of the entire corpus, as well as a representative vector on the doc2vec model that was trained on the corpus ahead of time.
  • The user can choose whether to use the existing file system or update it by going through all the documents and renderings again and then training the models (a process that takes a lot of time).
  • After entering the query, it goes through the same processing as the documents and is assigned the required vectors.
  • The system calculates the similarity values between each document and the query according to all the following indicators:
      o Similarity between BOW vectors based on scalar multiplication.
      o Jaccard index similarity between BOW vectors.
      o Similarity based on the TF-IDF method with the BM25 transformation.
      o Vector space-based similarity trained on the doc2vec model.
      o Similarity is based on all the indices.
  • The system asks the user which similarity function to use, and prints the first 30 documents according to this ranking (descending order).
  • The user is then given the option to enter another query or finish.
  
![image](https://user-images.githubusercontent.com/73187207/198517257-1cee1e8f-80ae-4b32-acfc-17520fb75c87.png)

The system was written in Python language version 3.8.

To run the system:
  • All imported packages must be installed.
  • Paths to the file systems must be created in Hebrew and English as they appear in the fixed variables at the top of the main code (after the imports) or for exchanges replace the path with the code.
  • Install separately a yap application (installation instructions here) and run it as an HTTP server on port 8000.
  • Then run the main file only.
