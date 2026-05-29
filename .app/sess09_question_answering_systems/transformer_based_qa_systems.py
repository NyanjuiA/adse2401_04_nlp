"""
=============================================================================================================
Python script to demonstrates Transformer-based Question Answering System.
=============================================================================================================
This program demonstrates Transformer-based Question Answering (QA) System using information about
tourist destinations in Kenya.

The system performs the following tasks:
    1. Load tourism data from a JSON file
    2. Convert the data into readable text contexts
    3. Uses TF-IDF retrieval to identify the most revelant context
    4. Uses a Transformer-based Question Answering model to extract an answer
    5. Displays the answer in the console

Dataset location:
    files/kenya_tourism.json

Requirements:
    !pip install transformers torch scikit-learn


Author: Nyanjui
Date: 29 May 2026
"""
# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------------
DATASET_FILE = Path('../files/kenya_tourism.json')

MODEL_NAME = "distilbert-base-cased-distilled-squad"

# --------------------------------------------------------------------------------
# 2. Data loading functions
# --------------------------------------------------------------------------------
def load_dataset(file_path):

    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    return data

def build_contexts(dataset):

    contexts = []
    site_names = []

    sites = dataset["sites"]

    for site in sites:

        name = site["name"]
        category = site["category"]
        region = site["region"]
        description = site["description"]
        best_time = site["best_time_to_visit"]

        highlights =  ", ".join(site["highlights"])
        activities = ", ".join(site["activities"])

        accessibility = site["accessibility"]

        context = f"""
        {name} is a {category} destination located in {region}.
        
        Description:
        {description}
        
        Best time to visit:
        {best_time}
        
        Highlights:
        {highlights}
        
        Activities:
        {activities}
        
        Accessibility:
        {accessibility}
        """

        context = context.strip()

        contexts.append(context)
        site_names.append(name)

    return contexts, site_names

# --------------------------------------------------------------------------------
# 3. Retrieval Functions
# --------------------------------------------------------------------------------
def create_tfidf_matrix(contexts):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(contexts)
    return vectorizer, tfidf_matrix

def retrieve_best_context(question,vectorizer,tfidf_matrix,context,site_names):

    question_vector = vectorizer.transform([question])

    similarity_scores = cosine_similarity(question_vector,tfidf_matrix)

    best_index = similarity_scores.argmax()

    best_context = context[best_index]
    best_site = site_names[best_index]

    similarity_score = similarity_scores[0][best_index]
    return best_context,best_site,similarity_score

# --------------------------------------------------------------------------------
# 4. Question Answering Functions
# --------------------------------------------------------------------------------
def load_qa_pipeline():

    qa_pipeline = pipeline("question-answering", model=MODEL_NAME)
    return qa_pipeline

def answer_question(question, context, qa_pipeline):

    result = qa_pipeline(question=question, context=context)

    return result

# --------------------------------------------------------------------------------
# 5. Main Execution Function
# --------------------------------------------------------------------------------
def main() -> None:

    print("=" * 78)
    print(" KENYA TOURISM TRANSFORMER QUESTION ANSWERING SYSTEM")
    print("=" * 78)

    print("\nLoading dataset...")

    dataset = load_dataset(DATASET_FILE)

    print("Building tourism contexts...")

    contexts, site_names = build_contexts(dataset)

    print("Creating TF-IDF retrieval index...")

    vectorizer, tfidf_matrix = create_tfidf_matrix(contexts)

    print("Loading Transformer QA model...")
    print("Please wait on first execution...\n")

    qa_pipeline = load_qa_pipeline()

    print("System ready.\nType 'exit' or 'quit' to quit.")

    while True:

        print("=" * 78)
        question = input("Kindly ask a Kenyan tourism question: \n>_")

        if question.lower() == "exit" or question.lower() == "quit":
            print("\n" + "=" * 78)
            print("\n End of Transformer Question Answering Demonstration...")
            print("=" * 78)
            break

        if len(question.strip()) == 0:
            print("Please enter a valid question")
            continue

        # ------------------------------------------------------------------
        # Retrieve relevant context
        # ------------------------------------------------------------------
        best_context, best_site, similarity_score = (
            retrieve_best_context(question, vectorizer, tfidf_matrix, contexts, site_names)
        )

        # ------------------------------------------------------------------
        # Generate answer using Transformer
        # ------------------------------------------------------------------
        result = answer_question(question, best_context, qa_pipeline)

        answer = result["answer"]
        confidence = result["score"]

        # ------------------------------------------------------------------
        # Display results
        # ------------------------------------------------------------------
        print("\nMost Relevant Site:")
        print(best_site)

        print("\nAnswer:")
        print(answer)

        print("\nTransformer Confidence Score:")
        print(f"{confidence:.3f}")

        print(f"\nTF-IDF Similarity Score :"
              f"{similarity_score:.3f}")

        print("\nRetrieved Context:")
        print(best_context)


# --------------------------------------------------------------------------------
# 6. Run the script by invoking it's main() function
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()