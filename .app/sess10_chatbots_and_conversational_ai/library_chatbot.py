"""
=============================================================================================================
Python script to demonstrates a library chatbot System.
=============================================================================================================
This program demonstrates a Library chatbot System using book details from a json file ( library_books.json)

features:
- Loads a JSON book catalogue
- Creates a text representation of the book details
- Generates embeddings using all-MiniLM-L6-V2
- Performs semantic search with cosine similarity
- Supports:
    * Semantic search
    * Recommendations
    * Author search
    * Category search
    * Availability checks
- Uses a command-line chat interface

Dataset location:
    files/library_books.json

Requirements:
    !pip install sentence-transformers pandas numpy


Author: Nyanjui
Date: 04 June 2026
"""
# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------
from __future__ import annotations

import json, re, sys
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------
# 1. LibraryChatbot class
# --------------------------------------------------------------------------------
class LibraryChatbot:

    def __init__(self, json_path: str) -> None:

        self.json_path = Path(json_path)

        self.df = self._load_catalogue()

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Creating searchable text...")
        self.df["search_text"] = self.df.apply(
            self._create_search_text,
            axis=1,
        )

        print("Generating book embeddings...")
        self.embeddings = self.model.encode(
            self.df["search_text"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        print(f"Loaded {len(self.df)} books.\n")

    def _load_catalogue(self) -> pd.DataFrame:

        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            return pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Error: Catalogue not found: {self.json_path}")
            sys.exit(1)
        except json.decoder.JSONDecodeError:
            print(f"Error: Invalid JSON format.")
            sys.exit(1)
        except Exception as exc:
            print(f"Unexpected error loading catalogue: {exc}")
            sys.exit(1)

    @staticmethod
    def _create_search_text(row: pd.Series) -> str:

        keywords = row.get("keywords", [])

        if isinstance(keywords, list):
            keyword_text = " ".join(map(str, keywords))
        else:
            keyword_text = str(keywords)

        return (
            f"Title: {row.get('title', '')}."
            f"Author: {row.get('author', '')}."
            f"Category: {row.get('category', '')}."
            f"Description: {row.get('description', '')}."
            f"Reading Level: {row.get('reading_level', '')}."
            f"Keywords: {keyword_text}."
        )

    @staticmethod
    def cosine_similarity(
            query_embeddings: np.ndarray,
            document_embeddings: np.ndarray) -> np.ndarray:

        query_norm = np.linalg.norm(query_embeddings)

        document_norms = np.linalg.norm(document_embeddings, axis=1)

        similarities = (
                np.dot(document_embeddings, query_embeddings) / (document_norms * query_norm)
        )

        return similarities

    def semantic_search(self, query: str, top_k : int = 5,) -> pd.DataFrame:

        query_embeddings = self.model.encode(query, convert_to_numpy=True)

        similarities = self.cosine_similarity(
            query_embeddings,self.embeddings
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]

        return results

    def search_by_author(self, author_name:str) -> pd.DataFrame:

        mask = self.df["author"].astype(str).str.contains(
            author_name,
            case=False,
            na=False,
        )

        return self.df[mask]

    def search_by_category(self, category: str) -> pd.DataFrame:

        mask = self.df["category"].astype(str).str.contains(
            category,
            case=False,
            na=False,
        )

        return self.df[mask]

    def check_availability(self, title:str) -> Optional[pd.Series]:

        mask = self.df["title"].astype(str).str.lower() == title.lower()

        matches = self.df[mask]

        if matches.empty:
            return None

        return matches.iloc[0]

    @staticmethod
    def display_books(results: pd.DataFrame) -> None:

        if results.empty:
            print("\nNo matching books found.\n")
            return

        print("\nResults:\n")

        for _, row in results.iterrows():
            print(f"Title: {row.get('title', '')}")
            print(f"Author: {row.get('author', 'unknown')}")
            print(f"Category: {row.get('category', 'unknown')}")
            print(f"Year: {row.get('published_year', 'unknown')}")

            if "similarity" in row:
                print(f"Similarity: {row['similarity']:.3f}")

            print("-" * 60)
            print()

    def handle_author_query(self,query:str) -> None:

        match = re.search(
            r"Books by: (.+)",query,re.IGNORECASE
        )

        if not match:
            print("Please enter a valid author name.")
            return

        author = match.group(1).strip()

        results = self.search_by_author(author)
        self.display_books(results)

    def handle_category_query(self,query:str) -> None:

        match = re.search(
            r"show (.+?) books",query,re.IGNORECASE
        )

        if not match:
            print("Please enter a valid category name.")
            return
        category = match.group(1).strip()

        results = self.search_by_category(category)
        self.display_books(results)

    def handle_availability_query(self,query:str) -> None:

        match = re.search(
            r"is (.+) available",query,re.IGNORECASE
        )

        if not match:
            print("Please enter a valid book title.")
            return

        title = match.group(1).strip()

        book = self.check_availability(title)

        if book is None:
            print(f"\nNo book titled '{title}' found.\n")
            return

        print(f"\nAvailability Information:")
        print(f"Title: {book['title']}")
        print(f"Available: {book['availability']}")
        print(f"Copies Available: {book['copies_available']}")
        print(f"Shelf Location: {book['shelf_location']}")

    def handle_semantic_query(self, query:str) -> None:

        results = self.semantic_search(query)
        self.display_books(results)

    def process_query(self, query:str) -> None:

        query_lower = query.lower()

        if query_lower.startswith("books by"):
            self.handle_author_query(query)

        elif query_lower.startswith("show"):
            self.handle_category_query(query)

        elif query_lower.startswith("is"):
            self.handle_availability_query(query)

        elif (
                query_lower.startswith("find") or query_lower.startswith("recommend")
        ):
            self.handle_semantic_query(query)

        else:
            print("\nSorry, I did not understand your query.\n"
                  "Sample queries to use:\n"
                  " find books about astronomy"
                  "\n   recommend books about programming"
                  "\n   books by Harper Lee"
                  "\n   show fiction books"
                  "\n   is the The Great Gatsby available\n"
                  )

    def chat(self) -> None:

        print("=" * 60)
        print("Library Search Assistant/Chatbot")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            try:
                query = input("Library Assistant/Chatbot >").strip()

                if not query:
                    continue

                if query.lower() in { "exit", "quit" }:
                    print("\b👋🏿Goodbye.")
                    break

                self.process_query(query)

            except KeyboardInterrupt:
                print("\n\nSession Ended!")
                break

            except Exception as e:
                print(f"\nAn error occurred: {e}\n")

# --------------------------------------------------------------------------------
# 2. Main Execution Function
# --------------------------------------------------------------------------------
def main() -> None:

    json_path = "../files/library_books.json"

    # Instantiate a LibraryChatbot
    chatbot = LibraryChatbot(json_path)
    chatbot.chat()


# --------------------------------------------------------------------------------
# 3. Run the script by invoking it's main() function
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
