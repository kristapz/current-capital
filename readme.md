# Current Capital

A Python pipeline to process SEC filings and generate structured insights using BigQuery, Vertex AI, and OpenAI.

## 📌 Overview

This project extracts meaningful “facts” from SEC filings (e.g., 8-K, 10-K documents), classifies them into research fields, generates embeddings using both Google Vertex AI and OpenAI embeddings, and stores the results in [BigQuery](https://cloud.google.com/bigquery) with a normalized schema.

---

## ✨ Features

- **Automated data ingestion** from SEC filings using `scraper/` scripts  
- **Prompt-driven insight extraction** via OpenAI (`o1-mini` model)  
- **Dual embedding pipelines:**  
  - Google Vertex AI embeddings  
  - OpenAI `text-embedding-3-large` embeddings  
- **Facts normalization** into BigQuery rows with metadata (`row_number`, `link`, `field`, `processed_at`)  
- **Retry logic** with Tenacity for robust API calls  
- **Modular and extensible design** for adding new data sources or models  

---

## 🛠 Architecture

```text
[ SEC Filings ] 
       ↓ scraper/
[ Article + Link ] 
       ↓ main.py & process_article_with_openai()
[ Insight Extraction + Field Classification ]
       ↓ Embedding ⇄ Vertex AI and OpenAI
       ↓ Row Assembly + Validation
       ↓ Insert into BigQuery "papers" table
