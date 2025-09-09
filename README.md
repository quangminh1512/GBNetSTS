# GBNet-STS: A Graph-Based Model for Semantic Textual Similarity

## 📌 Overview
Semantic Textual Similarity (STS) is a core problem in NLP with wide-ranging applications such as machine translation, speech recognition, question answering, and text summarization.  
This repository contains the implementation of **GBNet-STS**, a **Graph-Based Network** designed to measure semantic similarity between Vietnamese sentence pairs.

Unlike traditional approaches that rely only on embeddings or lexical overlap, GBNet-STS integrates **multiple semantic layers**:
- **Cosine Similarity** (distributional semantics with embeddings)  
- **Semantic Longest Common Subsequence (SemLCS)**  
- **Semantic Jaccard Similarity (SemJaccard)**  

By combining these complementary perspectives in a unified graph structure, GBNet-STS achieves **state-of-the-art performance** in Vietnamese STS tasks.

---

## 🚀 Key Contributions
1. **Semantic-LCS**: Extends traditional LCS by allowing semantically related words (synonyms, embedding similarity) to align.  
2. **Semantic-Jaccard**: Extends Jaccard by incorporating semantic matching and IDF weighting.  
3. **Graph-Based Multi-Layer Model**: Jointly models lexical, syntactic, and embedding similarities.  
4. **ViSTS Dataset**: A benchmark dataset of **600 Vietnamese sentence pairs**, carefully curated and manually annotated.  

---

## 📊 Experimental Results
- GBNet-STS consistently outperforms baseline embedding-based methods (e.g., PhoBERT, Qwen, LLaMA).  
- Achieved **higher Pearson and Spearman correlations** with human-annotated similarity scores.  
- Robust against paraphrasing, synonym usage, and syntactic variations.

---

## 📂 Dataset & Source Code

### 📊 Dataset
- **`ViSentSim-600 - Standard-raw.xlsx`**  
  - Raw dataset containing 100 initial sentence pairs (not yet expert-reviewed).  
- **`ViSentSim-600 - Standard-reviewed.xlsx`**  
  - Expert-reviewed dataset with 100 validated sentence pairs.  

> ⚠️ Note: Both files are subsets of the full **ViSTS (600 sentence pairs)** dataset, provided here for demonstration and reproducibility.

---

### 💻 Code
- **`GBNetSTS.py`**  
  - Main Python file to run the **Graph-Based Semantic Textual Similarity (GBNet-STS)** model. 
