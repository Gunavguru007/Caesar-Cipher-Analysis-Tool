# Caesar Cipher Analysis Tool 🔐

A comprehensive Python tool for encrypting, decrypting, and analyzing text using the Caesar cipher. Built with Streamlit for an interactive web interface, this application provides features such as:

- Text encryption/decryption
- Frequency analysis (unigrams, bigrams, trigrams)
- Brute force attack with dictionary matching
- Export options (CSV, PDF, PNG)

## 📸 Screenshot
![Caesar Cipher Tool UI](screenshot.png) *(Add a screenshot of your running app here)*

## 🧰 Features

### 1. Caesar Cipher Encrypt/Decrypt
- Supports custom alphabets and case sensitivity.
- Handles unknown characters via ignore/remove/replace options.
- Visualizes character mapping in a table.

### 2. Frequency Analysis
- Compares ciphertext frequencies with expected language patterns (default: English).
- Auto-detects possible key based on chi-squared comparison.
- Displays frequency charts using Plotly.

### 3. Brute Force Attack
- Tries all keys in a given range.
- Uses NLTK English words or a custom dictionary.
- Shows percentage match and best decryption result.

## 🛠️ Requirements

Make sure to install the required dependencies listed in `requirements.txt`.

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/caesar-cipher-tool.git 
   cd caesar-cipher-tool
