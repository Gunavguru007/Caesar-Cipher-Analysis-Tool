import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import string
import csv
from io import StringIO, BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
from nltk.corpus import words as nltk_words
import nltk
from nltk.util import ngrams
import plotly.express as px

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Try to import Kaleido for Plotly image export
try:
    import kaleido
    KALEIDO_INSTALLED = True
except ImportError:
    KALEIDO_INSTALLED = False

# Default language frequencies (English)
LANGUAGE_FREQUENCIES = {
    'English': {
        'unigrams': {
            'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702,
            'f': 2.228, 'g': 2.015, 'h': 6.094, 'i': 6.966, 'j': 0.153,
            'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749, 'o': 7.507,
            'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327, 't': 9.056,
            'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974,
            'z': 0.074
        },
        'bigrams': {
            'th': 2.71, 'he': 2.33, 'in': 2.03, 'er': 1.78, 'an': 1.61,
            're': 1.41, 'nd': 1.38, 'at': 1.28, 'on': 1.27, 'nt': 1.24,
        },
        'trigrams': {
            'the': 1.81, 'and': 0.73, 'ing': 0.72, 'her': 0.56, 'hat': 0.56,
            'his': 0.55, 'tha': 0.52, 'ere': 0.42, 'for': 0.42, 'ent': 0.42,
        }
    },
    # Add more languages as needed
}

class CaesarCipher:
    def __init__(self, alphabet=None, case_sensitive=True, unknown_char_handling='ignore', replace_char=' '):
        self.alphabet = alphabet or string.ascii_uppercase
        self.case_sensitive = case_sensitive
        self.unknown_char_handling = unknown_char_handling
        self.replace_char = replace_char
        
    def encrypt(self, text, key):
        return self._transform(text, key)
    
    def decrypt(self, text, key):
        return self._transform(text, -key)
    
    def _transform(self, text, key):
        result = []
        alphabet_len = len(self.alphabet)
        
        for char in text:
            original_case = char
            if not self.case_sensitive:
                char = char.lower()
            
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_index = (index + key) % alphabet_len
                new_char = self.alphabet[new_index]
                
                if not self.case_sensitive and original_case.isupper():
                    new_char = new_char.upper()
                result.append(new_char)
            else:
                if self.unknown_char_handling == 'ignore':
                    result.append(char)
                elif self.unknown_char_handling == 'remove':
                    continue
                elif self.unknown_char_handling == 'replace':
                    result.append(self.replace_char)
        
        return ''.join(result)
    
    def get_mapping(self, key):
        mapping = {}
        alphabet_len = len(self.alphabet)
        for i, char in enumerate(self.alphabet):
            new_index = (i + key) % alphabet_len
            mapping[char] = self.alphabet[new_index]
        return mapping

class FrequencyAnalyzer:
    def __init__(self, language='English'):
        self.language = language
        self.language_freq = LANGUAGE_FREQUENCIES.get(language, LANGUAGE_FREQUENCIES['English'])
    
    def calculate_frequencies(self, text, n=1):
        text = text.lower()
        
        if n == 1:
            counter = Counter(text)
            total = sum(counter.values())
            return {char: (count / total) * 100 for char, count in counter.items() if char in self.language_freq['unigrams']}
        else:
            # Generate n-grams
            text_ngrams = ngrams(text, n)
            counter = Counter(text_ngrams)
            total = sum(counter.values())
            return {''.join(gram): (count / total) * 100 for gram, count in counter.items()}
    
    def compare_frequencies(self, text_freq, n=1):
        if n == 1:
            target_freq = self.language_freq['unigrams']
        elif n == 2:
            target_freq = self.language_freq['bigrams']
        elif n == 3:
            target_freq = self.language_freq['trigrams']
        else:
            return None
        
        # Calculate chi-squared statistic
        chi_squared = 0
        common_items = set(text_freq.keys()) & set(target_freq.keys())
        
        for item in common_items:
            expected = target_freq[item]
            observed = text_freq.get(item, 0)
            chi_squared += ((observed - expected) ** 2) / expected
        
        return chi_squared
    
    def find_best_key(self, ciphertext, max_key=26):
        best_key = 0
        best_score = float('inf')
        
        cipher = CaesarCipher()
        
        for key in range(max_key + 1):
            decrypted = cipher.decrypt(ciphertext, key)
            freq = self.calculate_frequencies(decrypted)
            score = self.compare_frequencies(freq)
            
            if score < best_score:
                best_score = score
                best_key = key
        
        return best_key, best_score

class BruteForcer:
    def __init__(self, dictionary=None):
        self.dictionary = set(dictionary or nltk_words.words())
    
    def brute_force_decrypt(self, ciphertext, key_range, cipher_alphabet=None):
        cipher = CaesarCipher(alphabet=cipher_alphabet)
        results = []
        
        for key in key_range:
            decrypted = cipher.decrypt(ciphertext, key)
            matches = self.count_dictionary_matches(decrypted)
            results.append({
                'key': key,
                'decrypted_text': decrypted,
                'matches': matches,
                'match_percentage': (matches / len(decrypted.split())) * 100 if decrypted.split() else 0
            })
        
        return results
    
    def count_dictionary_matches(self, text):
        words = text.lower().split()
        return sum(1 for word in words if word in self.dictionary)

class ReportGenerator:
    @staticmethod
    def generate_mapping_csv(mapping):
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Original', 'Mapped'])
        for orig, mapped in mapping.items():
            writer.writerow([orig, mapped])
        return output.getvalue()
    
    @staticmethod
    def generate_pdf(text, title="Caesar Cipher Result"):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, title)
        
        y_position = 700
        for line in text.split('\n'):
            if y_position < 50:
                c.showPage()
                y_position = 750
            c.drawString(100, y_position, line)
            y_position -= 15
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

# Streamlit UI
def main():
    st.set_page_config(page_title="Caesar Cipher Tool", layout="wide")
    
    st.title("🔐 Caesar Cipher Analysis Tool")
    st.markdown("""
    A comprehensive tool for encrypting, decrypting, and analyzing text using the Caesar cipher.
    """)
    
    if not KALEIDO_INSTALLED:
        st.warning("For full functionality including PNG exports, please install Kaleido: `pip install kaleido`")
    
    tab1, tab2, tab3 = st.tabs([
        "📌 Caesar Cipher (Encrypt/Decrypt)", 
        "📊 Frequency Analysis", 
        "🔍 Brute Force Attack"
    ])
    
    with tab1:
        render_caesar_tab()
    
    with tab2:
        render_frequency_tab()
    
    with tab3:
        render_brute_force_tab()

def render_caesar_tab():
    st.header("Caesar Cipher - Encrypt/Decrypt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio("Mode", ["Encrypt 🔒", "Decrypt 🔓"], horizontal=True, key="mode_radio")
        text = st.text_area("Input Text", height=150, key="caesar_text")
        
        alphabet_options = {
            "English (A-Z)": string.ascii_uppercase,
            "English (a-z)": string.ascii_lowercase,
            "Custom Alphabet": None
        }
        
        selected_alphabet = st.selectbox(
            "Alphabet", 
            list(alphabet_options.keys()),
            key="alphabet_select"
        )
        
        if selected_alphabet == "Custom Alphabet":
            custom_alphabet = st.text_input("Enter custom alphabet characters", 
                                          value="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                          key="custom_alphabet")
            alphabet = custom_alphabet
        else:
            alphabet = alphabet_options[selected_alphabet]
        
        key = st.number_input("Key", min_value=0, max_value=100, value=3, key="caesar_key")
        case_sensitive = st.checkbox("Case Sensitive", value=True, key="case_sensitive")
        
        unknown_char_handling = st.selectbox(
            "Unknown Character Handling",
            ["ignore", "remove", "replace"],
            index=0,
            key="unknown_char"
        )
        
        replace_char = " "
        if unknown_char_handling == "replace":
            replace_char = st.text_input("Replacement Character", max_chars=1, value=" ", key="replace_char")
    
    with col2:
        if text:
            cipher = CaesarCipher(
                alphabet=alphabet,
                case_sensitive=case_sensitive,
                unknown_char_handling=unknown_char_handling,
                replace_char=replace_char
            )
            
            if mode == "Encrypt 🔒":
                result = cipher.encrypt(text, key)
                mapping = cipher.get_mapping(key)
            else:
                result = cipher.decrypt(text, key)
                mapping = cipher.get_mapping(-key)
            
            st.text_area("Result", value=result, height=150, key="caesar_result")
            
            with st.expander("Character Mapping"):
                st.write("Each character is mapped as follows:")
                mapping_df = pd.DataFrame(list(mapping.items()), columns=["Original", "Mapped"])
                st.table(mapping_df)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="Copy to Clipboard",
                        data=result,
                        file_name="caesar_result.txt",
                        mime="text/plain",
                        key="copy_clipboard"
                    )
                with col2:
                    csv_data = ReportGenerator.generate_mapping_csv(mapping)
                    st.download_button(
                        label="Export Mapping to CSV",
                        data=csv_data,
                        file_name="caesar_mapping.csv",
                        mime="text/csv",
                        key="export_csv"
                    )
                with col3:
                    pdf_data = ReportGenerator.generate_pdf(result, "Caesar Cipher Result")
                    st.download_button(
                        label="Export to PDF",
                        data=pdf_data,
                        file_name="caesar_result.pdf",
                        mime="application/pdf",
                        key="export_pdf"
                    )
        else:
            st.info("Enter text to see the encryption/decryption result")

def render_frequency_tab():
    st.header("Frequency Analysis")
    
    ciphertext = st.text_area("Ciphertext", height=150, key="freq_ciphertext")
    
    if ciphertext:
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Language",
                list(LANGUAGE_FREQUENCIES.keys()),
                index=0,
                key="freq_language"
            )
            
            ngram_level = st.selectbox(
                "N-gram Analysis Level",
                ["Unigram", "Bigram", "Trigram"],
                index=0,
                key="ngram_level"
            )
            
            n = 1
            if ngram_level == "Bigram":
                n = 2
            elif ngram_level == "Trigram":
                n = 3
            
            auto_detect = st.toggle("Auto-detect Key", value=True, key="auto_detect")
            
            if not auto_detect:
                key = st.number_input("Manual Key", min_value=0, max_value=100, value=0, key="manual_key")
        
        with col2:
            analyzer = FrequencyAnalyzer(language)
            
            if auto_detect:
                best_key, best_score = analyzer.find_best_key(ciphertext)
                st.success(f"Suggested key: {best_key} (confidence score: {100 - best_score:.1f}%)")
                decrypted = CaesarCipher().decrypt(ciphertext, best_key)
            else:
                decrypted = CaesarCipher().decrypt(ciphertext, key)
                best_key = key
            
            st.text_area("Decrypted Text", value=decrypted, height=150, key="decrypted_text")
        
        # Frequency analysis chart
        st.subheader("Frequency Analysis")
        
        cipher_freq = analyzer.calculate_frequencies(ciphertext, n=n)
        lang_freq = analyzer.language_freq['unigrams'] if n == 1 else analyzer.language_freq['bigrams'] if n == 2 else analyzer.language_freq['trigrams']
        
        # Prepare data for visualization
        freq_data = []
        for item, freq in lang_freq.items():
            freq_data.append({
                'Item': item,
                'Frequency': freq,
                'Type': 'Expected'
            })
        
        for item, freq in cipher_freq.items():
            freq_data.append({
                'Item': item,
                'Frequency': freq,
                'Type': 'Ciphertext'
            })
        
        df = pd.DataFrame(freq_data)
        
        fig = px.bar(
            df, 
            x='Item', 
            y='Frequency', 
            color='Type',
            barmode='group',
            title=f"{ngram_level} Frequency Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            if KALEIDO_INSTALLED:
                st.download_button(
                    label="Download Chart as PNG",
                    data=fig.to_image(format="png"),
                    file_name="frequency_chart.png",
                    mime="image/png",
                    key="download_chart"
                )
            else:
                st.warning("Install Kaleido for PNG export: `pip install kaleido`")
        with col2:
            report_text = f"""Frequency Analysis Report
Language: {language}
Suggested Key: {best_key}
Decrypted Text:
{decrypted}
"""
            pdf_data = ReportGenerator.generate_pdf(report_text, "Frequency Analysis Report")
            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="frequency_report.pdf",
                mime="application/pdf",
                key="download_report"
            )

def render_brute_force_tab():
    st.header("Brute Force Attack")
    
    ciphertext = st.text_area("Ciphertext", height=150, key="brute_force_ciphertext")
    
    if ciphertext:
        col1, col2 = st.columns(2)
        
        with col1:
            key_range = st.slider(
                "Key Range",
                min_value=1,
                max_value=26,
                value=(1, 26),
                key="key_range"
            )
            
            dictionary_option = st.selectbox(
                "Dictionary",
                ["English (NLTK)", "Custom Dictionary"],
                index=0,
                key="dictionary_option"
            )
            
            custom_dict = None
            if dictionary_option == "Custom Dictionary":
                uploaded_file = st.file_uploader("Upload Dictionary", type=["txt"], key="dict_upload")
                if uploaded_file:
                    custom_dict = set(uploaded_file.read().decode('utf-8').splitlines())
        
        with col2:
            if dictionary_option == "English (NLTK)":
                dictionary = set(nltk_words.words())
            else:
                dictionary = custom_dict or set()
            
            brute_forcer = BruteForcer(dictionary)
            
            if st.button("Run Brute Force Attack", key="brute_force_button"):
                with st.spinner("Running brute force attack..."):
                    results = brute_forcer.brute_force_decrypt(
                        ciphertext,
                        range(key_range[0], key_range[1] + 1)
                    )
                    
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('match_percentage', ascending=False)
                    
                    st.subheader("Results")
                    st.dataframe(
                        results_df,
                        column_config={
                            "key": "Key",
                            "decrypted_text": "Decrypted Text",
                            "matches": "Matches",
                            "match_percentage": "Match %"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Highlight the best result
                    best_result = results_df.iloc[0]
                    st.success(f"Best match: Key {best_result['key']} with {best_result['match_percentage']:.1f}% match")
                    st.text_area("Best Decryption", value=best_result['decrypted_text'], height=150, key="best_decryption")
                    
                    # Download options
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="brute_force_results.csv",
                        mime="text/csv",
                        key="download_results"
                    )

if __name__ == "__main__":
    main()
