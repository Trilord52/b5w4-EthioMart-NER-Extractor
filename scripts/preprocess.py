import pandas as pd
import os
import re
import unicodedata

def normalize_amharic(text):
    """Normalize Amharic text: remove diacritics, normalize similar characters."""
    # Remove diacritics (combining marks)
    text = ''.join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn')
    # Normalize common Amharic character variants (add more as needed)
    text = text.replace('ሃ', 'ሀ').replace('ኅ', 'ሀ').replace('ኃ', 'ሀ')
    text = text.replace('ሐ', 'ሀ').replace('ኻ', 'ሀ')
    text = text.replace('ሓ', 'ሀ').replace('ኸ', 'ሀ')
    return text

def clean_message(text):
    """Remove URLs, emojis, and non-Amharic/English noise from text."""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', str(text))
    # Remove emojis and non-text symbols
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove non-Amharic/English characters except numbers and basic punctuation
    text = re.sub(r'[^\u1200-\u1399a-zA-Z0-9፡።፣፤፥፦፧፨.,!?\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_telegram_data(input_csv, output_dir):
    """Preprocess Telegram data: normalize, clean, deduplicate, and structure."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # Drop duplicates based on message content
    df = df.drop_duplicates(subset=['Message'])
    # Normalize and clean messages
    df['Message'] = df['Message'].apply(normalize_amharic).apply(clean_message)
    # Remove empty or irrelevant messages
    df = df[df['Message'].str.strip().astype(bool)]
    # Separate metadata and message content
    meta_cols = [col for col in df.columns if col != 'Message']
    df_meta = df[meta_cols]
    df_msg = df[['Message']]
    # Save cleaned and structured data
    output_csv = os.path.join(output_dir, 'telegram_data_cleaned.csv')
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == '__main__':
    preprocess_telegram_data('data/raw/telegram_data_combined.csv', 'data/processed') 