import pandas as pd
import os
import random
import re

def amharic_tokenize(text):
    # Simple whitespace and punctuation-based tokenizer for Amharic
    tokens = re.findall(r'\w+|[\u1369-\u137C\u1200-\u137F\u1380-\u1399\u2D80-\u2DDF\u200C\u200D\u002E\u002C\u0021\u061F\u1361\u1362\u1363\u1364\u1365\u1366\u1367\u1368\u1369\u136A\u136B\u136C\u136D\u136E\u136F\u1370\u1371\u1372\u1373\u1374\u1375\u1376\u1377\u1378\u1379\u137A\u137B\u137C\u137D\u137E\u137F\u1380\u1381\u1382\u1383\u1384\u1385\u1386\u1387\u1388\u1389\u138A\u138B\u138C\u138D\u138E\u138F\u1390\u1391\u1392\u1393\u1394\u1395\u1396\u1397\u1398\u1399\u0020]+', text)
    return tokens

PRODUCT_CUES = {"Table", "Desk", "Edge", "Guard", "Strip", "Laptop", "Blender", "Tablet", "Shoes", "Car", "Puzzle", "Gift", "PC", "LAPTOP", "COMPUTER", "Sticker", "Furniture", "Wall", "Blocks", "Science", "Talent", "Rubiks", "Cubes", "Nike", "Adidas", "ASUS", "Lenovo", "MSI", "Sonifer", "Skechers", "Converse", "potato", "masher", "Ironing", "Board", "ታብሌት", "መኪና", "ሻውር", "የቤት", "የልጅ", "የማብራሪያ", "የገበያ"}
PRICE_CUES = {"ዋጋ", "Price", "birr", "ብር", "፦", "ዋጋ፦"}
LOC_CUES = {"አድራሻ", "ቦታ", "Bole", "Addis", "Abeba", "ሆቴል", "ሴንተር", "ፎቅ", "ቤት", "ሱቅ", "ህንፃ", "ቁ.", "ቢሮ", "ቅርንጫፍ", "፦"}

# Helper to check if a token is a number (for price)
def is_number(token):
    return bool(re.fullmatch(r"[0-9]+(,?[0-9]+)*", token))

def is_phone_number(token):
    # Ethiopian phone numbers are usually 9 or more digits
    return bool(re.fullmatch(r"[0-9]{9,}", token))

def is_price_token(token):
    # Recognize tokens like '750ብር', '1000birr', '500ብር'
    return bool(re.fullmatch(r"[0-9]+(,?[0-9]+)*(ብር|birr)", token))

def is_amharic(token):
    return bool(re.match(r"[\u1200-\u1399]+", token))

def label_tokens(tokens):
    labels = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Product entity (multi-token, Amharic or English)
        if token in PRODUCT_CUES or is_amharic(token) and len(token) > 2:
            start = i
            while i < len(tokens) and (tokens[i] in PRODUCT_CUES or is_amharic(tokens[i]) and len(tokens[i]) > 2):
                i += 1
            for j in range(start, i):
                labels.append("B-PRODUCT" if j == start else "I-PRODUCT")
            continue
        # Price entity: Case 1 - token is a number immediately followed by 'birr' or 'ብር' (e.g., '750ብር')
        if is_price_token(token) and not is_phone_number(token) and len(token) < 10:
            labels.append("B-PRICE")
            i += 1
            continue
        # Price entity: Case 2 - number before a separate 'birr' or 'ብር' token
        if is_number(token) and not is_phone_number(token) and len(token) < 7:
            if i + 1 < len(tokens) and tokens[i+1] in {"birr", "ብር"}:
                labels.append("B-PRICE")
                labels.append("I-PRICE")
                i += 2
                continue
            # Only label as price if previous token was a price cue
            if i > 0 and tokens[i-1] in PRICE_CUES:
                labels.append("I-PRICE")
            else:
                labels.append("O")
            i += 1
            continue
        # If token is 'birr' or 'ብር' and previous token is a number, treat both as price
        if token in {"birr", "ብር"} and i > 0 and is_number(tokens[i-1]) and not is_phone_number(tokens[i-1]) and len(tokens[i-1]) < 7:
            # Already labeled previous as B-PRICE or I-PRICE, so just label this as I-PRICE
            labels.append("I-PRICE")
            i += 1
            continue
        # Price entity (multi-token, only if number is <7 digits and near a price cue)
        if token in PRICE_CUES:
            labels.append("B-PRICE")
            i += 1
            # Label following numbers as I-PRICE if they are not phone numbers and <7 digits, or price-like tokens
            while i < len(tokens) and ((is_number(tokens[i]) and not is_phone_number(tokens[i]) and len(tokens[i]) < 7) or tokens[i] in {"ብር", "birr", "፦"} or is_price_token(tokens[i])):
                labels.append("I-PRICE")
                i += 1
            continue
        # Location entity (multi-token, after አድራሻ or ፦, group Amharic/English/number tokens)
        if token in LOC_CUES:
            labels.append("B-LOC")
            i += 1
            # Group next tokens as I-LOC if they are Amharic, numbers, or in LOC_CUES (up to 5 tokens)
            loc_count = 0
            while i < len(tokens) and (is_amharic(tokens[i]) or is_number(tokens[i]) or tokens[i] in LOC_CUES) and loc_count < 5:
                labels.append("I-LOC")
                i += 1
                loc_count += 1
            continue
        # Phone numbers (9+ digits)
        if is_phone_number(token):
            labels.append("O")
            i += 1
            continue
        # Default
        labels.append("O")
        i += 1
    return labels

def extract_conll_template(input_csv, output_txt, num_messages=40, random_seed=42):
    df = pd.read_csv(input_csv)
    messages = df['Message'].dropna().unique()
    random.seed(random_seed)
    selected = random.sample(list(messages), min(num_messages, len(messages)))

    with open(output_txt, 'w', encoding='utf-8') as f:
        for msg in selected:
            tokens = amharic_tokenize(str(msg))
            labels = label_tokens(tokens)
            for token, label in zip(tokens, labels):
                if token.strip():
                    f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line between messages
    print(f"Demo-style labeled template for {len(selected)} messages written to {output_txt}")

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    extract_conll_template(
        input_csv='data/raw/telegram_data_combined.csv',
        output_txt='data/processed/ner_sample_conll.txt',
        num_messages=40
    ) 