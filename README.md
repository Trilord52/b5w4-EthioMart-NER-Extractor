# EthioMart Amharic E-Commerce NER 🇪🇹🛒

Extracting Product, Price, and Location Entities from Amharic Telegram E-Commerce Posts

## Project Overview
This project builds an NLP pipeline to extract structured information from unstructured Telegram messages posted by Amharic-speaking e-commerce vendors. The goal is to identify key entities:

- 🛍️ **PRODUCT**
- 💰 **PRICE**
- 📍 **LOCATION**

The extracted information will support vendor analytics, dashboards, and intelligent recommendations for the Ethiopian e-commerce ecosystem.

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── telegram_data_combined.csv
│   │   ├── channels_to_crawl.xlsx
│   │   └── photos/
│   └── processed/
│       ├── ner_sample_conll.txt
│       └── preprocessed_telegram_data.csv
├── notebooks/
│   ├── fine_tune_ner_colab.ipynb
│   ├── model_comparison.ipynb  # PC version
│   ├── model_comparison_colab.ipynb  # Google Colab version
│   ├── model_comparison_ner.ipynb  # NER-focused, visual, Colab-ready
│   ├── model_interpretability_colab.ipynb  # Interpretability (SHAP/LIME)
│   └── vendor_scorecard_colab.ipynb  # Vendor analytics & scorecard (Task 6)
├── scripts/
│   ├── fine_tune_ner.py
│   ├── telegram_scraper.py
│   ├── preprocess.py
│   └── extract_conll_template.py
├── requirements.txt
├── CONTRIBUTING.md
├── README.md
```

- **data/raw/**: Raw scraped data and input files
- **data/raw/photos/**: Downloaded images from Telegram channels
- **data/processed/**: Cleaned and labeled data for NER
- **notebooks/**: Jupyter/Colab notebooks for model training and experimentation
- **scripts/**: All data pipeline, labeling, and model scripts

## Quick Start

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run the Data Extraction and Labeling Pipeline

- **Preprocessing & Cleaning:**
    - `python scripts/preprocess.py`
- **Generate NER Labeling Template:**
    - `python scripts/extract_conll_template.py`

This will load the raw Telegram dataset and output a sample labeled file in CoNLL format at `data/processed/ner_sample_conll.txt`.

## Methods
- **Rule-based labeling system:**
    - **PRODUCT:** Keyword anchors (e.g., "Laptop", "Blender", "ታብሌት", etc.)
    - **PRICE:** Regex and keyword cues for patterns like `"6800 ብር"`, `"ዋጋ 5000"`, or numbers followed by `birr`/`ብር`
    - **LOCATION:** Keyword-based clues like `ፎቅ`, `ሞል`, `ቦታ`, `አድራሻ`
- **BIO scheme:** Labels follow the BIO format (B-, I-, O) for multi-token entities.
- **Heuristic tokenization:** Handles both Amharic and English tokens.

## Outputs
- `data/raw/telegram_data_combined.csv`: Raw scraped Telegram messages
- `data/processed/ner_sample_conll.txt`: Weakly labeled sample data in CoNLL format for NER

## Model Fine-Tuning (Task 3)

### A. Local Script Usage

```bash
pip install -r requirements.txt
python scripts/fine_tune_ner.py
```

- The script will:
  - Load and preprocess the CoNLL-formatted NER data.
  - Tokenize and align labels for the model.
  - Fine-tune a pre-trained transformer (default: XLM-Roberta, easily switchable).
  - Print and save performance metrics (F1-score, accuracy, classification report).
  - Save the fine-tuned model and tokenizer to `results_ner_amharic/`.

### B. Google Colab Notebook

- Open `notebooks/fine_tune_ner_colab.ipynb` in Google Colab.
- Follow the step-by-step cells:
  - Install dependencies
  - Upload your data
  - Run all cells for training, evaluation, and model saving

---

### Example Output

```
Loaded 120 sentences.
***** Running training *****
  Num examples = 108
  Num Epochs = 3
  ...
***** Running Evaluation *****
Evaluation Results:
              precision    recall  f1-score   support

     B-LOC       0.85      0.80      0.82        10
     B-PRICE     0.90      0.88      0.89        16
     B-PRODUCT   0.92      0.90      0.91        22
     I-LOC       0.80      0.75      0.77         8
     I-PRICE     0.88      0.85      0.86        12
     I-PRODUCT   0.91      0.89      0.90        18
     O           0.98      0.99      0.99       200

micro avg       0.95      0.94      0.94       286
macro avg       0.89      0.87      0.88       286
weighted avg    0.94      0.94      0.94       286

F1-score: 0.9402
Accuracy: 0.9437

Model and tokenizer saved to results_ner_amharic
```

---

### Reproducibility & Customization

- All code is modular and well-commented.
- Change model, learning rate, batch size, or epochs by editing the config section in the script or notebook.
- For more details, see the code comments and markdown in the notebook.

## Coding Standards & Contribution
- Follow [PEP8](https://peps.python.org/pep-0008/) for Python code style.
- Use modular functions/classes and add docstrings.
- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Next Steps
- Manual validation and correction of entity labels
- Fine-tuning a BERT-based (or similar) Amharic NER model
- Model evaluation and interpretability
- Entity-based analytics and dashboards

## Model Comparison & Selection (Task 4)

### A. Local/PC Version
- Run `notebooks/model_comparison.ipynb` to compare XLM-Roberta, mBERT, and DistilBERT on your Amharic NER data.
- Update the data path if needed.
- Results are summarized in a table at the end of the notebook.

### B. Google Colab Version
- Open `notebooks/model_comparison_colab.ipynb` in Google Colab.
- Mount your Google Drive and set the data path to your CoNLL file (e.g., `/content/drive/MyDrive/NER/ner_sample_conll.txt`).
- Run all cells to fine-tune and compare models. Results and logs are saved to your Drive.

---

## Model Interpretability (Task 5)

- Open `notebooks/model_interpretability_colab.ipynb` in Google Colab.
- Mount your Google Drive and set the model path to your fine-tuned model directory (e.g., `/content/drive/MyDrive/NER/results/XLM-Roberta/`).
- Run all cells to interpret model predictions using SHAP and LIME.
- Summarize findings and recommendations in the final cell.

---

## Vendor Scorecard for Micro-Lending (Task 6)

- Open `notebooks/vendor_scorecard_colab.ipynb` in Google Colab.
- Mount your Google Drive and set the data path to your Telegram data CSV (e.g., `/content/drive/MyDrive/NER/telegram_data_combined.csv`).
- The notebook will:
  - Load and sample your Telegram data
  - Group by vendor/channel
  - Calculate posting frequency, average views, top post, average price, and a lending score
  - Output and save a summary scorecard table
  - Visualize the top vendors
- The scorecard helps identify promising vendors for micro-lending based on real engagement and business activity metrics.

---

## Requirements

- All requirements for local and Colab runs are in `requirements.txt`:
```
telethon
python-dotenv
pandas
openpyxl
unicodedata2
transformers
datasets
seqeval
numpy
shap
lime
matplotlib
seaborn
```
- For Google Colab, all dependencies are installed in the first cell of each notebook.
