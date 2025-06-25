# mymed-backup

A Python-based backup and data-processing toolkit for MyMed chat function application (currently prototype stays in the web).
It scrapes, processes, evaluates and stores your MyMed data, and provides a Streamlit frontend for exploration.

### Features

- **Web scraping** of MyMed data via `scraping/`  
- **Data cleansing & transformation** in `data_processing/`  
- **Automated evaluation** pipelines in `evaluation/`  
- **Google Sheets integration** via `gsheet.py`  
- **Streamlit UI** for ad-hoc data exploration in `frontend.py`  
- **Standalone backend** logic in `backend.py`  

---

### Github Branches Use Case
<img width="277" alt="Screenshot 2025-06-25 at 10 14 08 AM" src="https://github.com/user-attachments/assets/661cc64d-78c6-4f1b-8635-c2d6e6d3a955" />

main: get english answers (without health info)
user-params: using dynamic health parameters from users
svenska-checkpoint: get swedish answers (without health info)

---

### Code architecture and Diagrams

![GitDiagram (2)](https://github.com/user-attachments/assets/1345e5ca-0794-493a-a808-3f8b8e9647c8)

---

### Prerequisites

- Python 3.8 or higher  
- `pip` (bundled with Python)  
- A Google service account JSON key for Sheets API  
- (Optional) A `.env.local` file for environment-specific secrets  

---

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/briannoelkesuma/mymed-backup.git
   cd mymed-backup
   ```

2. **(Recommended) Create & activate a virtual environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

### Configuration

1. **Google Sheets credentials**  
   - Place your service-account JSON file as `gsheet_creds.json` in the project root.  
   - Grant the service account “Editor” access to your target Google Sheet(s).

2. **Environment variables**  
   - Create a `.env` file in the root directory:  
     ```dotenv
     SECRET_KEY_1=XXXX
     ```
   - These values will be loaded by the application at runtime.

3. **Ranges definition**  
   - Edit `ranges.json` to map health data fields to categories (from the current limits).

---

### Usage

#### 1. Run the full pipeline  
Performs scraping, processing, evaluation, and pushes results to Google Sheets:
```bash
python backend.py
```

#### 2. Launch the Streamlit dashboard  
Explore and visualize your data interactively:
```bash
streamlit run frontend.py
```

#### 3. Google Sheets helper  
- The current answers tagged to each category in the app is here: https://docs.google.com/spreadsheets/d/1EZtkimGzd3rp_Kk-8USJFahChMmRIIEfpRWYVDseSvU/edit?usp=sharing
- For evaluation, replace this line in backend.py
```python
eval_sheet_manager = GoogleSheetManager("MyMed_Agent_Evaluation", "testing (R3)") // ("first parameter workbook name", "second parameter worksheet name")
```

---

### Project Structure

```
mymed-backup/
├── .streamlit/               # Streamlit config files
├── data_processing/          # ETL & data-cleaning modules
├── evaluation/               # Evaluation scripts & reports stored in Gsheets
├── scraping/                 # Web-scraping & helpers with AI
├── backend.py                # Orchestrates full pipeline
├── frontend.py               # Streamlit UI dashboard
├── gsheet.py                 # Google Sheets upload/download utilities
├── gsheet_creds.json         # Google service-account key (gitignored)
├── ranges.json               # Cell-range mappings
├── requirements.txt          # Python dependencies
└── .env.local                # Environment-specific secrets (gitignored)
```

---

### Acknowledgements

- Thanks to all contributors and the open-source community for support and inspiration.  
- Built with ❤️ by Brian Noelkesuma  
