import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleSheetManager:
    def __init__(self, sheet_name: str, worksheet_name: str):
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        self.creds_path = "gsheet_creds.json"  # Ensure this file is in your project root
        self.sheet_name = sheet_name
        self.worksheet_name = worksheet_name
        self.headers = [
            "Question", "Answer", "Retrieved_Contexts", 
            "LLM_Context_Query_Score", "LLM_Context_Query_Score_Reasoning", 
            "LLM_Context_Answer_Score", "LLM_Context_Answer_Score_Reasoning", 
            "LLM_Answer_Query_Score", "LLM_Answer_Query_Score_Reasoning", 
            "Human_Score", "Human_Score_Reasoning"
        ]
        self.client = self._initialize_client()
        self.sheet = self._get_sheet()
        self._setup_sheet()

    def _initialize_client(self) -> gspread.Client:
        """ Initializes the Google Sheets client using service account credentials. """
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_path, self.scope)
            return gspread.authorize(creds)
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets client: {e}")
            raise

    def _get_sheet(self) -> gspread.Worksheet:
        """ Opens the specified Google Sheet and worksheet tab. """
        try:
            spreadsheet = self.client.open(self.sheet_name)
            worksheet = spreadsheet.worksheet(self.worksheet_name)  # Select by worksheet name
            logger.info(f"Connected to Google Sheet: '{self.sheet_name}', Worksheet: '{self.worksheet_name}'")
            return worksheet
        except gspread.SpreadsheetNotFound:
            logger.error(f"Google Sheet '{self.sheet_name}' not found! Ensure the service account has access.")
            raise
        except gspread.WorksheetNotFound:
            logger.error(f"Worksheet '{self.worksheet_name}' not found in '{self.sheet_name}'! Check if the tab exists.")
            raise
        except Exception as e:
            logger.error(f"Failed to open Google Sheet: {e}")
            raise

    def _setup_sheet(self) -> None:
        """ Ensures the sheet has the correct headers. """
        try:
            existing_headers = self.sheet.row_values(1)
            if not existing_headers:
                self.sheet.append_row(self.headers)
                logger.info(f"Google Sheet '{self.sheet_name}' - Worksheet '{self.worksheet_name}' headers initialized.")
        except Exception as e:
            logger.error(f"Failed to set up Google Sheet headers: {e}")
            raise

    def log_agent_response(self, query: str, response: str, contexts: str, eval_scores: Dict[str, float]) -> None:
        """ Logs AI response and evaluation scores into the Google Sheet. """
        try:
            row_data = [
                query,
                response,
                contexts,
                eval_scores.get("llm_context_query_score", "N/A"),
                eval_scores.get("llm_context_query_score_reasoning", "N/A"),
                eval_scores.get("llm_context_answer_score", "N/A"),
                eval_scores.get("llm_context_answer_score_reasoning", "N/A"),
                eval_scores.get("llm_answer_query_score", "N/A"),
                eval_scores.get("llm_answer_query_score_reasoning", "N/A"),
                "",  # Empty column for human score
                ""   # Empty column for human feedback reasoning
            ]
            self.sheet.append_row(row_data)
            logger.info(f"Logged AI response for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to log AI response: {e}")
            raise

    def update_human_feedback(self, row: int, human_score: int, human_reasoning: str) -> None:
        """ Updates human evaluation feedback for a given row in the Google Sheet. """
        try:
            self.sheet.update_cell(row, len(self.headers) - 1, human_score)
            self.sheet.update_cell(row, len(self.headers), human_reasoning)
            logger.info(f"Updated human feedback for row {row}")
        except Exception as e:
            logger.error(f"Failed to update human feedback: {e}")
            raise
