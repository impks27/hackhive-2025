import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import shutil
import pandas as pd
import numpy as np

# Mocking AnalysisLauncher to prevent real execution
mock_analysis_launcher = MagicMock()
mock_analysis_launcher.process.return_value = [
    {"category": "Loan", "confidence_score": 0.95, "sub_category": {"name": "Mortgage", "confidence_score": 0.9},
     "extracted_fields": {"deal_name": "Home Loan", "amount": 50000, "transaction_date": "2024-01-01", "account_number": "12345", "currency": "USD"}}
]

class TestLendingServiceApp(unittest.TestCase):
    
    def setUp(self):
        from app import LendingServiceApp  # Import inside to avoid global execution
        self.app = LendingServiceApp()
        self.app.temp_dir = "test_temp"
        os.makedirs(self.app.temp_dir, exist_ok=True)
        # Add project root to sys.path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    def tearDown(self):
        if os.path.exists(self.app.temp_dir):
            shutil.rmtree(self.app.temp_dir)
    
    @patch("os.listdir", return_value=["file1.pdf", "file2.eml"])
    @patch("classifier.AnalysisLauncher", return_value=mock_analysis_launcher)
    def test_analyze_files(self, mock_launcher, mock_listdir):
        self.app.file_paths = ["test_temp/file1.pdf", "test_temp/file2.eml"]
        result = self.app.analyze_files()
        
        self.assertEqual(len(result), 2)
        self.assertIn("file1.pdf", result)
        self.assertIn("file2.eml", result)
    
    def test_clean_inventory(self):
        self.app.clean_inventory()
        self.assertFalse(os.path.exists(self.app.temp_dir))
    
    def test_flatten_output(self):
        output_dict = {
            "file1.pdf": [{
                "category": "Loan", "confidence_score": 0.95,
                "sub_category": {"name": "Mortgage", "confidence_score": 0.9},
                "extracted_fields": {"deal_name": "Home Loan", "amount": 50000, "transaction_date": "2024-01-01", "account_number": "12345", "currency": "USD"}
            }]
        }
        df = self.app.flatten_output(output_dict)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df.iloc[0]["File Name"], "file1.pdf")
        self.assertEqual(df.iloc[0]["Amount"], 50000)
    
if __name__ == "__main__":
    unittest.main()
