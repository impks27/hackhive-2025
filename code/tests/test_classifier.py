import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from classifier import AnalysisLauncher

class TestAnalysisLauncher(unittest.TestCase):
    
    def setUp(self):
        self.launcher = AnalysisLauncher("temp")

    @patch("pdfplumber.open")
    def test_extract_text_from_pdf(self, mock_pdfplumber):
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock() for _ in range(2)]
        mock_pdf.pages[0].extract_text.return_value = "Page 1 text"
        mock_pdf.pages[1].extract_text.return_value = "Page 2 text"
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        result = self.launcher.extract_text_from_pdf("dummy.pdf")
        self.assertEqual(result, "Page 1 text Page 2 text")
    
    @patch("builtins.open", new_callable=mock_open, read_data="sample text")
    def test_read_file(self, mock_file):
        result = self.launcher.read_file("dummy.txt")
        self.assertEqual(result, "sample text")
    
    @patch("mailparser.parse_from_file")
    def test_extract_text_from_eml(self, mock_mailparser):
        mock_mail = MagicMock()
        mock_mail.body = "Email body text"
        mock_mail.attachments = []
        mock_mailparser.return_value = mock_mail
        
        body, attachment = self.launcher.extract_text_from_eml("dummy.eml")
        self.assertEqual(body, "Email body text")
        self.assertEqual(attachment, "")
    
    # @patch("classifier.AnalysisLauncher.extract_text_from_file", return_value="Extracted file content")
    # @patch("classifier.AnalysisLauncher.read_file", side_effect=lambda x: json.dumps({"Loan": "loan_subcategories.json"}) if "resources" in x else "")
    # @patch("ollama.chat")
    # def test_process(self, mock_ollama_chat, mock_read_file, mock_extract_text):
    #     # Ensure response is wrapped in ```json\n...\n``` as expected by extract_json_block
    #     mock_ollama_chat.return_value = {
    #         "message": {
    #             "content": "```json\n" + json.dumps([{
    #                 "classification": {"category": "Loan", "confidence_score": 0.95},
    #                 "associated_text": "Sample classified text",
    #                 "extracted_fields": {"deal_name": "Deal123"}
    #             }]) + "\n```"
    #         }
    #     }

    #     result = self.launcher.process("dummy.pdf")

    #     # Assertions
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), 1)
    #     self.assertEqual(result[0]["category"], "Loan")
    #     self.assertEqual(result[0]["confidence_score"], 0.95)
    #     self.assertEqual(result[0]["extracted_fields"]["deal_name"], "Deal123")


    
    @patch("re.search")
    def test_extract_json_block(self, mock_re_search):
        mock_re_search.return_value = MagicMock(group=lambda _: '{"key": "value"}')
        result = self.launcher.extract_json_block("```json\n{\"key\": \"value\"}\n```")
        self.assertEqual(result, '{"key": "value"}')

if __name__ == "__main__":
    unittest.main()
