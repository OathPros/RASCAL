import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cohere_reranker import cohere_rerank_action


class CohereRerankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"action_id": "a1", "title": "Laptop", "description": "Request a laptop"},
            {"action_id": "a2", "title": "VPN", "description": "Reset VPN access"},
        ]

    @patch.dict(os.environ, {"COHERE_ENABLED": "false"}, clear=True)
    def test_local_only_mode_returns_none(self):
        self.assertIsNone(cohere_rerank_action("vpn", self.candidates))

    @patch.dict(os.environ, {"COHERE_ENABLED": "true"}, clear=True)
    def test_missing_key_returns_none(self):
        self.assertIsNone(cohere_rerank_action("vpn", self.candidates))

    @patch.dict(os.environ, {"COHERE_ENABLED": "true", "COHERE_API_KEY": "test"}, clear=True)
    @patch("cohere_reranker.importlib.import_module")
    def test_exception_returns_none(self, mock_import_module):
        mock_import_module.side_effect = RuntimeError("boom")
        self.assertIsNone(cohere_rerank_action("vpn", self.candidates))

    @patch.dict(os.environ, {"COHERE_ENABLED": "true", "COHERE_API_KEY": "test"}, clear=True)
    @patch("cohere_reranker.importlib.import_module")
    def test_invalid_index_returns_none(self, mock_import_module):
        client = mock_import_module.return_value.ClientV2.return_value
        client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=99, relevance_score=0.9)]
        )
        self.assertIsNone(cohere_rerank_action("vpn", self.candidates))

    @patch.dict(os.environ, {"COHERE_ENABLED": "true", "COHERE_API_KEY": "test"}, clear=True)
    def test_empty_candidates_returns_none(self):
        self.assertIsNone(cohere_rerank_action("vpn", []))

    @patch.dict(os.environ, {"COHERE_ENABLED": "true", "COHERE_API_KEY": "test"}, clear=True)
    @patch("cohere_reranker.importlib.import_module")
    def test_successful_selection_returns_candidate_with_score(self, mock_import_module):
        client = mock_import_module.return_value.ClientV2.return_value
        client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=1, relevance_score=0.87)]
        )

        selected = cohere_rerank_action("vpn", self.candidates)

        self.assertEqual(selected["action_id"], "a2")
        self.assertEqual(selected["cohere_relevance_score"], 0.87)
        client.rerank.assert_called_once()
        _, kwargs = client.rerank.call_args
        self.assertEqual(kwargs["top_n"], 1)
        self.assertEqual(len(kwargs["documents"]), 2)


if __name__ == "__main__":
    unittest.main()
