import os
import unittest
from unittest.mock import patch

import app


class ChatSelectionTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    @patch.dict(os.environ, {"COHERE_ENABLED": "false", "COHERE_TOP_K": "25"}, clear=True)
    def test_chat_uses_local_when_cohere_disabled(self):
        response = self.client.post("/api/chat", json={"message": "vpn access"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["selection_source"], "local")
        self.assertEqual(data["ranking_debug"]["selection_source"], "local")
        self.assertLessEqual(len(data["candidates"]), 3)
        self.assertFalse(data["ranking_debug"]["fallback_used"])

    @patch.dict(os.environ, {"COHERE_ENABLED": "true", "COHERE_TOP_K": "25"}, clear=True)
    def test_chat_falls_back_when_key_missing(self):
        response = self.client.post("/api/chat", json={"message": "vpn access"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["selection_source"], "local")
        self.assertLessEqual(len(data["candidates"]), 3)
        self.assertTrue(data["ranking_debug"]["fallback_used"])

    @patch.dict(os.environ, {"COHERE_ENABLED": "false", "COHERE_TOP_K": "25"}, clear=True)
    def test_chat_uses_context_to_refine_search(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "for vpn", "context": ["I need remote access"]},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["latest_message"], "for vpn")
        self.assertEqual(data["context"], ["I need remote access"])
        self.assertIn("I need remote access", data["query"])
        self.assertIn("for vpn", data["query"])
        self.assertLessEqual(len(data["candidates"]), 3)


class ChatIntentGateTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def _post_with_intent(self, message, intent):
        with patch.dict(os.environ, {
            "COHERE_ENABLED": "true",
            "COHERE_API_KEY": "test-key",
            "INTENT_REFINEMENT_ENABLED": "true",
            "INTENT_REFINEMENT_DEBUG": "true",
            "AZURE_FOUNDRY_ENDPOINT": "https://example.test/models",
            "AZURE_FOUNDRY_API_KEY": "secret",
            "AZURE_FOUNDRY_MODEL": "DeepSeek-V4-Pro",
        }, clear=True), patch("app.refine_intent_with_llm", return_value=intent), patch("app.cohere_rerank_action", return_value=None) as mock_cohere:
            response = self.client.post("/api/chat", json={"message": message})
        return response, mock_cohere

    def test_puppy_returns_no_service_action_candidates(self):
        response, mock_cohere = self._post_with_intent("I need a puppy", {
            "intent_classification": "non_it_or_bogus",
            "status": "non_it_or_bogus",
            "confidence": 0.98,
            "user_goal": "get a puppy",
            "normalized_query": "",
            "ranking_keywords": [],
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["candidates"], [])
        self.assertIsNone(data["selected_action_id"])
        self.assertFalse(data["cohere_attempted"])
        mock_cohere.assert_not_called()

    def test_kitten_returns_no_service_action_candidates(self):
        response, mock_cohere = self._post_with_intent("what about a kitten?", {
            "intent_classification": "non_it_or_bogus",
            "status": "non_it_or_bogus",
            "confidence": 0.96,
            "user_goal": "ask about a kitten",
            "normalized_query": "",
            "ranking_keywords": [],
        })
        data = response.get_json()
        self.assertEqual(data["candidates"], [])
        self.assertFalse(data["cohere_attempted"])
        mock_cohere.assert_not_called()

    def test_gibberish_returns_no_service_action_candidates(self):
        response, mock_cohere = self._post_with_intent("asdfasdfasdf", {
            "intent_classification": "unsafe_or_unusable",
            "status": "unsafe_or_unusable",
            "confidence": 0.91,
            "user_goal": "",
            "normalized_query": "",
            "ranking_keywords": [],
        })
        data = response.get_json()
        self.assertEqual(data["candidates"], [])
        self.assertFalse(data["cohere_attempted"])
        mock_cohere.assert_not_called()

    def test_passport_york_returns_valid_service_action_candidates(self):
        response, _mock_cohere = self._post_with_intent("I can’t log into Passport York", {
            "intent_classification": "valid_it_service_request",
            "status": "valid_it_service_request",
            "confidence": 0.94,
            "user_goal": "get help logging into Passport York",
            "normalized_query": "Passport York login account access",
            "ranking_keywords": ["passport york", "account", "login"],
        })
        data = response.get_json()
        self.assertGreater(len(data["candidates"]), 0)
        self.assertEqual(data["intent_classification"], "valid_it_service_request")

    def test_vpn_access_returns_valid_service_action_candidates(self):
        response, _mock_cohere = self._post_with_intent("I need VPN access", {
            "intent_classification": "valid_it_service_request",
            "status": "valid_it_service_request",
            "confidence": 0.95,
            "user_goal": "request VPN access",
            "normalized_query": "VPN access remote access",
            "ranking_keywords": ["vpn", "remote access"],
        })
        data = response.get_json()
        self.assertGreater(len(data["candidates"]), 0)
        self.assertEqual(data["intent_classification"], "valid_it_service_request")


if __name__ == "__main__":
    unittest.main()
