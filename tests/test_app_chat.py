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


if __name__ == "__main__":
    unittest.main()
