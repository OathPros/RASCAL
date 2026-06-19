import json
import os
import unittest
from unittest.mock import patch

import app
from intent_refinement import (
    _completion_url,
    _extract_completion_text,
    build_ranking_query,
    extract_json_object,
    get_intent_refinement_config,
    heuristic_intent_fallback,
    should_refine_intent,
    validate_intent_response,
)


class IntentRefinementUnitTests(unittest.TestCase):
    def test_ambiguity_detector_triggers_on_vague_text(self):
        rankings = [{"score": 20.0}, {"score": 10.0}]
        self.assertTrue(should_refine_intent(rankings, "I need access to the thing"))

    def test_ambiguity_detector_does_not_trigger_on_clear_high_confidence(self):
        rankings = [{"score": 20.0}, {"score": 10.0}]
        self.assertFalse(should_refine_intent(rankings, "Reset my Passport York password"))

    def test_valid_llm_json_is_accepted_and_clamped(self):
        raw = json.dumps({
            "status": "ready_to_rank",
            "user_goal": "reset password",
            "normalized_query": "reset Passport York password",
            "likely_service_area": "accounts",
            "request_type": "password_reset",
            "missing_information": "not-a-list",
            "clarifying_question": None,
            "confidence": 2,
            "ranking_keywords": ["passport york", "password"],
        })
        parsed = validate_intent_response(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["confidence"], 1.0)
        self.assertEqual(parsed["missing_information"], [])

    def test_invalid_llm_json_falls_back_safely(self):
        self.assertIsNone(validate_intent_response("Here is no usable object"))

    def test_json_object_is_extracted_from_wrapped_text(self):
        parsed = validate_intent_response('Here is JSON: {"intent_classification":"non_it_or_bogus","confidence":0.8}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent_classification"], "non_it_or_bogus")

    def test_json_object_is_extracted_from_fenced_text(self):
        parsed = extract_json_object('```json\n{"intent_classification":"valid_it_service_request"}\n```')
        self.assertEqual(parsed["intent_classification"], "valid_it_service_request")

    def test_heuristic_fallback_blocks_pet_queries(self):
        parsed = heuristic_intent_fallback("I need a puppy")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent_classification"], "non_it_or_bogus")

    def test_heuristic_fallback_blocks_gibberish(self):
        parsed = heuristic_intent_fallback("asdfasdfasdf")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent_classification"], "unsafe_or_unusable")

    def test_ready_to_rank_builds_improved_query(self):
        query = build_ranking_query({
            "user_goal": "request access",
            "normalized_query": "request access to SharePoint onboarding site",
            "likely_service_area": "SharePoint",
            "request_type": "access_request",
            "ranking_keywords": ["new employee setup", "permissions"],
        })
        self.assertIn("SharePoint", query)
        self.assertIn("new employee setup", query)

    @patch.dict(os.environ, {
        "INTENT_REFINEMENT_ENABLED": "true",
        "INTENT_REFINEMENT_PROVIDER": "anthropic",
        "INTENT_REFINEMENT_TARGET_URL": "https://llm.example.test/claude-haiku",
        "INTENT_REFINEMENT_API_CODE": "institution-code",
        "INTENT_REFINEMENT_MODEL": "claude-haiku-4-5",
    }, clear=True)
    def test_config_accepts_claude_haiku_target_url_and_api_code(self):
        config = get_intent_refinement_config()
        self.assertTrue(config.enabled)
        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.target_url, "https://llm.example.test/claude-haiku")
        self.assertEqual(config.api_key, "institution-code")
        self.assertEqual(config.auth_header, "x-api-key")
        self.assertEqual(config.model, "claude-haiku-4-5")

    @patch.dict(os.environ, {
        "INTENT_REFINEMENT_ENABLED": "true",
        "AZURE_FOUNDRY_ENDPOINT": "https://example.services.ai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "secret",
        "AZURE_FOUNDRY_MODEL": "deepseek-v3",
    }, clear=True)
    def test_azure_foundry_aliases_use_models_chat_completions_url(self):
        config = get_intent_refinement_config()
        self.assertEqual(config.provider, "azure-foundry")
        self.assertEqual(config.auth_header, "api-key")
        self.assertEqual(
            _completion_url(config),
            "https://example.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
        )

    @patch.dict(os.environ, {
        "INTENT_REFINEMENT_ENABLED": "true",
        "INTENT_REFINEMENT_PROVIDER": "azure-foundry",
        "INTENT_REFINEMENT_TARGET_URL": "https://example.services.ai.azure.com/models/chat/completions?api-version=2025-01-01-preview",
        "INTENT_REFINEMENT_API_KEY": "secret",
        "INTENT_REFINEMENT_MODEL": "deepseek-v3",
    }, clear=True)
    def test_azure_foundry_keeps_full_chat_completions_url(self):
        config = get_intent_refinement_config()
        self.assertEqual(
            _completion_url(config),
            "https://example.services.ai.azure.com/models/chat/completions?api-version=2025-01-01-preview",
        )

    def test_anthropic_response_text_is_extracted(self):
        text = _extract_completion_text(
            {"content": [{"type": "text", "text": "{\"status\":\"ready_to_rank\"}"}]},
            "anthropic",
        )
        self.assertEqual(text, "{\"status\":\"ready_to_rank\"}")


class IntentRefinementChatTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    @patch.dict(os.environ, {"COHERE_ENABLED": "false", "INTENT_REFINEMENT_ENABLED": "false"}, clear=True)
    def test_intent_refinement_disabled_preserves_existing_flow(self):
        response = self.client.post("/api/chat", json={"message": "vpn access"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["selection_source"], "local")
        self.assertNotIn("type", data)
        self.assertLessEqual(len(data["candidates"]), 3)

    @patch.dict(os.environ, {
        "COHERE_ENABLED": "false",
        "INTENT_REFINEMENT_ENABLED": "true",
        "AZURE_FOUNDRY_ENDPOINT": "https://example.test/models",
        "AZURE_FOUNDRY_API_KEY": "secret",
        "AZURE_FOUNDRY_MODEL": "DeepSeek-V4-Pro",
    }, clear=True)
    @patch("app.refine_intent_with_llm")
    def test_needs_clarification_returns_clarification_response(self, mock_refine):
        mock_refine.return_value = {
            "status": "needs_clarification",
            "user_goal": "access onboarding thing",
            "normalized_query": "access onboarding system",
            "likely_service_area": None,
            "request_type": "access_request",
            "missing_information": ["system name"],
            "clarifying_question": "Which onboarding tool, site, form, or system do you need access to?",
            "confidence": 0.42,
            "ranking_keywords": ["access", "permissions"],
        }
        response = self.client.post("/api/chat", json={"message": "I need access to the thing for onboarding new staff"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["type"], "clarification")
        self.assertIn("Which onboarding", data["question"])
        self.assertEqual(data["intent_state"]["turn_count"], 1)

    @patch.dict(os.environ, {
        "COHERE_ENABLED": "false",
        "INTENT_REFINEMENT_ENABLED": "true",
        "INTENT_REFINEMENT_DEBUG": "true",
        "AZURE_FOUNDRY_ENDPOINT": "https://example.test/models",
        "AZURE_FOUNDRY_API_KEY": "secret",
        "AZURE_FOUNDRY_MODEL": "DeepSeek-V4-Pro",
    }, clear=True)
    @patch("app.refine_intent_with_llm")
    def test_ready_to_rank_uses_catalogue_ranking_with_improved_query(self, mock_refine):
        mock_refine.return_value = {
            "status": "ready_to_rank",
            "user_goal": "request SharePoint access",
            "normalized_query": "request access to SharePoint site for new employee setup onboarding requests",
            "likely_service_area": "SharePoint",
            "request_type": "access_request",
            "missing_information": [],
            "clarifying_question": None,
            "confidence": 0.9,
            "ranking_keywords": ["new employee setup", "onboarding"],
        }
        response = self.client.post("/api/chat", json={"message": "The SharePoint site where managers submit new employee setup requests"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["type"], "ranked_results")
        self.assertIn("SharePoint", data["ranking_query"])
        catalogue_ids = {action.action_id for action in app.ACTIONS}
        self.assertTrue({c["action_id"] for c in data["candidates"]}.issubset(catalogue_ids))


    @patch.dict(os.environ, {
        "COHERE_ENABLED": "false",
        "INTENT_REFINEMENT_ENABLED": "true",
        "AZURE_FOUNDRY_ENDPOINT": "https://example.test/models",
        "AZURE_FOUNDRY_API_KEY": "secret",
        "AZURE_FOUNDRY_MODEL": "DeepSeek-V4-Pro",
    }, clear=True)
    @patch("app.refine_intent_with_llm")
    def test_out_of_scope_intent_does_not_return_catalogue_matches(self, mock_refine):
        mock_refine.return_value = {
            "status": "out_of_scope",
            "user_goal": "get a puppy",
            "normalized_query": "",
            "likely_service_area": None,
            "request_type": None,
            "missing_information": [],
            "clarifying_question": None,
            "confidence": 0.95,
            "ranking_keywords": [],
        }
        response = self.client.post("/api/chat", json={"message": "I need a puppy"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["type"], "out_of_scope")
        self.assertEqual(data["candidates"], [])
        self.assertIsNone(data["selected_action_id"])
        self.assertEqual(data["selection_source"], "intent_refinement")

    @patch.dict(os.environ, {
        "COHERE_ENABLED": "false",
        "INTENT_REFINEMENT_ENABLED": "true",
        "AZURE_FOUNDRY_ENDPOINT": "https://example.test/models",
        "AZURE_FOUNDRY_API_KEY": "secret",
        "AZURE_FOUNDRY_MODEL": "DeepSeek-V4-Pro",
    }, clear=True)
    @patch("app.refine_intent_with_llm", return_value=None)
    def test_llm_failure_does_not_crash_chat(self, _mock_refine):
        response = self.client.post("/api/chat", json={"message": "I need access to the thing"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("candidates", data)
        self.assertEqual(data["selection_source"], "local")


if __name__ == "__main__":
    unittest.main()
