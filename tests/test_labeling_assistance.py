"""
Tests for AI-assisted human-labeling workflow (src.suggest_labels and labeling integration).

All tests use synthetic records and injected fake clients / responses.
No real Gemini API calls are made and no API key is required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import labeling as L
from src.labeling_server import LabelerApp
from src.suggest_labels import (
    ConfigurationError,
    InvalidSuggestionResponseError,
    LabelSuggester,
    LabelSuggestion,
    empty_suggestions_store,
    generate_all_suggestions,
    load_suggestions_store,
    save_suggestions_store,
)
from tests.test_labeling import _record, _synthetic_records, write_golden_set


class FakeGeminiClient:
    """Mock Gemini client for deterministic testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = None

    def generate_content(self, prompt: str):
        self.call_count += 1
        self.last_prompt = prompt
        mock_resp = MagicMock()
        mock_resp.text = self.response_text
        return mock_resp


@pytest.fixture
def taxa():
    return L.load_intent_names()


@pytest.fixture
def valid_suggestion_json():
    return json.dumps({
        "suggested_intent": "Technical Troubleshooting",
        "suggested_escalation": "AUTO_HANDLE",
        "reason": "Customer needs help updating display drivers.",
        "confidence": 0.95,
    })


# ---------------------------------------------------------------------------
# Prompt construction & isolation tests
# ---------------------------------------------------------------------------

class TestSuggesterPrompt:
    def test_prompt_contains_required_fields(self):
        suggester = LabelSuggester()
        prompt = suggester.build_prompt(
            customer_message="My screen is flickering",
            conversation_context="MICROSOFT (1): Hello\nCUSTOMER (2): Hi",
            microsoft_responses=["Please update drivers."],
        )
        assert "My screen is flickering" in prompt
        assert "MICROSOFT (1): Hello" in prompt
        assert "Please update drivers." in prompt
        assert "Technical Troubleshooting" in prompt
        assert "AUTO_HANDLE" in prompt
        assert "ESCALATE_TO_HUMAN" in prompt

    def test_prompt_never_contains_human_labels(self):
        """Ensure human labels are never leaked into the prompt."""
        rec_with_labels = {
            "golden_id": "GOLDEN-0001",
            "conversation_id": 101,
            "customer_message": "Can't sign in",
            "conversation_context": "MICROSOFT: How can we help?",
            "microsoft_responses": ["Reset your password."],
            "intent_label": "SECRET_HUMAN_INTENT",
            "escalation_label": "SECRET_HUMAN_ESCALATION",
            "notes": "SECRET_HUMAN_NOTE",
        }
        fake_client = FakeGeminiClient(json.dumps({
            "suggested_intent": "Account & Login",
            "suggested_escalation": "ESCALATE_TO_HUMAN",
            "reason": "Login issue",
            "confidence": 0.9,
        }))
        suggester = LabelSuggester(client=fake_client)
        suggester.suggest_for_record(rec_with_labels)

        prompt = fake_client.last_prompt
        assert "SECRET_HUMAN_INTENT" not in prompt
        assert "SECRET_HUMAN_ESCALATION" not in prompt
        assert "SECRET_HUMAN_NOTE" not in prompt
        assert "intent_label" not in prompt
        assert "escalation_label" not in prompt


# ---------------------------------------------------------------------------
# Validation & response parsing tests
# ---------------------------------------------------------------------------

class TestSuggesterValidation:
    def test_valid_ai_suggestion(self, valid_suggestion_json):
        client = FakeGeminiClient(valid_suggestion_json)
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        sug = suggester.suggest_for_record(rec)

        assert isinstance(sug, LabelSuggestion)
        assert sug.golden_id == "GOLDEN-0001"
        assert sug.conversation_id == 101
        assert sug.suggested_intent == "Technical Troubleshooting"
        assert sug.suggested_escalation == "AUTO_HANDLE"
        assert sug.confidence == 0.95
        assert "display drivers" in sug.reason

    def test_invalid_intent_raises(self):
        client = FakeGeminiClient(json.dumps({
            "suggested_intent": "Invented Intent Name",
            "suggested_escalation": "AUTO_HANDLE",
            "reason": "test",
            "confidence": 0.9,
        }))
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(InvalidSuggestionResponseError, match="suggested_intent"):
            suggester.suggest_for_record(rec)

    def test_invalid_escalation_raises(self):
        client = FakeGeminiClient(json.dumps({
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "MAYBE_ESCALATE",
            "reason": "test",
            "confidence": 0.9,
        }))
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(InvalidSuggestionResponseError, match="suggested_escalation"):
            suggester.suggest_for_record(rec)

    def test_malformed_json_raises(self):
        client = FakeGeminiClient("This is not valid JSON")
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(InvalidSuggestionResponseError, match="malformed JSON"):
            suggester.suggest_for_record(rec)

    def test_empty_response_raises(self):
        client = FakeGeminiClient("")
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(InvalidSuggestionResponseError, match="empty response"):
            suggester.suggest_for_record(rec)

    def test_confidence_clamping(self):
        client = FakeGeminiClient(json.dumps({
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "AUTO_HANDLE",
            "reason": "test",
            "confidence": 1.5,
        }))
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        sug = suggester.suggest_for_record(rec)
        assert sug.confidence == 1.0

    def test_non_numeric_confidence_raises(self):
        client = FakeGeminiClient(json.dumps({
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "AUTO_HANDLE",
            "reason": "test",
            "confidence": "high",
        }))
        suggester = LabelSuggester(client=client)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(InvalidSuggestionResponseError, match="Non-numeric confidence"):
            suggester.suggest_for_record(rec)

    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        suggester = LabelSuggester(api_key=None, client=None)
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is not set"):
            suggester.suggest_for_record(rec)


# ---------------------------------------------------------------------------
# Caching & store tests
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cached_suggestion_reuses_without_calling_gemini(self, valid_suggestion_json):
        client = FakeGeminiClient(valid_suggestion_json)
        suggester = LabelSuggester(client=client)
        store = empty_suggestions_store()
        rec = _record("GOLDEN-0001", 101)

        # First call: invokes client and populates store
        sug1 = suggester.suggest_for_record(rec, suggestions_store=store)
        assert client.call_count == 1
        assert sug1.suggested_intent == "Technical Troubleshooting"

        # Second call with store: client NOT called again
        sug2 = suggester.suggest_for_record(rec, suggestions_store=store)
        assert client.call_count == 1
        assert sug2.suggested_intent == "Technical Troubleshooting"

    def test_force_flag_bypasses_cache(self, valid_suggestion_json):
        client = FakeGeminiClient(valid_suggestion_json)
        suggester = LabelSuggester(client=client)
        store = empty_suggestions_store()
        rec = _record("GOLDEN-0001", 101)

        suggester.suggest_for_record(rec, suggestions_store=store)
        assert client.call_count == 1

        suggester.suggest_for_record(rec, suggestions_store=store, force=True)
        assert client.call_count == 2

    def test_suggestions_store_persistence(self, tmp_path, valid_suggestion_json):
        client = FakeGeminiClient(valid_suggestion_json)
        suggester = LabelSuggester(client=client)
        store = empty_suggestions_store()
        rec = _record("GOLDEN-0001", 101)

        suggester.suggest_for_record(rec, suggestions_store=store)
        save_path = tmp_path / "test.suggestions.json"
        save_suggestions_store(store, str(save_path))

        reloaded = load_suggestions_store(str(save_path))
        assert reloaded["count_suggested"] == 1
        assert "GOLDEN-0001" in reloaded["suggestions"]
        assert reloaded["suggestions"]["GOLDEN-0001"]["suggested_intent"] == "Technical Troubleshooting"


# ---------------------------------------------------------------------------
# Human decision & provenance auditing tests
# ---------------------------------------------------------------------------

class TestHumanAcceptanceAndModification:
    def test_human_acceptance_marks_accepted_source(self, taxa):
        store = L.empty_labels_store()
        rec = _record("GOLDEN-0001", 101)
        ai_sug = {
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "AUTO_HANDLE",
        }
        # Human accepts the suggestion
        entry = L.set_label(
            store,
            rec,
            intent_label="Technical Troubleshooting",
            escalation_label="AUTO_HANDLE",
            valid_intents=taxa,
            ai_suggestion=ai_sug,
        )
        assert entry["label_source"] == "human_accepted_ai"
        assert entry["intent_label"] == "Technical Troubleshooting"
        assert entry["escalation_label"] == "AUTO_HANDLE"
        assert entry["labeled_by"] == "human"

    def test_human_modification_marks_modified_source(self, taxa):
        store = L.empty_labels_store()
        rec = _record("GOLDEN-0001", 101)
        ai_sug = {
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "AUTO_HANDLE",
        }
        # Human reviews suggestion and changes escalation to ESCALATE_TO_HUMAN
        entry = L.set_label(
            store,
            rec,
            intent_label="Technical Troubleshooting",
            escalation_label="ESCALATE_TO_HUMAN",
            notes="Modified because customer was unhappy",
            valid_intents=taxa,
            ai_suggestion=ai_sug,
        )
        assert entry["label_source"] == "human_modified_ai"
        assert entry["intent_label"] == "Technical Troubleshooting"
        assert entry["escalation_label"] == "ESCALATE_TO_HUMAN"
        assert entry["notes"] == "Modified because customer was unhappy"

    def test_human_direct_when_no_ai_suggestion(self, taxa):
        store = L.empty_labels_store()
        rec = _record("GOLDEN-0001", 101)
        entry = L.set_label(
            store,
            rec,
            intent_label="Account & Login",
            escalation_label="ESCALATE_TO_HUMAN",
            valid_intents=taxa,
            ai_suggestion=None,
        )
        assert entry["label_source"] == "human_direct"

    def test_explicit_label_source_preserved(self, taxa):
        store = L.empty_labels_store()
        rec = _record("GOLDEN-0001", 101)
        entry = L.set_label(
            store,
            rec,
            intent_label="Account & Login",
            escalation_label="ESCALATE_TO_HUMAN",
            label_source="human_accepted_ai",
            valid_intents=taxa,
        )
        assert entry["label_source"] == "human_accepted_ai"

    def test_invalid_label_source_raises(self, taxa):
        store = L.empty_labels_store()
        rec = _record("GOLDEN-0001", 101)
        with pytest.raises(ValueError, match="label_source"):
            L.set_label(
                store,
                rec,
                intent_label="Account & Login",
                escalation_label="ESCALATE_TO_HUMAN",
                label_source="ai_automated_ground_truth",
                valid_intents=taxa,
            )


# ---------------------------------------------------------------------------
# Batch generation & server integration tests
# ---------------------------------------------------------------------------

class TestBatchAndServerIntegration:
    def test_generate_all_suggestions_batches(self, tmp_path, valid_suggestion_json):
        golden_path, _ = write_golden_set(tmp_path, _synthetic_records(4))
        suggestions_path = str(tmp_path / "suggestions.json")
        client = FakeGeminiClient(valid_suggestion_json)

        store = generate_all_suggestions(
            golden_path=golden_path,
            suggestions_path=suggestions_path,
            batch_size=2,
            delay=0.0,
            client=client,
        )
        assert store["count_suggested"] == 4
        assert client.call_count == 4

        # Rerun does not call client again (fully cached)
        store2 = generate_all_suggestions(
            golden_path=golden_path,
            suggestions_path=suggestions_path,
            batch_size=2,
            delay=0.0,
            client=client,
        )
        assert store2["count_suggested"] == 4
        assert client.call_count == 4  # Still 4!

    def test_server_serves_suggestion_and_saves_accepted_source(self, tmp_path, valid_suggestion_json):
        golden_path, _ = write_golden_set(tmp_path, _synthetic_records(2))
        labels_path = str(tmp_path / "labels.json")
        suggestions_path = str(tmp_path / "suggestions.json")

        sug_store = empty_suggestions_store(source=golden_path)
        sug_store["suggestions"]["GOLDEN-0001"] = {
            "golden_id": "GOLDEN-0001",
            "conversation_id": 101,
            "suggested_intent": "Technical Troubleshooting",
            "suggested_escalation": "AUTO_HANDLE",
            "reason": "Update drivers",
            "confidence": 0.95,
        }
        save_suggestions_store(sug_store, suggestions_path)

        app = LabelerApp(
            golden_path=golden_path,
            labels_path=labels_path,
            suggestions_path=suggestions_path,
        )

        # GET record returns suggestion
        rec = app.get_record("GOLDEN-0001")
        assert rec["suggestion"] is not None
        assert rec["suggestion"]["suggested_intent"] == "Technical Troubleshooting"

        # Save with accept
        save_res = app.save("GOLDEN-0001", {
            "intent_label": "Technical Troubleshooting",
            "escalation_label": "AUTO_HANDLE",
            "label_source": "human_accepted_ai",
            "notes": "",
        })
        assert save_res["saved"]["label_source"] == "human_accepted_ai"

        # Save record 2 without suggestion -> human_direct
        save_res2 = app.save("GOLDEN-0002", {
            "intent_label": "Account & Login",
            "escalation_label": "ESCALATE_TO_HUMAN",
            "notes": "",
        })
        assert save_res2["saved"]["label_source"] == "human_direct"
