"""
Tests for the local human-labeling workflow (src.labeling + src.label_golden_set).

All tests use a tiny synthetic Golden Set written to a tmp dir.  No real data
files and no AI calls are involved.  The only repo file touched is
config/intents.yaml, which is optional and only READ.
"""

import json
import subprocess
import sys

import pytest

from src import labeling as L
from src.agent.escalation import EscalationDecision
from src.labeling import ESCALATION_VALUES


# ---------------------------------------------------------------------------
# Synthetic Golden Set helpers
# ---------------------------------------------------------------------------

def _record(golden_id, conversation_id):
    return {
        "golden_id": golden_id,
        "conversation_id": conversation_id,
        "customer_message": "My laptop will not start after the update.",
        "conversation_context": "MICROSOFT (1): How can we help?",
        "microsoft_responses": ["Hello, how can we help?", "Try a clean boot."],
        "source_tweets": [
            {"tweet_id": 1, "author_id": "MicrosoftHelps", "inbound": False,
             "text": "How can we help?", "created_at": "Tue Jan 01 12:00:00 +0000 2018"},
            {"tweet_id": 2, "author_id": "customer", "inbound": True,
             "text": "My laptop will not start after the update.",
             "created_at": "Tue Jan 01 12:01:00 +0000 2018"},
        ],
        "metadata": {"brand": "MicrosoftHelps", "num_brand_turns": 1,
                     "num_customer_turns": 1},
        "intent_label": "",
        "escalation_label": "",
        "notes": "",
    }


def _synthetic_records(n=3):
    return [_record(f"GOLDEN-{i:04d}", 100 + i) for i in range(1, n + 1)]


def write_golden_set(tmp_path, records):
    payload = {
        "schema_version": "1.0.0",
        "source": "synthetic",
        "count": len(records),
        "label_fields": ["intent_label", "escalation_label"],
        "records": records,
    }
    golden_path = tmp_path / "golden_set.json"
    golden_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(golden_path), payload


def _load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parent(path):
    from pathlib import Path
    return Path(path).parent


@pytest.fixture
def taxa():
    """Valid 10-intent taxonomy names from config/intents.yaml (read-only)."""
    return L.load_intent_names()


@pytest.fixture
def tiny_set(tmp_path):
    golden_path, payload = write_golden_set(tmp_path, _synthetic_records(3))
    labels_path = str(tmp_path / "golden_set.labels.json")
    return golden_path, labels_path, payload


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_golden_set_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            L.load_golden_set(str(tmp_path / "missing.json"))

    def test_load_golden_set_returns_payload(self, tiny_set):
        golden_path, _, payload = tiny_set
        loaded = L.load_golden_set(golden_path)
        assert loaded["count"] == 3
        assert len(L.golden_records(loaded)) == 3
        assert loaded["records"][0]["golden_id"] == payload["records"][0]["golden_id"]

    def test_golden_records_rejects_missing_records_list(self):
        with pytest.raises(ValueError):
            L.golden_records({"count": 0})

    def test_load_labels_returns_empty_store_when_missing(self, tmp_path):
        store = L.load_labels(str(tmp_path / "nope.json"))
        assert store == L.empty_labels_store(str(tmp_path / "nope.json"))
        assert store["labels"] == {}
        assert store["count_labeled"] == 0

    def test_load_labels_round_trip(self, tiny_set, taxa):
        _, labels_path, _ = tiny_set
        store = L.load_labels(labels_path)
        rec = _record("GOLDEN-0001", 101)
        L.set_label(store, rec, "Technical Troubleshooting",
                    "AUTO_HANDLE", "clear case", valid_intents=taxa)
        L.save_labels(store, labels_path)

        reloaded = L.load_labels(labels_path)
        assert reloaded["labels"]["GOLDEN-0001"]["intent_label"] == \
            "Technical Troubleshooting"
        assert reloaded["labels"]["GOLDEN-0001"]["escalation_label"] == "AUTO_HANDLE"
        assert reloaded["labels"]["GOLDEN-0001"]["notes"] == "clear case"
        assert reloaded["count_labeled"] == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_label_has_no_errors(self, taxa):
        assert L.validate_label("Account & Login", "ESCALATE_TO_HUMAN",
                                None, taxa) == []

    def test_missing_intent_reported(self, taxa):
        errors = L.validate_label("", "AUTO_HANDLE", None, taxa)
        assert any("intent_label" in e for e in errors)

    def test_invalid_intent_reported(self, taxa):
        errors = L.validate_label("Made Up Intent", "AUTO_HANDLE", None, taxa)
        assert any("intent_label" in e for e in errors)

    def test_invalid_escalation_reported(self, taxa):
        errors = L.validate_label("Billing & Payments", "AUTO", None, taxa)
        assert any("escalation_label" in e for e in errors)

    def test_notes_optional(self, taxa):
        assert L.validate_label("Warranty / Repair", "AUTO_HANDLE",
                                None, taxa) == []
        assert L.validate_label("Warranty / Repair", "AUTO_HANDLE",
                                "note here", taxa) == []
        assert L.validate_label("Warranty / Repair", "AUTO_HANDLE",
                                "", taxa) == []

    def test_set_label_writes_human_entry_with_timestamps(self, tiny_set, taxa):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        entry = L.set_label(
            store,
            payload["records"][0],
            "Technical Troubleshooting",
            "AUTO_HANDLE",
            notes="",
            valid_intents=taxa,
        )
        assert entry["golden_id"] == "GOLDEN-0001"
        assert entry["conversation_id"] == 101
        assert entry["labeled_by"] == "human"
        assert entry["labeled_at"]
        assert entry["modified_at"]
        assert store["count_labeled"] == 1

    def test_set_label_invalid_raises_value_error(self, tiny_set, taxa):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        with pytest.raises(ValueError):
            L.set_label(store, payload["records"][0], "Nope", "MAYBE",
                        valid_intents=taxa)

    def test_escalation_values_match_agent_enum(self):
        assert ESCALATION_VALUES == sorted(
            [EscalationDecision.AUTO_HANDLE.value,
             EscalationDecision.ESCALATE_TO_HUMAN.value]
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_labels_creates_file(self, tiny_set, taxa):
        _, labels_path, payload = tiny_set
        store = L.empty_labels_store(source="synthetic")
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        path = L.save_labels(store, labels_path)
        assert path.exists()
        assert _load_raw(labels_path)["labels"]["GOLDEN-0001"]["intent_label"] == \
            "Order / Delivery"

    def test_incremental_saves_preserve_prior_labels(self, tiny_set, taxa):
        _, labels_path, payload = tiny_set
        store = L.empty_labels_store()
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        L.save_labels(store, labels_path)
        # Second record labeled and saved again; first must survive.
        L.set_label(store, payload["records"][1], "Account & Login",
                    "ESCALATE_TO_HUMAN", valid_intents=taxa)
        L.save_labels(store, labels_path)

        reloaded = L.load_labels(labels_path)
        assert reloaded["count_labeled"] == 2
        assert reloaded["labels"]["GOLDEN-0001"]["intent_label"] == "Order / Delivery"
        assert reloaded["labels"]["GOLDEN-0002"]["intent_label"] == "Account & Login"

    def test_save_does_not_modify_golden_set(self, tiny_set, taxa):
        golden_path, labels_path, payload = tiny_set
        before = open(golden_path, "rb").read()
        store = L.empty_labels_store()
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        L.save_labels(store, labels_path)
        after = open(golden_path, "rb").read()
        assert before == after

    def test_count_labeled_counts_only_complete_entries(self):
        store = L.empty_labels_store()
        store["labels"]["GOLDEN-0001"] = {
            "golden_id": "GOLDEN-0001", "intent_label": "Account & Login",
            "escalation_label": "", "notes": "",
        }
        assert L.count_labeled(store) == 0
        store["labels"]["GOLDEN-0001"]["escalation_label"] = "AUTO_HANDLE"
        assert L.count_labeled(store) == 1

    def test_atomic_save_leaves_no_temp_files(self, tiny_set):
        _, labels_path, _ = tiny_set
        store = L.empty_labels_store()
        L.save_labels(store, labels_path)
        leftovers = list(_parent(labels_path).glob("golden_set.labels.json.*"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# Resume behaviour
# ---------------------------------------------------------------------------

class TestResume:
    def test_first_unlabeled_index_is_zero_on_fresh_store(self, tiny_set):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        assert L.first_unlabeled_index(L.golden_records(payload), store) == 0

    def test_first_unlabeled_advances_after_labeling_front(self, tiny_set, taxa):
        _, _, payload = tiny_set
        records = L.golden_records(payload)
        store = L.empty_labels_store()
        L.set_label(store, records[0], "Order / Delivery", "AUTO_HANDLE",
                    valid_intents=taxa)
        assert L.first_unlabeled_index(records, store) == 1

    def test_resumes_into_gap_in_the_middle(self, tiny_set, taxa):
        _, _, payload = tiny_set
        records = L.golden_records(payload)
        store = L.empty_labels_store()
        L.set_label(store, records[0], "Order / Delivery", "AUTO_HANDLE",
                    valid_intents=taxa)
        L.set_label(store, records[2], "Microsoft Store", "AUTO_HANDLE",
                    valid_intents=taxa)
        # Record 2 (index 1) is the earliest unlabeled -> resume point.
        assert L.first_unlabeled_index(records, store) == 1

    def test_all_labeled_returns_length(self, tiny_set, taxa):
        _, _, payload = tiny_set
        records = L.golden_records(payload)
        store = L.empty_labels_store()
        for rec in records:
            L.set_label(store, rec, "Order / Delivery", "AUTO_HANDLE",
                        valid_intents=taxa)
        assert L.first_unlabeled_index(records, store) == len(records)

    def test_intent_only_record_is_not_labeled(self):
        store = L.empty_labels_store()
        store["labels"]["GOLDEN-0001"] = {
            "golden_id": "GOLDEN-0001", "intent_label": "Account & Login",
            "escalation_label": "", "notes": "",
        }
        assert L.is_record_labeled(store, "GOLDEN-0001") is False

    def test_first_unlabeled_record_helper(self, tiny_set, taxa):
        _, _, payload = tiny_set
        records = L.golden_records(payload)
        store = L.empty_labels_store()
        L.set_label(store, records[0], "Order / Delivery", "AUTO_HANDLE",
                    valid_intents=taxa)
        rec = L.first_unlabeled_record(records, store)
        assert rec is not None
        assert rec["golden_id"] == "GOLDEN-0002"


# ---------------------------------------------------------------------------
# Merge / export
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_fills_labels_and_preserves_source(self, tiny_set, taxa):
        golden_path, labels_path, payload = tiny_set
        store = L.empty_labels_store(source=golden_path)
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", "shipping question", valid_intents=taxa)
        L.save_labels(store, labels_path)

        merged = L.merge_labels(payload, L.load_labels(labels_path))
        target = merged[0]
        assert target["intent_label"] == "Order / Delivery"
        assert target["escalation_label"] == "AUTO_HANDLE"
        assert target["notes"] == "shipping question"
        assert target["customer_message"] == "My laptop will not start after the update."
        assert target["conversation_context"] == "MICROSOFT (1): How can we help?"
        assert target["microsoft_responses"] == [
            "Hello, how can we help?", "Try a clean boot."]
        assert target["metadata"] == {"brand": "MicrosoftHelps",
                                      "num_brand_turns": 1,
                                      "num_customer_turns": 1}
        assert len(target["source_tweets"]) == 2

    def test_merge_leaves_unlabeled_records_empty(self, tiny_set, taxa):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        merged = L.merge_labels(payload, store)
        assert merged[1]["intent_label"] == ""
        assert merged[1]["escalation_label"] == ""
        assert merged[1]["notes"] == ""

    def test_merge_does_not_mutate_input_payload(self, tiny_set, taxa):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        before = json.dumps(payload, sort_keys=True)
        L.merge_labels(payload, store)
        assert json.dumps(payload, sort_keys=True) == before

    def test_export_writes_separate_file_without_touching_source(
            self, tiny_set, tmp_path, taxa):
        golden_path, labels_path, payload = tiny_set
        store = L.empty_labels_store(source=golden_path)
        L.set_label(store, payload["records"][0], "Order / Delivery",
                    "AUTO_HANDLE", valid_intents=taxa)
        L.save_labels(store, labels_path)
        before = open(golden_path, "rb").read()

        dest = L.export_labeled_set(
            payload, L.load_labels(labels_path),
            output_path=str(tmp_path / "out" / "golden_set.labeled.json"),
        )
        out = _load_raw(str(dest))
        assert out["count"] == 3
        assert out["records"][0]["intent_label"] == "Order / Delivery"
        assert open(golden_path, "rb").read() == before


# ---------------------------------------------------------------------------
# The workflow must never auto-generate labels or call AI models
# ---------------------------------------------------------------------------

class TestNeverAutoGenerates:
    def test_labeling_module_has_no_model_imports(self):
        source = open(L.__file__, "r", encoding="utf-8").read()
        assert "google.generativeai" not in source
        assert "genai" not in source
        assert "openai" not in source
        assert "anthropic" not in source

    def test_labeling_module_has_no_random_suggestion_logic(self):
        source = open(L.__file__, "r", encoding="utf-8").read()
        for token in ("random.", "random(", "shuffle", ".sample("):
            assert token not in source

    def test_cli_module_has_no_model_imports(self):
        import src.label_golden_set as cli
        source = open(cli.__file__, "r", encoding="utf-8").read()
        assert "google.generativeai" not in source
        assert "genai" not in source
        assert "openai" not in source

    def test_set_label_requires_human_values(self, tiny_set, taxa):
        _, _, payload = tiny_set
        store = L.empty_labels_store()
        # No default/suggestion machinery: calling with nothing raises because
        # the intent and escalation must be supplied by a human.
        with pytest.raises(Exception):
            L.set_label(store, payload["records"][0], "", "",
                        valid_intents=taxa)


# ---------------------------------------------------------------------------
# CLI (non-interactive paths, via subprocess against a synthetic set)
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_list_reports_progress(self, tmp_path):
        golden_path, _payload = write_golden_set(tmp_path, _synthetic_records(3))
        labels_path = str(tmp_path / "labels.json")
        result = subprocess.run(
            [sys.executable, "-m", "src.label_golden_set", "--list",
             "--golden-path", golden_path, "--labels-path", labels_path],
            cwd=str(L.PROJECT_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "0 of 3 labeled" in result.stdout
        assert "GOLDEN-0001" in result.stdout

    def test_cli_unknown_record_exits_nonzero(self, tmp_path):
        golden_path, _payload = write_golden_set(tmp_path, _synthetic_records(3))
        labels_path = str(tmp_path / "labels.json")
        result = subprocess.run(
            [sys.executable, "-m", "src.label_golden_set", "--record", "GOLDEN-9999",
             "--golden-path", golden_path, "--labels-path", labels_path],
            cwd=str(L.PROJECT_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1
        assert "Unknown golden_id" in result.stderr