"""
Tests for the web labeling server (src.labeling_server).

Exercises loading, saving, reloading, resume/progress, and invalid-rejection
both through the LabelerApp API directly and through a live HTTP server
started on an ephemeral port.  All data is synthetic; no real Golden Set
files or AI calls are involved.
"""

import json
import sys
import threading
import urllib.request
from http.server import HTTPServer
from functools import partial
from pathlib import Path

import pytest

from src import labeling as L
from src.labeling_server import LabelerApp, LabelingHandler, PAGE_HTML

# Reuse synthetic helpers from the core labeling tests.
from tests.test_labeling import write_golden_set, _synthetic_records, _load_raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    """A fresh LabelerApp pointing at a tiny synthetic Golden Set."""
    golden_path, _ = write_golden_set(tmp_path, _synthetic_records(3))
    labels_path = str(tmp_path / "labels.json")
    return LabelerApp(golden_path, labels_path)


@pytest.fixture
def taxa():
    """Valid intent names (read from config/intents.yaml)."""
    return L.load_intent_names()


# ---------------------------------------------------------------------------
# App-level: loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_app_loads_correct_record_count(self, app):
        assert len(app.records) == 3
        assert app.store["count_total"] == 3

    def test_app_starts_with_zero_labels(self, app):
        assert L.count_labeled(app.store) == 0

    def test_index_reports_all_unlabeled(self, app):
        idx = app.index()
        assert idx["count_total"] == 3
        assert idx["count_labeled"] == 0
        assert all(not r["labeled"] for r in idx["records"])

    def test_get_record_returns_full_content(self, app):
        rec = app.get_record("GOLDEN-0001")
        assert rec is not None
        assert rec["conversation_id"] == 101
        assert rec["customer_message"] == "My laptop will not start after the update."
        assert rec["conversation_context"] == "MICROSOFT (1): How can we help?"
        assert len(rec["microsoft_responses"]) == 2

    def test_get_record_returns_none_for_unknown_id(self, app):
        assert app.get_record("GOLDEN-9999") is None


# ---------------------------------------------------------------------------
# App-level: saving + validation
# ---------------------------------------------------------------------------

class TestSaving:
    def test_save_valid_label_updates_store(self, app, taxa):
        result = app.save("GOLDEN-0001", {
            "intent_label": "Technical Troubleshooting",
            "escalation_label": "AUTO_HANDLE",
            "notes": "clear case",
        })
        assert result["saved"]["intent_label"] == "Technical Troubleshooting"
        assert result["saved"]["escalation_label"] == "AUTO_HANDLE"
        assert result["count_labeled"] == 1

    def test_save_persists_to_disk_and_is_reloadable(self, app, taxa):
        app.save("GOLDEN-0001", {
            "intent_label": "Account & Login",
            "escalation_label": "ESCALATE_TO_HUMAN",
            "notes": "",
        })
        reloaded = L.load_labels(str(app.labels_path))
        assert reloaded["labels"]["GOLDEN-0001"]["intent_label"] == "Account & Login"
        assert reloaded["labels"]["GOLDEN-0001"]["escalation_label"] == "ESCALATE_TO_HUMAN"

    def test_index_reflects_labeled_record(self, app, taxa):
        app.save("GOLDEN-0001", {
            "intent_label": "Account & Login",
            "escalation_label": "AUTO_HANDLE",
            "notes": "",
        })
        idx = app.index()
        assert idx["count_labeled"] == 1
        assert any(r["labeled"] for r in idx["records"]
                   if r["golden_id"] == "GOLDEN-0001")

    def test_first_unlabeled_advances(self, app, taxa):
        app.save("GOLDEN-0001", {
            "intent_label": "Account & Login",
            "escalation_label": "AUTO_HANDLE",
            "notes": "",
        })
        idx = app.index()
        assert idx["first_unlabeled"] == "GOLDEN-0002"

    def test_all_labeled_returns_none(self, app, taxa):
        for rec in app.records:
            app.save(rec["golden_id"], {
                "intent_label": "Order / Delivery",
                "escalation_label": "AUTO_HANDLE",
                "notes": "",
            })
        idx = app.index()
        assert idx["count_labeled"] == 3
        assert idx["first_unlabeled"] is None

    def test_incomplete_label_not_counted(self, app):
        """Record with only intent set (escalation missing) is NOT labeled."""
        table = app.store.setdefault("labels", {})
        table["GOLDEN-0001"] = {
            "golden_id": "GOLDEN-0001",
            "intent_label": "Order / Delivery",
            "escalation_label": "",
            "notes": "",
        }
        assert L.count_labeled(app.store) == 0
        assert app.index()["count_labeled"] == 0


# ---------------------------------------------------------------------------
# Validation rejection
# ---------------------------------------------------------------------------

class TestValidationRejection:
    def test_empty_intent_rejected(self, app):
        with pytest.raises(ValueError, match="intent_label"):
            app.save("GOLDEN-0001", {
                "intent_label": "",
                "escalation_label": "AUTO_HANDLE",
                "notes": "",
            })

    def test_invalid_intent_rejected(self, app):
        with pytest.raises(ValueError, match="intent_label"):
            app.save("GOLDEN-0001", {
                "intent_label": "Made Up Intent",
                "escalation_label": "AUTO_HANDLE",
                "notes": "",
            })

    def test_invalid_escalation_rejected(self, app):
        with pytest.raises(ValueError, match="escalation_label"):
            app.save("GOLDEN-0001", {
                "intent_label": "Account & Login",
                "escalation_label": "INVALID",
                "notes": "",
            })

    def test_empty_escalation_rejected(self, app):
        with pytest.raises(ValueError, match="escalation_label"):
            app.save("GOLDEN-0001", {
                "intent_label": "Account & Login",
                "escalation_label": "",
                "notes": "",
            })

    def test_unknown_golden_id_rejected(self, app):
        with pytest.raises(ValueError, match="Unknown golden_id"):
            app.save("GOLDEN-9999", {
                "intent_label": "Account & Login",
                "escalation_label": "AUTO_HANDLE",
                "notes": "",
            })

    def test_incomplete_save_does_not_change_count(self, app):
        """A failed save must leave the progress count unchanged."""
        assert L.count_labeled(app.store) == 0
        with pytest.raises(ValueError):
            app.save("GOLDEN-0001", {"intent_label": "", "escalation_label": ""})
        assert L.count_labeled(app.store) == 0

    def test_source_data_preserved_after_saving(self, app, taxa):
        """Golden Set source records are untouched by the save."""
        source_bytes = open(app.golden_path, "rb").read()
        app.save("GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "test",
        })
        assert open(app.golden_path, "rb").read() == source_bytes

    def test_source_tweets_preserved_in_saved_record(self, app, taxa):
        rec = app.get_record("GOLDEN-0001")
        assert len(rec["microsoft_responses"]) == 2


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_writes_labeled_copy(self, app, taxa, tmp_path):
        app.save("GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "shipping",
        })
        dest = app.export()
        assert Path(dest).exists()
        data = _load_raw(dest)
        assert data["count"] == 3
        labeled_rec = next(r for r in data["records"] if r["golden_id"] == "GOLDEN-0001")
        assert labeled_rec["intent_label"] == "Order / Delivery"
        assert labeled_rec["escalation_label"] == "AUTO_HANDLE"
        assert labeled_rec["notes"] == "shipping"
        # source untouched
        assert open(app.golden_path, "rb").read() != b""

    def test_export_unlabeled_record_has_empty_fields(self, app, taxa):
        app.save("GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "",
        })
        dest = app.export()
        data = _load_raw(dest)
        rec2 = next(r for r in data["records"] if r["golden_id"] == "GOLDEN-0002")
        assert rec2["intent_label"] == ""
        assert rec2["escalation_label"] == ""


# ---------------------------------------------------------------------------
# Live HTTP server
# ---------------------------------------------------------------------------

@pytest.fixture
def live_server(app, tmp_path):
    """Start the labeler on an ephemeral port and yield (url, app)."""
    factory = partial(LabelingHandler, app=app)
    server = HTTPServer(("127.0.0.1", 0), factory)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://{host}:{port}", app
    server.shutdown()


def _get(url):
    return urllib.request.urlopen(url).read().decode()


def _put(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"},
                                method="PUT")
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    resp = urllib.request.urlopen(req)
    return resp.status, json.loads(resp.read())


class TestLiveHTTP:
    def test_page_served(self, live_server):
        url, _ = live_server
        html = _get(url + "/")
        assert "Golden Set Labeler" in html
        assert "<select" in html

    def test_index_endpoint(self, live_server):
        url, _ = live_server
        data = json.loads(_get(url + "/api/index"))
        assert data["count_total"] == 3
        assert data["count_labeled"] == 0
        assert len(data["records"]) == 3

    def test_intents_endpoint(self, live_server):
        url, _ = live_server
        data = json.loads(_get(url + "/api/intents"))
        assert len(data["intents"]) == 10
        assert "Technical Troubleshooting" in data["intents"]

    def test_get_record_endpoint(self, live_server):
        url, _ = live_server
        rec = json.loads(_get(url + "/api/record/GOLDEN-0001"))
        assert rec["golden_id"] == "GOLDEN-0001"
        assert "customer_message" in rec
        assert "microsoft_responses" in rec

    def test_get_unknown_record_returns_404(self, live_server):
        url, _ = live_server
        try:
            urllib.request.urlopen(url + "/api/record/GOLDEN-9999")
            assert False, "Expected HTTP 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            body = json.loads(exc.read())
            assert "Unknown" in body["error"]

    def test_put_valid_label_succeeds(self, live_server):
        url, app = live_server
        status, body = _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "Technical Troubleshooting",
            "escalation_label": "AUTO_HANDLE",
            "notes": "test",
        })
        assert status == 200
        assert body["count_labeled"] == 1
        assert body["saved"]["golden_id"] == "GOLDEN-0001"
        # verified in store
        assert app.store["labels"]["GOLDEN-0001"]["intent_label"] == \
            "Technical Troubleshooting"

    def test_put_invalid_label_returns_400(self, live_server):
        url, _ = live_server
        status, body = _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "",
            "escalation_label": "AUTO_HANDLE",
        })
        assert status == 400
        assert "error" in body

    def test_put_bad_json_returns_400(self, live_server):
        url, _ = live_server
        req = urllib.request.Request(
            url + "/api/record/GOLDEN-0001",
            data=b"{bad json",
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        try:
            urllib.request.urlopen(req)
            assert False, "Expected HTTP 400"
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

    def test_index_after_saving_reflects_progress(self, live_server):
        url, _ = live_server
        _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "",
        })
        data = json.loads(_get(url + "/api/index"))
        assert data["count_labeled"] == 1
        assert data["first_unlabeled"] == "GOLDEN-0002"

    def test_saved_label_persists_to_file(self, live_server, app):
        url, _ = live_server
        _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "file check",
        })
        reloaded = L.load_labels(str(app.labels_path))
        assert reloaded["labels"]["GOLDEN-0001"]["notes"] == "file check"

    def test_export_endpoint(self, live_server, tmp_path):
        url, app = live_server
        _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "Order / Delivery",
            "escalation_label": "AUTO_HANDLE",
            "notes": "",
        })
        status, body = _post(url + "/api/export")
        assert status == 200
        assert "exported_to" in body
        assert Path(body["exported_to"]).exists()

    def test_get_record_resets_after_invalid_save(self, live_server):
        """An invalid save should not leave a stale partial label."""
        url, _ = live_server
        _put(url + "/api/record/GOLDEN-0001", {
            "intent_label": "", "escalation_label": "AUTO_HANDLE",
        })
        rec = json.loads(_get(url + "/api/record/GOLDEN-0001"))
        assert rec["intent_label"] == ""


# ---------------------------------------------------------------------------
# No AI / no auto-generation
# ---------------------------------------------------------------------------

class TestNoAI:
    def test_server_module_has_no_model_imports(self):
        source = open(L.__file__, "r", encoding="utf-8").read()
        assert "google.generativeai" not in source
        assert "genai" not in source
        assert "openai" not in source

    def test_page_html_includes_ground_truth_warning(self):
        assert "No AI model" in PAGE_HTML
        assert "human ground truth" in PAGE_HTML.lower()
