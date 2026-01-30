"""
Functional tests for green agent using the REAL mutation dataset.

These tests verify that:
1. Green agent can process the actual mutated_dataset.csv
2. Results include proper metrics and statistics
3. Label distribution matches expected patterns
4. Sample rows are included for debugging
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Add parent directory to path for test_agent imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "tests"))

from test_agent import send_text_message


def _get_free_port() -> int:
    """Find an available port for mock purple server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout_s: float = 10.0) -> None:
    """Wait for server to be ready by polling agent card endpoint"""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(f"{url}/.well-known/agent-card.json", timeout=1.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server not ready: {url}")


@pytest.fixture()
def mock_purple():
    """
    Start mock purple server that returns realistic labels.
    
    For the real dataset, we'll make it return SAC most of the time
    since that's a common mutation type.
    """
    port = _get_free_port()
    url = f"http://127.0.0.1:{port}"

    script = Path(__file__).parent.parent / "tests" / "mock_purple_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), "--host", "127.0.0.1", "--port", str(port), "--card-url", f"{url}/"],
    )

    try:
        _wait_ready(url)
        httpx.post(f"{url}/debug/reset", timeout=2)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _extract_result_data(events):
    """
    Extract DataPart from task artifacts.
    
    Handles both streaming and non-streaming responses.
    """
    for event in events:
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            
            # Check task.artifacts directly (non-streaming)
            if hasattr(task, 'artifacts') and task.artifacts:
                for artifact in task.artifacts:
                    for part in getattr(artifact, 'parts', []) or []:
                        root = getattr(part, 'root', None)
                        if root and hasattr(root, 'data'):
                            return root.data
            
            # Also check update.artifact (streaming)
            if update:
                kind = getattr(update, 'kind', None)
                if kind == "artifact-update":
                    artifact = getattr(update, 'artifact', None)
                    if artifact:
                        for part in getattr(artifact, 'parts', []) or []:
                            root = getattr(part, 'root', None)
                            if root and hasattr(root, 'data'):
                                return root.data
    
    return None


@pytest.mark.asyncio
async def test_real_dataset_default_path(agent, mock_purple):
    """
    Test that green agent can load and process the default dataset.
    
    This verifies the dataset path resolution works correctly.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 5,  # Process just a few rows for speed
        },
    }

    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)

    data = _extract_result_data(events)
    assert data is not None, "Green agent did not emit a data artifact"

    # Verify basic structure
    assert "metrics" in data
    assert "label_stats" in data
    assert "config_used" in data
    
    # Verify it processed rows
    assert data["metrics"]["rows_evaluated"] == 5
    
    # Verify exactly 5 calls (1 per row)
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 5, f"Expected 5 purple calls for 5 rows, got {calls}"


@pytest.mark.asyncio
async def test_real_dataset_structure(agent, mock_purple):
    """
    Test that the result structure contains all expected fields.
    
    Think of this like checking that a restaurant menu has all the sections:
    appetizers, mains, desserts, drinks. We're verifying our output has
    all the "sections" a consumer would expect.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 3},
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Check top-level structure
    assert "metrics" in data
    assert "label_stats" in data
    assert "samples" in data
    assert "config_used" in data

    # Check metrics structure
    metrics = data["metrics"]
    assert "rows_evaluated" in metrics
    assert "correct" in metrics
    assert "accuracy" in metrics
    assert metrics["rows_evaluated"] == 3
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Check label_stats structure
    label_stats = data["label_stats"]
    assert "pred_counts" in label_stats
    assert "pred_correct_counts" in label_stats
    
    # Verify label counts are non-negative
    for label, count in label_stats["pred_counts"].items():
        assert count >= 0
        assert label in {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}

    # Check samples structure
    assert isinstance(data["samples"], list)
    assert len(data["samples"]) <= 3  # Should not exceed max_rows


@pytest.mark.asyncio
async def test_real_dataset_columns(agent, mock_purple):
    """
    Test that green agent correctly reads the dataset columns.
    
    The real dataset has specific column names that the green agent
    needs to handle properly.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 2,
            "ruleset_column": "ruleset",
            "gold_column": "trad_result",
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Verify config was used correctly
    config_used = data["config_used"]
    assert config_used["ruleset_column"] == "ruleset"
    assert config_used["gold_column"] == "trad_result"
    
    # Verify processing completed
    assert data["metrics"]["rows_evaluated"] == 2


@pytest.mark.asyncio
async def test_real_dataset_label_distribution(agent, mock_purple):
    """
    Test that label statistics are tracked correctly.
    
    Think of this like counting votes in an election - we need accurate
    tallies of how many times each candidate (label) was predicted.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 10},
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    label_stats = data["label_stats"]
    pred_counts = label_stats["pred_counts"]
    
    # Verify total predictions equals rows evaluated
    total_predictions = sum(pred_counts.values())
    assert total_predictions == data["metrics"]["rows_evaluated"]
    
    # Verify all predicted labels are valid
    for label in pred_counts.keys():
        assert label in {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}


@pytest.mark.asyncio
async def test_real_dataset_sample_rows(agent, mock_purple):
    """
    Test that sample rows are included for debugging.
    
    Like a teacher showing their work, the green agent should provide
    examples of individual predictions for inspection.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 25},  # More than the 20 sample limit
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    samples = data["samples"]
    
    # Should have samples but capped at 20
    assert len(samples) > 0
    assert len(samples) <= 20
    
    # Each sample should have required fields
    for sample in samples:
        assert "row_index" in sample
        assert "gold" in sample
        assert "pred" in sample
        assert "purple_response_preview" in sample
        
        # Preview should be truncated
        assert len(sample["purple_response_preview"]) <= 200


@pytest.mark.asyncio
async def test_real_dataset_max_rows_enforcement(agent, mock_purple):
    """
    Test that max_rows config is properly enforced.
    
    Like a bouncer at a club with a capacity limit - we need to ensure
    the green agent stops processing at the specified limit.
    """
    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 7},
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Should process exactly 7 rows
    assert data["metrics"]["rows_evaluated"] == 7
    
    # Should make exactly 7 calls
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 7


@pytest.mark.asyncio
async def test_real_dataset_accuracy_range(agent, mock_purple):
    """
    Test that accuracy is calculated and falls within valid range.
    
    Accuracy is like a batting average - it should be between 0 and 1,
    representing the percentage of correct predictions.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 15},
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    metrics = data["metrics"]
    
    # Accuracy must be in valid range
    assert 0.0 <= metrics["accuracy"] <= 1.0
    
    # Correct count should not exceed total
    assert metrics["correct"] <= metrics["rows_evaluated"]
    
    # Accuracy should match calculation
    expected_accuracy = metrics["correct"] / metrics["rows_evaluated"]
    assert abs(metrics["accuracy"] - expected_accuracy) < 0.0001


@pytest.mark.asyncio
async def test_real_dataset_no_duplicate_processing(agent, mock_purple):
    """
    Test that each row is processed exactly once.
    
    Like ensuring each person votes only once in an election,
    we need to verify no row gets double-counted.
    """
    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    req = {
        "participants": {"agent": mock_purple},
        "config": {"max_rows": 12},
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Rows evaluated should equal purple agent calls
    rows_evaluated = data["metrics"]["rows_evaluated"]
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    
    assert rows_evaluated == calls == 12


@pytest.mark.asyncio
async def test_real_dataset_config_tracking(agent, mock_purple):
    """
    Test that the config used is properly tracked in results.
    
    Like a scientific paper documenting experimental parameters,
    the results should include what configuration was used.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 3,
            "ruleset_column": "ruleset",
            "gold_column": "trad_result",
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    config_used = data["config_used"]
    
    # Verify all config params are tracked
    assert "dataset_path" in config_used
    assert "ruleset_column" in config_used
    assert "gold_column" in config_used
    assert "max_rows" in config_used
    assert "allowed_labels" in config_used
    
    # Verify allowed labels are complete
    assert set(config_used["allowed_labels"]) == {"SAC", "SCC", "STC", "WAC", "WCC", "WTC"}


@pytest.mark.asyncio 
async def test_real_dataset_full_default_run(agent, mock_purple):
    """
    Test a full run with default configuration.
    
    This is the "smoke test" - does everything work end-to-end
    with minimal configuration?
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {},  # Use all defaults
    }

    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Should process default max_rows (50)
    assert data["metrics"]["rows_evaluated"] == 50
    
    # Should have made 50 calls
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 50

    # Should have samples
    assert len(data["samples"]) == 20  # Capped at 20

    # Should have label statistics
    assert len(data["label_stats"]["pred_counts"]) > 0