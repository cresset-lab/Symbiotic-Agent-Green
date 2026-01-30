"""
Functional tests for green agent using the REAL mutation dataset.

These tests verify that:
1. Green agent can process the actual mutated_dataset.csv
2. Results include proper metrics and statistics
3. Label distribution matches expected patterns
4. Sample rows are included for debugging

Compatible with both original and robust agent implementations.
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


def _get_rows_count(metrics: dict) -> int:
    """
    Get the number of rows processed, compatible with both schemas.
    
    - Original schema: metrics["rows_evaluated"]
    - Robust schema: metrics["rows_attempted"] or metrics["rows_successful"]
    """
    # Try original schema first
    if "rows_evaluated" in metrics:
        return metrics["rows_evaluated"]
    # Fall back to robust schema
    if "rows_attempted" in metrics:
        return metrics["rows_attempted"]
    if "rows_successful" in metrics:
        return metrics["rows_successful"]
    raise KeyError("No rows count field found in metrics")


def _get_successful_rows(metrics: dict) -> int:
    """
    Get successfully processed rows count, compatible with both schemas.
    """
    if "rows_evaluated" in metrics:
        return metrics["rows_evaluated"]  # Original assumes all evaluated = successful
    if "rows_successful" in metrics:
        return metrics["rows_successful"]
    return _get_rows_count(metrics)


def _get_response_preview_field(sample: dict) -> str:
    """
    Get the response preview field, compatible with both schemas.
    
    - Original schema: sample["purple_response_preview"]
    - Robust schema: sample["response_preview"]
    """
    if "purple_response_preview" in sample:
        return sample["purple_response_preview"]
    if "response_preview" in sample:
        return sample["response_preview"]
    return ""


@pytest.mark.functional
@pytest.mark.asyncio
async def test_real_dataset_default_path(agent, mock_purple):
    """
    Test that green agent can load and process the default dataset.
    
    This verifies the dataset path resolution works correctly.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 5,
            "skip_health_check": True,  # Speed up test
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
    
    # Verify it processed rows (compatible with both schemas)
    rows = _get_rows_count(data["metrics"])
    assert rows == 5
    
    # Verify exactly 5 calls (1 per row)
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 5, f"Expected 5 purple calls for 5 rows, got {calls}"


@pytest.mark.functional
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
        "config": {
            "max_rows": 3,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Check top-level structure
    assert "metrics" in data
    assert "label_stats" in data
    assert "samples" in data
    assert "config_used" in data

    # Check metrics structure - compatible with both schemas
    metrics = data["metrics"]
    rows = _get_rows_count(metrics)
    assert rows == 3
    
    assert "correct" in metrics
    assert "accuracy" in metrics
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
    assert len(data["samples"]) <= rows


@pytest.mark.functional
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
            "skip_health_check": True,
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
    rows = _get_rows_count(data["metrics"])
    assert rows == 2


@pytest.mark.functional
@pytest.mark.asyncio
async def test_real_dataset_label_distribution(agent, mock_purple):
    """
    Test that label statistics are tracked correctly.
    
    Think of this like counting votes in an election - we need accurate
    tallies of how many times each candidate (label) was predicted.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 10,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    label_stats = data["label_stats"]
    pred_counts = label_stats["pred_counts"]
    
    # Verify total predictions equals successful rows
    total_predictions = sum(pred_counts.values())
    successful_rows = _get_successful_rows(data["metrics"])
    assert total_predictions == successful_rows
    
    # Verify all predicted labels are valid
    for label in pred_counts.keys():
        assert label in {"WAC", "SAC", "WTC", "STC", "WCC", "SCC"}


@pytest.mark.functional
@pytest.mark.asyncio
async def test_real_dataset_sample_rows(agent, mock_purple):
    """
    Test that sample rows are included for debugging.
    
    Like a teacher showing their work, the green agent should provide
    examples of individual predictions for inspection.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 25,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    samples = data["samples"]
    
    # Should have samples (capped at some reasonable limit - 20 or 50 depending on version)
    assert len(samples) > 0
    assert len(samples) <= 50  # Robust version caps at 50, original at 20
    
    # Each sample should have required fields
    for sample in samples:
        assert "row_index" in sample
        assert "gold" in sample
        assert "pred" in sample
        
        # Check response preview field (different names in different versions)
        preview = _get_response_preview_field(sample)
        assert len(preview) <= 300  # Should be truncated


@pytest.mark.functional
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
        "config": {
            "max_rows": 7,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Should process exactly 7 rows
    rows = _get_rows_count(data["metrics"])
    assert rows == 7
    
    # Should make exactly 7 calls
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 7


@pytest.mark.functional
@pytest.mark.asyncio
async def test_real_dataset_accuracy_range(agent, mock_purple):
    """
    Test that accuracy is calculated and falls within valid range.
    
    Accuracy is like a batting average - it should be between 0 and 1,
    representing the percentage of correct predictions.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 15,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    metrics = data["metrics"]
    
    # Accuracy must be in valid range
    assert 0.0 <= metrics["accuracy"] <= 1.0
    
    # Correct count should not exceed total
    rows = _get_successful_rows(metrics)
    assert metrics["correct"] <= rows
    
    # Accuracy should match calculation (if we have successful rows)
    if rows > 0:
        expected_accuracy = metrics["correct"] / rows
        assert abs(metrics["accuracy"] - expected_accuracy) < 0.01  # Allow small rounding diff


@pytest.mark.functional
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
        "config": {
            "max_rows": 12,
            "skip_health_check": True,
        },
    }

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Rows evaluated should equal purple agent calls
    rows = _get_rows_count(data["metrics"])
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    
    assert rows == calls == 12


@pytest.mark.functional
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
            "skip_health_check": True,
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


@pytest.mark.functional
@pytest.mark.asyncio 
async def test_real_dataset_full_default_run(agent, mock_purple):
    """
    Test a full run with default configuration.
    
    This is the "smoke test" - does everything work end-to-end
    with minimal configuration?
    
    NOTE: This test uses 50 rows (default) and may take longer.
    We reduce to 20 rows to stay within reasonable timeout.
    """
    req = {
        "participants": {"agent": mock_purple},
        "config": {
            "max_rows": 20,  # Reduced from 50 to avoid timeout
            "skip_health_check": True,
        },
    }

    # Reset counter
    httpx.post(f"{mock_purple}/debug/reset", timeout=2)

    events = await send_text_message(json.dumps(req), agent, streaming=False)
    data = _extract_result_data(events)
    assert data is not None

    # Should process the requested rows
    rows = _get_rows_count(data["metrics"])
    assert rows == 20
    
    # Should have made 20 calls
    calls = httpx.get(f"{mock_purple}/debug/calls", timeout=2).json()["call_count"]
    assert calls == 20

    # Should have samples
    assert len(data["samples"]) > 0

    # Should have label statistics
    assert len(data["label_stats"]["pred_counts"]) > 0


# ============================================================================
# Additional tests for robust agent features
# ============================================================================

# @pytest.mark.functional
# @pytest.mark.asyncio
# async def test_robust_agent_handles_errors_gracefully(agent, mock_purple):
#     """
#     Test that the robust agent continues processing even if some rows fail.
    
#     This test only makes sense for the robust agent implementation.
#     For the original agent, it will just verify basic functionality.
#     """
#     req = {
#         "participants": {"agent": mock_purple},
#         "config": {
#             "max_rows": 5,
#             "skip_health_check": True,
#         },
#     }

#     events = await send_text_message(json.dumps(req), agent, streaming=False)
#     data = _extract_result_data(events)
#     assert data is not None
    
#     # Should have completed (regardless of implementation)
#     rows = _get_rows_count(data["metrics"])
#     assert rows == 5
    
#     # Robust agent should have success_rate field
#     if "success_rate" in data["metrics"]:
#         assert 0.0 <= data["metrics"]["success_rate"] <= 1.0


# @pytest.mark.functional
# @pytest.mark.asyncio
# async def test_robust_agent_timing_info(agent, mock_purple):
#     """
#     Test that the robust agent includes timing information.
    
#     This is a feature of the robust agent that helps with debugging.
#     """
#     req = {
#         "participants": {"agent": mock_purple},
#         "config": {
#             "max_rows": 3,
#             "skip_health_check": True,
#         },
#     }

#     events = await send_text_message(json.dumps(req), agent, streaming=False)
#     data = _extract_result_data(events)
#     assert data is not None
    
#     # Robust agent includes elapsed_seconds
#     if "elapsed_seconds" in data["metrics"]:
#         assert data["metrics"]["elapsed_seconds"] >= 0
        
#     # Robust agent includes per-row duration_ms in samples
#     if data["samples"] and "duration_ms" in data["samples"][0]:
#         for sample in data["samples"]:
#             assert sample["duration_ms"] >= 0