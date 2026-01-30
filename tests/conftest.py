import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )
    parser.addoption(
        "--no-auto-start",
        action="store_true",
        default=False,
        help="Disable automatic server startup (expect server already running)",
    )


def _is_server_running(url: str) -> bool:
    """Check if a server is already running at the given URL."""
    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _wait_for_server(url: str, timeout_s: float = 30.0) -> bool:
    """Wait for server to be ready by polling agent card endpoint."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def agent(request):
    """
    Agent URL fixture. 
    
    Behavior:
    - If server is already running at --agent-url: use it
    - If --no-auto-start: expect server running (fail if not)
    - Otherwise: automatically start server, then shut it down after tests
    
    Usage examples:
        # Auto-start mode (default)
        pytest tests/
        
        # Use existing server (legacy mode)
        pytest --no-auto-start tests/
        
        # Use server at different URL
        pytest --agent-url http://localhost:8080 tests/
    """
    url = request.config.getoption("--agent-url")
    no_auto_start = request.config.getoption("--no-auto-start")
    
    # Check if server is already running
    if _is_server_running(url):
        print(f"\n✓ Using existing server at {url}")
        yield url
        return
    
    # If no auto-start, fail immediately
    if no_auto_start:
        pytest.exit(
            f"Server not running at {url} and --no-auto-start specified",
            returncode=1
        )
    
    # Auto-start mode: Start the server
    print(f"\nStarting green agent server at {url}...")
    
    # Find repository root and server script
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent  # Go up from tests/ to repo root
    
    src_dir = repo_root / "src"
    server_script = src_dir / "server.py"
    
    if not server_script.exists():
        pytest.exit(
            f"Cannot find server.py at {server_script}. "
            f"Make sure you're running from the repository root.",
            returncode=1
        )
    
    # Parse URL to get host and port
    # Simple parsing for http://host:port format
    url_parts = url.replace("http://", "").replace("https://", "")
    if ":" in url_parts:
        host, port_str = url_parts.split(":", 1)
        port = int(port_str.rstrip("/"))
    else:
        host = url_parts.rstrip("/")
        port = 9009
    
    # Start the server process
    proc = subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            "--host", host,
            "--port", str(port),
            "--card-url", f"{url}/",
        ],
        cwd=str(repo_root),  # Run from repo root so paths resolve correctly
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Wait for server to be ready
        if not _wait_for_server(url, timeout_s=30.0):
            # Server didn't start - get the output
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
            pytest.exit(
                f"Server failed to start at {url} within 30 seconds.\n"
                f"STDOUT: {stdout}\n"
                f"STDERR: {stderr}",
                returncode=1
            )
        
        print(f"✓ Server ready at {url}")
        
        # Yield control to tests
        yield url
        
    finally:
        # Cleanup: shut down the server
        print(f"\n🛑 Shutting down green agent server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()