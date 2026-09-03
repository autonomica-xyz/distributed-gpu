from __future__ import annotations

import pytest

from gpu_orchestrator.cli import main
from gpu_orchestrator.store import Store


def test_revoke_token_cli_disables_credential(tmp_path, capsys):
    db = tmp_path / "control.sqlite3"
    store = Store(db)
    raw = store.issue_token(
        role="agent",
        subject="worker-a",
        token_id="agent-worker-a",
    )
    store.close()

    assert main([
        "revoke-token", "--db", str(db), "--token-id", "agent-worker-a"
    ]) == 0
    assert "revoked token agent-worker-a" in capsys.readouterr().out

    store = Store(db)
    with pytest.raises(PermissionError):
        store.authenticate(raw)
    store.close()

    assert main([
        "revoke-token", "--db", str(db), "--token-id", "agent-worker-a"
    ]) == 1
    assert "not found or already disabled" in capsys.readouterr().err
