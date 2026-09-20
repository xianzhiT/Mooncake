"""CLI contracts using the real manager and a write-once Store stand-in."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from test_weight_cache import WriteOnceStore, write_model
from mooncake.weight_store import WeightCacheClient, model_file_chunk_key


def test_cli_is_available():
    assert importlib.util.find_spec("mooncake.weight_store.cli") is not None


@pytest.fixture
def cli_store(monkeypatch, tmp_path):
    from mooncake.weight_store import cli

    class Store(WriteOnceStore):
        def __init__(self):
            super().__init__()
            self.setup_args = None
            self.closed = 0

        def setup(self, *args):
            self.setup_args = args
            return 0

        def close(self):
            self.closed += 1

    store = Store()
    monkeypatch.setattr(cli, "_new_store", lambda: store)
    config = tmp_path / "connection.json"
    config.write_text(
        json.dumps(
            {
                "master_server_address": "127.0.0.1:50051",
                "management_lock_dir": str(tmp_path / "locks"),
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_WEIGHT_CONFIG", str(config))
    return cli, store, config


def test_management_roundtrip(cli_store, tmp_path, capsys):
    cli, store, _ = cli_store
    source = write_model(tmp_path / "source")
    assert cli.main(["import", "model-a", str(source), "--json"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["complete"] is True
    assert not output.err  # JSON mode suppresses Python progress messages.
    assert store.setup_args[2] == 0  # CLI must never host imported objects.
    assert store.closed == 1
    for command in ("inspect", "verify"):
        assert cli.main(["--json", command, "model-a"]) == 0
        assert json.loads(capsys.readouterr().out)["complete"] is True
    assert cli.main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == ["model-a"]
    assert cli.main(["delete", "model-a", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["removed_objects"] > 0
    assert not store.data
    assert store.closed == 5


def test_incomplete_inspect_and_verify(cli_store, tmp_path, capsys):
    cli, store, _ = cli_store
    source = write_model(tmp_path / "source", single_file=True)
    assert cli.main(["import", "broken", str(source)]) == 0
    capsys.readouterr()
    store.remove(model_file_chunk_key("broken", "model.safetensors", 0), True)
    assert cli.main(["inspect", "broken", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["complete"] is False
    assert cli.main(["verify", "broken", "--json"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"] == "IncompleteCheckpointError"


def test_busy_and_query_failure_are_errors(cli_store, capsys):
    cli, store, config = cli_store
    directory = json.loads(config.read_text())["management_lock_dir"]
    manager = WeightCacheClient(store, management_lock_dir=directory)
    with manager._management_lock("busy"):
        assert cli.main(["delete", "busy", "--json"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "CheckpointBusyError"

    def fail_query(pattern):
        raise RuntimeError("master unavailable")

    store.query_keys_by_regex = fail_query
    assert cli.main(["list", "--json"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "master unavailable" in json.loads(output.err)["message"]
    assert store.closed == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"global_segment_size": 1024},
        {"local_buffer_size": -1},
        {"replica_num": True},
        {"protocol": "cxl"},
        {"master_server_address": ""},
    ],
)
def test_invalid_configuration_never_connects(cli_store, patch, capsys):
    cli, store, config = cli_store
    values = json.loads(config.read_text())
    values.update(patch)
    config.write_text(json.dumps(values))
    assert cli.main(["list", "--json"]) == 1
    assert store.setup_args is None
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_setup_failure_closes_store(cli_store, capsys):
    cli, store, _ = cli_store
    store.setup = lambda *args: -1
    assert cli.main(["list"]) == 1
    assert "setup failed" in capsys.readouterr().err
    assert store.closed == 1


def test_explicit_config_overrides_environment(cli_store, tmp_path, capsys):
    cli, store, _ = cli_store
    alternate = tmp_path / "alternate.json"
    alternate.write_text('{"master_server_address": "another-host:50051"}')
    assert cli.main(["--config", str(alternate), "list"]) == 0
    assert store.setup_args[-1] == "another-host:50051"
    assert "No checkpoints" in capsys.readouterr().out


def test_help_needs_no_configuration_or_native_store(cli_store, capsys):
    cli, store, _ = cli_store
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["verify", "--help"])
    assert exit_info.value.code == 0
    assert "does not compare content hashes" in capsys.readouterr().out
    assert store.setup_args is None


@pytest.mark.parametrize("contents", ["[]", "{invalid", "{}"])
def test_bad_config_file(cli_store, contents, capsys):
    cli, store, config = cli_store
    config.write_text(contents)
    assert cli.main(["--json", "list"]) == 1
    assert not capsys.readouterr().out
    assert store.setup_args is None


@pytest.mark.skipif(
    not os.getenv("MOONCAKE_WEIGHT_TEST_MASTER"), reason="requires a test master"
)
def test_native_cli_data_survives_process_exit(tmp_path):
    from mooncake.store import MooncakeDistributedStore

    # This process provides long-lived storage; every CLI call is a new process.
    storage = MooncakeDistributedStore()
    master = os.environ["MOONCAKE_WEIGHT_TEST_MASTER"]
    try:
        assert (
            storage.setup(
                "127.0.0.1:0",
                "P2PHANDSHAKE",
                256 * 1024**2,
                128 * 1024**2,
                "tcp",
                "",
                master,
            )
            == 0
        )
        config = tmp_path / "native.json"
        config.write_text(
            json.dumps(
                {
                    "master_server_address": master,
                    "management_lock_dir": str(tmp_path / "locks"),
                }
            )
        )
        source = write_model(tmp_path / "source", single_file=True)
        checkpoint = "cli-" + tmp_path.name

        def run(*args):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mooncake.weight_store.cli",
                    "--config",
                    str(config),
                    "--json",
                    *args,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        try:
            assert run("import", checkpoint, str(source))["complete"]
            assert checkpoint in run("list")
            assert run("inspect", checkpoint)["complete"]
            assert run("verify", checkpoint)["complete"]
            manager = WeightCacheClient(storage, progress=False)
            assert (
                manager.read_file(checkpoint, "model.safetensors")
                == (source / "model.safetensors").read_bytes()
            )
        finally:
            assert run("delete", checkpoint)["removed_objects"] >= 0
        assert checkpoint not in run("list")
    finally:
        storage.close()
