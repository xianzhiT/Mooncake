"""Command-line management of weights in an existing Mooncake Store cluster."""

import argparse
import json
import os
import sys
from pathlib import Path

from .model_keyspace import validate_checkpoint_id
from .weight_cache import DEFAULT_MANAGEMENT_LOCK_DIR, WeightCacheClient


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="JSON connection configuration (or MOONCAKE_WEIGHT_CONFIG)",
    )
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit JSON results/errors and suppress progress",
    )
    parser = argparse.ArgumentParser(
        prog="mooncake-weight",
        parents=[common],
        description="Manage safetensors weights in an existing Mooncake Store cluster.",
        epilog="Import/delete require one management host and a shared lock directory.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "import": "Import a local model directory under a new checkpoint ID.",
        "list": "List checkpoint IDs; presence does not imply completeness.",
        "inspect": "Check expected key existence; does not validate chunk lengths or hashes.",
        "verify": "Read weights and check chunk lengths; does not compare content hashes.",
        "delete": "Delete checkpoint objects immediately; may interrupt readers.",
    }
    for command, description in descriptions.items():
        sub = commands.add_parser(
            command,
            parents=[common],
            help=description,
            description=description,
        )
        if command != "list":
            sub.add_argument("checkpoint_id")
        if command == "import":
            sub.add_argument("source", help="local HuggingFace safetensors directory")
    return parser


def _load_config(path):
    if not path:
        raise ValueError("set --config FILE or MOONCAKE_WEIGHT_CONFIG")
    supplied = json.loads(Path(path).read_text())
    if not isinstance(supplied, dict):
        raise ValueError("configuration must be a JSON object")
    config = {
        "master_server_address": "",
        "local_hostname": "127.0.0.1:0",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "tcp",
        "device_name": "",
        "local_buffer_size": 128 * 1024**2,
        "replica_num": 1,
        "management_lock_dir": DEFAULT_MANAGEMENT_LOCK_DIR,
    }
    unknown = supplied.keys() - config.keys()
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    config.update(supplied)
    for key, value in config.items():
        if key in ("local_buffer_size", "replica_num"):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        elif not isinstance(value, str) or (key != "device_name" and not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
    if config["protocol"] not in ("tcp", "rdma"):
        raise ValueError("protocol must be tcp or rdma")
    return config


def _new_store():
    # Delay the native dependency until after argument/configuration validation.
    from mooncake.store import MooncakeDistributedStore

    return MooncakeDistributedStore()


def _run(args, config):
    store = _new_store()
    try:
        result = store.setup(
            config["local_hostname"],
            config["metadata_server"],
            0,  # Never place weights in the short-lived CLI process.
            config["local_buffer_size"],
            config["protocol"],
            config["device_name"],
            config["master_server_address"],
        )
        if result != 0:
            raise RuntimeError(f"Store setup failed: {result}")
        client = WeightCacheClient(
            store,
            replica_num=config["replica_num"],
            progress=not args.json,
            management_lock_dir=config["management_lock_dir"],
        )
        if args.command == "list":
            return client.list_models(), 0
        if args.command == "delete":
            removed = client.delete_model(args.checkpoint_id)
            return {"checkpoint_id": args.checkpoint_id, "removed_objects": removed}, 0
        if args.command == "import":
            status = client.import_model(args.checkpoint_id, args.source)
        elif args.command == "inspect":
            status = client.inspect_model(args.checkpoint_id)
        else:
            status = client.verify_model(args.checkpoint_id)
        return status.to_dict(), 0 if status.complete else 1
    finally:
        store.close()


def _print_result(command, result, as_json):
    if as_json:
        print(json.dumps(result))
    elif command == "list":
        print("\n".join(result) if result else "No checkpoints found.")
    elif command == "delete":
        print(f"{result['checkpoint_id']}: removed {result['removed_objects']} objects")
    else:
        state = "complete" if result["complete"] else "incomplete"
        print(f"{result['checkpoint_id']}: {state}")
        print(
            f"Observed layout: {len(result['files'])} files, {result['total_size']} bytes"
        )
        for item in result["files"]:
            print(
                f"  {item['path']}: {item['size']} bytes, {item['chunk_count']} object(s)"
            )
        for key in result["missing_keys"]:
            print(f"  Missing: {key}")


def main(argv=None):
    args = _parser().parse_args(argv)
    args.json = getattr(args, "json", False)
    try:
        if args.command != "list":
            validate_checkpoint_id(args.checkpoint_id)
        config = _load_config(
            getattr(args, "config", None) or os.environ.get("MOONCAKE_WEIGHT_CONFIG")
        )
        result, code = _run(args, config)
        _print_result(args.command, result, args.json)
        return code
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        error = {"error": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "missing_keys"):
            error["missing_keys"] = exc.missing_keys
        print(json.dumps(error) if args.json else f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; a partial import may require explicit deletion.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
