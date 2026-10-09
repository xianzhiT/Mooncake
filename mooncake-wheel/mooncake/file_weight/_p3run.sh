#!/bin/bash
# P3 real-scale e2e runner — inside mooncake-fw:dev. /src=repo, /fw=host /root/fw.
set -e
P=/usr/bin/python3.10
export PYTHONPATH=$($P -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))")
LOG=/fw/wt3
rm -rf "$LOG"; mkdir -p "$LOG/offload"
pkill -f mooncake_master 2>/dev/null || true; sleep 1
# Larger weight_migration byte budget per round so a 64 MiB-chunk shard
# migrates in one reconcile cycle; offload cap generous for real-scale payloads.
/usr/local/bin/mooncake_master \
  --rpc_address=127.0.0.1 --rpc_port=50181 --default_kv_lease_ttl=1000 \
  --enable_offload=true --offload_on_evict=true --promotion_on_hit=true \
  --weight_migration_max_bytes_per_round=4294967296 \
  --root_fs_dir="" --enable_http_metadata_server=false --logtostderr=true \
  >"$LOG/master.log" 2>&1 &
MP=$!; sleep 3
export MOONCAKE_PROTOCOL=tcp MOONCAKE_MASTER=127.0.0.1:50181
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE MOONCAKE_LOCAL_HOSTNAME=localhost:17815
export MC_FORCE_TCP=1 MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$LOG/offload"
export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1
export MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=8
export MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=134217728
timeout 300 "$P" /src/mooncake-wheel/mooncake/file_weight/_p3_e2e.py \
  >"$LOG/p3.log" 2>&1
echo "RC=$?" >> "$LOG/p3.log"
kill "$MP" 2>/dev/null || true
