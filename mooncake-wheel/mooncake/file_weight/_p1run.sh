#!/bin/bash
# P1 e2e runner — inside mooncake-fw:dev. /src=repo, /fw=host /root/fw (writable).
set -e
P=/usr/bin/python3.10
export PYTHONPATH=$($P -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))")
LOG=/fw/wt
rm -rf "$LOG"; mkdir -p "$LOG/offload"
pkill -f mooncake_master 2>/dev/null || true; sleep 1
/usr/local/bin/mooncake_master \
  --rpc_address=127.0.0.1 --rpc_port=50181 --default_kv_lease_ttl=1000 \
  --enable_offload=true --offload_on_evict=true --promotion_on_hit=false \
  --root_fs_dir="" --enable_http_metadata_server=false --logtostderr=true \
  >"$LOG/master.log" 2>&1 &
MP=$!; sleep 3
export MOONCAKE_PROTOCOL=tcp MOONCAKE_MASTER=127.0.0.1:50181
export MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE MOONCAKE_LOCAL_HOSTNAME=localhost:17815
export MC_FORCE_TCP=1 MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$LOG/offload"
timeout 90 "$P" /src/mooncake-wheel/mooncake/file_weight/_p1_e2e.py \
  >"$LOG/p1.log" 2>&1
echo "RC=$?" >> "$LOG/p1.log"
kill "$MP" 2>/dev/null || true
