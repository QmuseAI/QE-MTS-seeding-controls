# Sourced by submit_all.sh / resubmit_run.sh: node selection (same logic as QEC_SB_paper1/code/slurm/common.sh)
#   EXCLUDE   nodes not to use, comma separated (default: none -> every node is used; e.g. EXCLUDE=n013)
#   N         number of array jobs = nodes used at the same time. Default: every usable node of the partition
#             except EXCLUDE (idle/mixed/allocated/completing; down/drain are skipped)
#   MAXN      upper limit for N (default 32)
#   WORKERS   worker processes per node (default 500 of the 512 logical CPUs)
#   EXTRA     other sbatch options, e.g. EXTRA="--partition=xxx --account=yyy"
EXCLUDE=${EXCLUDE-}
EXTRA=${EXTRA:-}
[ -n "$EXCLUDE" ] && EXTRA="$EXTRA --exclude=$EXCLUDE"
CPUS=${CPUS:-512}
WORKERS=${WORKERS:-500}
if [ -z "${N:-}" ]; then
    _part=$(echo "$EXTRA" | { grep -o -- '--partition=[^ ]*' || true; } | head -1 | cut -d= -f2)
    _nodes=""
    if command -v sinfo >/dev/null 2>&1; then
        _nodes=$(sinfo -h -N -t idle,mixed,allocated,completing ${_part:+-p "$_part"} -o "%N" 2>/dev/null | sort -u || true)
    fi
    if [ -n "$EXCLUDE" ] && [ -n "$_nodes" ]; then
        _nodes=$(echo "$_nodes" | { grep -v -x -F -f <(echo "$EXCLUDE" | tr ',' '\n') || true; })
    fi
    N=$(echo "$_nodes" | sed '/^$/d' | wc -l)
    [ "$N" -lt 1 ] && N=4
    [ "$N" -gt "${MAXN:-32}" ] && N=${MAXN:-32}
fi
echo "nodes: N=$N  cpus/task=$CPUS  workers/node=$WORKERS  exclude=${EXCLUDE:-none}  extra='$EXTRA'"
