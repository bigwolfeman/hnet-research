# Shared helpers for experiment queues (source it; set D=<queue dir> and SMOKE_GROUP=<wandb group> first).
#   log <msg>               append a timestamped line to $D/chain.log
#   wait_gpu                block while any train.py experiment runs (one model per GPU)
#   suite_gate              HNET_TEST_CUDA=1 pytest; exits the queue on failure
#   smoke <name> <exp> ...  200 steps with one eval and the coarse probe, wandb project hnet-smoke; 0 on success
#   shuffle_s0 <run_dir>    probe/coarse_shuffle_s{0} from <run_dir>/summary.json (stage index: 2nd arg, default 0)
# Call from the spike root (the queue scripts cd there).
PY=${PY:-.venv/bin/python}
log() { echo "$(date -Is) $*" >> "$D/chain.log"; }
wait_gpu() { while pgrep -f "[t]rain.py experiment=" > /dev/null; do sleep 30; done; }
suite_gate() {
    log "suite start"
    if ! HNET_TEST_CUDA=1 $PY -m pytest -q > "$D/suite.log" 2>&1; then
        log "SUITE FAILED, nothing started (see suite.log)"; exit 1
    fi
    log "suite ok: $(tail -1 "$D/suite.log")"
}
smoke() {
    local name=$1 exp=$2; shift 2
    mkdir -p "$D/smoke"
    if OMP_NUM_THREADS=2 $PY train.py experiment="$exp" train=ladder seed=0 hydra.run.dir="$D/smoke/$name" \
        train.max_steps=200 train.eval_every=200 train.resume_every=0 "train.ladder_fracs=[]" \
        wandb.project=hnet-smoke wandb.group="$SMOKE_GROUP" wandb.name="$name" "$@" > "$D/smoke/$name.log" 2>&1; then
        log "smoke ok $name: $(grep -o '"final_val_bpb": [0-9.]*' "$D/smoke/$name.log" | tail -1) $(grep -m1 '^params' "$D/smoke/$name.log" | cut -c1-40)"
        return 0
    fi
    log "SMOKE FAILED $name (see smoke/$name.log)"; return 1
}
shuffle_s0() {
    $PY -c "import json,sys; print(json.load(open(sys.argv[1] + '/summary.json'))['probes']['probe/coarse_shuffle_s' + sys.argv[2]])" "$1" "${2:-0}"
}
