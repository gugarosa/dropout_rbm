# Variables definition
DATA="mnist"
N_VISIBLE=784
N_HIDDEN=400
DEVICE="cuda" # "cpu" or "cuda"
MH="pso"
N_RUNS=1

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Optimizing the dropout parameter
    python optimization.py ${DATA} ${MH} -n_visible ${N_VISIBLE} -n_hidden ${N_HIDDEN} -device ${DEVICE} -seed ${SEED}
done
