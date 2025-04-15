#!/usr/bin/env bash

# Manually set POD_UID and MPI-related env vars for interactive testing without MPI Operator 
export POD_UID=`tr -cd '[:alpha:]' < /dev/urandom | head -c15`
export PMIX_HOSTNAME="worker0"
export OMPI_COMM_WORLD_SIZE=1
export OMPI_COMM_WORLD_RANK=0


TEST_ARTIFACTS_PATH="/shared/axlearn_artifacts/$POD_UID/"
mkdir -p "$TEST_ARTIFACTS_PATH"
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump/$OMPI_COMM_WORLD_RANK
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump/$OMPI_COMM_WORLD_RANK

source ./flag_list.sh $NEURON_DUMP_PATH $HLO_DUMP_PATH

# Manually specify CCOM interface, often required on EKS
export CCOM_SOCKET_IFNAME=eth0

# Neuron env vars for distributed training
nodes=`/neuron/scripts/nodelist_helper.py`
# devices_per_node=$((128/$NEURON_RT_VIRTUAL_CORE_SIZE))
devices_per_node=64
export NEURON_RT_NUM_CORES=64
#export COORDINATOR_ADDRESS=$(echo "$nodes" | head -n 1):64272
export NEURON_RT_ROOT_COMM_ID=localhost:33333
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $OMPI_COMM_WORLD_SIZE | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$OMPI_COMM_WORLD_RANK
unset OMPI_MCA_orte_hnp_uri

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

# Use tcmalloc - this is required
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
if [ -n "$LIBTCMALLOC" ]; then
    # Create a symbolic link to the found libtcmalloc version
    ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
    echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
    # Export LD_PRELOAD
    export LD_PRELOAD=/usr/lib/libtcmalloc.so
    echo "LD_PRELOAD set to: $LD_PRELOAD"
else
    echo "Error: libtcmalloc.so not found"
    exit 1
fi

# show env vars in logs
set

python3 -m axlearn.cloud.gcp.examples.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-70B-v2-flash \
        --trainer_dir="/path/to/70B/chkpt/step_00000100" \
        --mesh_selector="neuron-trn2.48xlarge-64" 2>&1 | tee ${OUTPUT_DIR}/${PMIX_HOSTNAME}.log

# Dummy checkpoints created using short TP4+FSDP training jobs:
#  70B-v2-flash 4L ckpt: /shared/axlearn_artifacts/lZddxhytMVopMZMMDBpRntqFe/axlearn_out/checkpoints/step_00000400
#  70B-v1-flash 4L ckpt: /shared/axlearn_artifacts/dsEahySmboHQZAEfgEZnxYApN/axlearn_out/checkpoints/step_00000400
#  3B ckpt: /shared/axlearn_artifacts/SuMAHGmvpNkHffDDanUtHBOve/axlearn_out/checkpoints/step_00000300
