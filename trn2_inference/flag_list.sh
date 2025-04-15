NEURON_DUMP_PATH=$1
HLO_DUMP_PATH=$2

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives --enable-ccop-compute-overlap --cc-pipeline-tiling-factor=32'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --partitioner-max-disj-runs=4'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-enable-dge-levels spill_reload --internal-backend-options=' --spill-reload-dmas-use-swdge '"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none"

if [ $ENABLE_NEURON_DUMP = 1 ]; then
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
fi

export XLA_FLAGS="--xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives"

if [ $ENABLE_HLO_DUMP = 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
fi

# Neuron runtime flags
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
export NEURON_RT_DBG_CC_INTER_MESH_MAX_NODES=0
#export NEURON_RT_DBG_MESH_CC=0  # commented-out as mesh is required for all2all

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1

if [ $ENABLE_JAX_COMPILATION_CACHE = 1 ]; then
    export JAX_COMPILATION_CACHE_DIR="$HOME/axlearn/cc_cache/"
    mkdir -p $JAX_COMPILATION_CACHE_DIR
fi

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
# export NEURON_FSDP=1  # commented-out as we don't want FSDP for inference
# export NEURON_FSDP_NUM_LAYER_COALESCE=1
# export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
# export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
