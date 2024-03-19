
NODE_RANK=0
NUM_GPUS=0
export LOCAL_RANK=-1
task_ratio=mlm.5.sap.5.masksem.1.pert.1
outdir=snap_pt/r2r/${task_ratio}.neigh

# train
python pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config configs/r2r_model.json \
    --config configs/r2r_pretrain.json \
    --output_dir $outdir --task_ratio ${task_ratio}
