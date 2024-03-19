
NODE_RANK=0
NUM_GPUS=1
export LOCAL_RANK=-1
task_ratio=mlm.1.mrc.1.sap.1.og.1.pert.1
outdir=snap_pt/rvr/${task_ratio}
# pert 34481
# train
python pretrain_src/train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config configs/rvr_model.json \
    --config configs/rvr_pretrain.json \
    --output_dir $outdir --task_ratio ${task_ratio} \
    --checkpoint ${outdir}/ckpts/model_step_22000.pt \
    --start_at 22000
