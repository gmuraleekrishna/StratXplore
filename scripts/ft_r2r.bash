DATA_ROOT=datasets

train_alg=dagger

features=vitclip
ft_dim=512
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

pref=mlm.5.sap.5.masksem.1.pert.1.neigh
iter=80000    # the selected iteration in pretraining
name=${pref}.${iter}.pc_lambda0.1.pc_teacher

outdir=snap_ft/r2r/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert 
          
      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 80

      --batch_size 8
      --lr 1e-5
      --iters 40000
      --log_every 500
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.3
      --pc_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      --pc_order 1"

export PYTHONPATH=$(pwd)/map_nav_src:$PYTHONPATH

# train -- uncomment for training
 python map_nav_src/r2r/main_nav.py $flag \
        --enable_path_correction \
        --val_unseen \
        --expl_sample \
        --know_root_path './knowledge_features' \
        --bert_ckpt_file snap_pt/r2r/${pref}/ckpts/model_step_${iter}.pt
#        --resume_file snap_ft/r2r/${name}/ckpts/latest_dict
#        --resume_file /mnt/Storage2/Krishna/datasets/bevbert/r2r_best
#         --use_weighted_candidate \
#         --eval_first \
# test
#python map_nav_src/r2r/main_nav.py $flag  \
#       --resume_file snap_ft/r2r/${name}/ckpts/latest_dict \
#       --know_root_path './knowledge_features' \
#      --expl_sample --submit --detailed_output --test --val_unseen \
#      --enable_path_correction \
#      --feedback pc_argmax
