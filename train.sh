CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch\
    --master_port 8888\
    --nproc_per_node=3\
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -num_gpus 3\
    -run_type train\
    -batch_size_per_gpu 2\
    -batch_size_per_gpu_eval 32\
    -model_dir ckpt/9.28_10epoch_b2_s42_lr1e-4_remove_residual\
    -epochs 10\
    -seed 42\
    -learning_rate 1e-4\
    -s_type en\
    -t_type en
