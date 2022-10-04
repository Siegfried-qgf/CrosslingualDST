CUDA_VISIBLE_DEVICES=0 python main.py\
    -s_type en\
    -t_type de\
    -ckpt ./ckpt/9.28_10epoch_b2_s42_lr1e-4_remove_residual/ckpt-epoch4\
    -run_type predict\
    -batch_size_per_gpu_eval 32\
    -output de_zeroshot







