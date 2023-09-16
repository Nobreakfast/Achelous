#CUDA_VISIBLE_DEVICES="1,2,3,4" torchrun \
#    --nproc_per_node=4 --master_port=65535 \
#    train.py --backbone mv --neck gdf --nd False --phi $1 --lr_init 0.01 \
#    --data_root /home/zhaohaocheng/Data/WaterScenes_new --ddp True --bs 32

python train.py --backbone mv --neck gdf --nd False --phi $1 --lr_init 0.01 \
--data_root /home/zhaohaocheng/Data/WaterScenes_new --fp16 True --bs 32

# python train_prune.py --backbone mv --neck gdf --nd False --phi S2 --lr_init 0.01 --data_root /home/zhaohaocheng/Data/WaterScenes_new --bs 32 --pth logs/best_epoch_weights.pth --pm 0.3 --log_dir plogs/01