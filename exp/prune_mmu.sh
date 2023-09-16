case $1 in
"ef")
  python train_prune.py --backbone ef --neck gdf --nd False --phi S2 --lr_init 0.01 --lr_decay cos --data_root /home/zhaohaocheng/Data/WaterScenes_new --bs 32 --log_dir plogs/v10-mmu/ef-gdf-S2-0.36 --nw $2 --pm 0.36 --pa mmu
  ;;
"mv")
  python train_prune.py --backbone mv --neck gdf --nd False --phi S2 --lr_init 0.01 --lr_decay cos --data_root /home/zhaohaocheng/Data/WaterScenes_new --bs 32 --log_dir plogs/v10-mmu/mv-gdf-S2-0.35 --nw $2 --pm 0.35 --pa mmu
  ;;
"pf")
  python train_prune.py --backbone pf --neck gdf --nd False --phi S2 --lr_init 0.01 --lr_decay cos --data_root /home/zhaohaocheng/Data/WaterScenes_new --bs 32 --log_dir plogs/v10-mmu/pf-gdf-S2-0.55 --nw $2 --pm 0.55 --pa mmu
  ;;
esac
