> energy.csv
echo "model, power, energy" >> energy.csv
python train_prune_energy.py --backbone mv --neck gdf --phi S2 --pa erk --pm 0.35
sleep 60
python train_prune_energy.py --backbone mv --neck cdf --phi S2 --pa erk --pm 0.48
sleep 60
python train_prune_energy.py --backbone pf --neck cdf --phi S2 --pa erk --pm 0.62
sleep 60
python train_prune_energy.py --backbone pf --neck cdf --phi S2 --pa erk --pm 0.55
sleep 60
python train_prune_energy.py --backbone pf --neck gdf --phi S2 --pa erk --pm 0.52
sleep 60
python train_prune_energy.py --backbone pf --neck gdf --phi S2 --pa erk --pm 0.45
sleep 60
python train_prune_energy.py --backbone ef --neck gdf --phi S2 --pa erk --pm 0.36
sleep 60
python train_prune_energy.py --backbone ef --neck cdf --phi S2 --pa erk --pm 0.50
sleep 60
python train_prune_energy.py --backbone mv --neck gdf --phi S2 --pa mmu --pm 0.35
sleep 60
python train_prune_energy.py --backbone ef --neck gdf --phi S2 --pa mmu --pm 0.36
# python train_prune_energy.py --backbone pf --neck gdf --phi S2 --pa mmu --pm 0.55
