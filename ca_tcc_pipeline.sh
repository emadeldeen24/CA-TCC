exp="HAR_experiment"
run="HAR"
dataset="HAR"

start=0
end=0

for i in $(eval echo {$start..$end})
do
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "self_supervised"
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "train_linear_1p"
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "ft_1p"
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "gen_psuedo_labels"
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "SupCon"
    python3 main.py --experiment_description $exp --run_description $run --seed $i --selected_dataset $dataset --training_mode "train_linear_SupCon_1p"
done