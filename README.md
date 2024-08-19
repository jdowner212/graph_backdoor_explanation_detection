# graph_backdoor_explanation_detection
Accompanies our paper, "Identifying Backdoor Training Samples in Graph Data: A GNN Explainer Approach with Novel Metrics".


# SETUP

## 1. Setup environment

	a. Navigate to /repo
	b. Run the following from the command line:
		conda create --name graph_backdoor_detection python=3.8.16
		pip install -r requirements.txt

## 2. Make necessary modifications to installed torch_geometric package

	a. Run this command to get path to installed torch_geometric package: 
		pip show torch-geometric | grep Location | awk '{print $2}' | sed 's|$|/torch_geometric|'

	b. Navigate to /repo_pub/setup and open modify_torch_geometric_script.py. 
		i. Change first path to above terminal output
		ii. Change second path to /path/to/repo_pub/setup/modified_torch_geometric_files
		iii. Save changes
	
	c. From command line, run 
		-python setup/modify_torch_geometric_script.py
	

3. Open utils/config.py and change root_dir to path/to/repo


# RUN ATTACK/EXPLAIN/DETECT

## Choose attack type, dataset, etc. (See repo/src/run.py for possible input arguments.)

	- Main choices:

		--backdoor_type (options: random, adaptive, clean_label)
		----> if random, can also choose graph_synthesis method (options: ER (Erdos-Renyi), SW (Small World), PA (Preferential Attachment)) 
		--trigger_size (options: any integer > 0)
		--attack_target_label (options: 0, 1)
		--poison_rate (options: any float between 0 and 1)
		--dataset (options: MUTAG, AIDS, IMDB-BINARY, PROTEINS)
		--model_hyp_set (Choice of pre-defined hyperparameter sets for GNN training. Typical options are A, B, or C, but may depend on dataset choice. See /repo/src/utils/config.py for all options -- feel free to change or add your own.)
   
	- Note: not all attack configurations will succeed. Experiment to find settings that work.

## Examples:

	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type random --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type adaptive --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4
	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type clean_label --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
