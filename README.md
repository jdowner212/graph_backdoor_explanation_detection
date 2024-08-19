# graph_backdoor_explanation_detection
Accompanies our paper, "Identifying Backdoor Training Samples in Graph Data: A GNN Explainer Approach with Novel Metrics".


# SETUP

## 1. Setup environment

	a. Navigate to desired repository location
	b. Run the following from the command line:
		conda create --name graph_backdoor_detection python=3.8.16
		git clone https://github.com/jdowner212/graph_backdoor_explanation_detection.git
		cd graph_backdoor_explanation_detection
		pip install -r setup/requirements.txt

## 3. Set repository path for code to reference

	- Run the following from the command line:
		repository_path="$(pwd)"
		if [[ "$OSTYPE" == "darwin"* ]]; then
			sed -i '' "s|root_dir = .*|root_dir = '$repository_path'|" "$repository_path/src/utils/config.py"
		else
			sed -i "s|root_dir = .*|root_dir = '$repository_path'|" "$repository_path/src/utils/config.py"
		fi

## 2. Make necessary modifications to installed torch_geometric package

	- Run the following from the command line: 
		repository_path="$(pwd)"
		torch_geometric_location=$(pip show torch-geometric | grep Location | awk '{print $2}' | sed 's|$|/torch_geometric|')
		if [[ "$OSTYPE" == "darwin"* ]]; then
			sed -i '' "s|dir_to_modify = .*|dir_to_modify = '$torch_geometric_location'|" $repository_path/setup/modify_torch_geometric_script.py
		else
			sed -i "s|dir_to_modify = .*|dir_to_modify = '$torch_geometric_location'|" $repository_path/setup/modify_torch_geometric_script.py
		fi
		setup_path=$(echo $(pwd)/setup/modified_torch_geometric_files)
		if [[ "$OSTYPE" == "darwin"* ]]; then
			sed -i '' "s|modified_files_dir = .*|modified_files_dir = '$setup_path'|" $repository_path/setup/modify_torch_geometric_script.py
		else
			sed -i "s|modified_files_dir = .*|modified_files_dir = '$setup_path'|" $repository_path/setup/modify_torch_geometric_script.py
		fi
		python setup/modify_torch_geometric_script.py
	



# RUN

## Choose attack type, dataset, etc. (See /src/run.py for possible input arguments.)

	- Main choices:

		--backdoor_type (options: random, adaptive, clean_label)
		----> if random, can also choose graph_synthesis method (options: ER (Erdos-Renyi), SW (Small World), PA (Preferential Attachment)) 
		--trigger_size (options: any integer > 0)
		--attack_target_label (options: 0, 1)
		--poison_rate (options: any float between 0 and 1)
		--dataset (options: MUTAG, AIDS, IMDB-BINARY, PROTEINS)
		--model_hyp_set (Choice of pre-defined hyperparameter sets for GNN training. Typical options are A, B, or C, but may depend on dataset choice. See /src/utils/config.py for all options -- feel free to change or add your own.)
   
	- Note: not all attack configurations will succeed. Experiment to find settings that work.

## Examples:

	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type random --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type adaptive --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4
	python run.py --run_attack --run_explain --attack_target_label 1 --backdoor_type clean_label --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
