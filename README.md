The code appendix accompanying our paper, "Identifying Backdoor Training Samples in Graph Data: A GNN Explainer Approach with Novel Metrics".
This file contains two sections: (I) Repository Structure, (II) Description of Relevant Code, and (II) Instructions for use.

# I. REPOSITORY STRUCTURE

graph_backdoor_explanation_detection/
├── README.md
├── setup/
│   ├── modify_torch_geometric_script.py
│   └── requirements.txt
├── src/
│   ├── config.py
│   ├── main.py
│   └── attack/
│       ├── backdoor_utils.py
│       └── run_attack.py
│       └── run_build_adaptive_generator.py
│   └── detect/
│       ├── metrics.py
│       └── run_explain_and_detect.py
│   └── explain/
│       └── explainer_utils.py
│   └── utils/
│       ├── config.py
│       └── data_utils.py
│       └── general_utils.py
│       └── models.py
│       └── plot_utils.py
├── data/
│       ├── clean/
│       └── poisoned/
├── explainer_results/
├── generators_adaptive_trigger/
├── surrogate_models/
└── training_results/

Empty folders will be populated running by experiments below.


# II. DESCRIPTION OF RELEVANT CODE

## 1. graph_backdoor_explanation_detection/src/attack/backdoor_utils.py

This file contains functions necessary for attacking datasets and training GNNs on the result. 

### Relevant functions:

	-- get_asr()
		Computes attack success rate (ASR), as mentioned in Section 1: Introduction of the paper (and referenced throughout).

	-- train_generator_single_iter()
		Single iteration of training adaptive trigger generator, as described in detail in Sectin E of the appendix.
		
	-- train_generator_iterative_loop()
		Iterative training of the adaptive trigger generator, as summarized on page 5 of our main paper, and detailed in Section E of the appendix.
		Alternates between retraining (1) the adaptive trigger generator and (2) the surrogate GNN it relies on for optimization.

	-- poison_data_adaptive_attack()
		Data poisoning process by trained adaptive trigger generator.

	-- random_trigger_synthesis()
		Random trigger generation, as described on page 3 of our paper (under "Threat model").

	-- poison_data_random_attack()
		Data poisoning process corresponding to random backdoor attack (Zhang et al.), as described on pages 2 and 3 of our paper.

	-- poison_data_clean_label_attack()
		Data poisoning process corresponding to clean-label attack (Xu and Picek, 2022) as described in Section E of the appendix.

	-- train_loop_backdoor()
		Training of GNN on poisoned dataset

## 2. graph_backdoor_explanation_detection/src/explain/explainer_utils.py

Functions for training explainer algorithms and returning explanations. In particular, GNNExplainer is important here, as it is used in our main experiments. See "GNN Explanation" on page 3 of our paper for more details on this process.

### Relevant funtions:

	-- run_explain()
		Function for training explainer (by default, GNNExplainer) and obtaining explanation on data input.
		GNNExplainer algorithm described on page 3 in our paper, and used in detection metrics and experiments throughout. 

## 3. graph_backdoor_explanation_detection/src/detect/metrics.py

This file contains functions pertaining to the 7 novel metrics introduced in our paper.

### Relevant functions:

	-- get_pred_confidence()
		Computes Metric 1: Prediction Confidence -- as defined in Section 4 (page 3) of our paper.

	-- get_explainability_score()
		Computes Metric 2: Explainability -- as defined in Equation (3) of our paper (Section 4, page 3)
		
	-- get_connectivity()
		Computes Metric 3: Connectivity -- as defined in Equation (5) of our paper (Section 4, page 4)

	-- node_degree_variance()
		Computes Metric 4: Node Degree Variance -- as defined in Equation (6) of our paper (page 4)

	-- subgraph_node_degree_variance():
		Computes Metric 5: Subgraph Node Degree Variance -- as defined in Equation (7) of our paper (Section 4, page 4)

	-- get_elbow_curvature()
		Computes Metric 6 and Metric 7
		Metric 6: Elbow -- as defined in Equation (8) of our paper (Section 4, page 4)
		Metric 7: Curvature -- as defined in Equation (9) of our paper (Section 4, page 4)

	-- get_boundary() and clean_val_boundary():
		Computes clean validation thresholds that are used to predict whether incoming samples are backdoor or clean.
		See "Clean Validation Extrema as Prediction Threshold" in our paper (Section 4, page 4)

	-- get_distance_from_clean_val()
		Used to compute "distance" metrics, as defined in equation 10 on (Section 4, page 5) of our paper.
		This is particularly used for elbow and curvature metrics -- see "Caveat for Loss Curve Metrics" on page 4.

	-- def display_detection_results():
		Computes the composite metric F1 score for detection for a provided NPMR (number of positive metrics required), 
		as described in Section 4 (page 5, under "Composite Metric") of paper.

# III. INSTRUCTIONS FOR USE

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

	python main.py --run_attack --run_explain --attack_target_label 1 --backdoor_type random --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
	python main.py --run_attack --run_explain --attack_target_label 1 --backdoor_type adaptive --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4
	python main.py --run_attack --run_explain --attack_target_label 1 --backdoor_type clean_label --dataset MUTAG --poison_rate 0.2 --model_hyp_set B --seed 2575 --trigger_size 4 --graph_type SW
