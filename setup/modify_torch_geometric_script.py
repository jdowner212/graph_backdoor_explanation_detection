import shutil
import os

''' TODO: modify paths '''
# path/to/torch_geometric
dir_to_modify = '/opt/homebrew/Caskroom/miniforge/base/envs/graph_backdoor_detection/lib/python3.8/site-packages/torch_geometric'
# path/to/repo_pub/setup/modified_torch_geometric_files
modified_files_dir = '/Users/janedowner/Desktop/Desktop/IDEAL/Project_1/repo_pub/setup/modified_torch_geometric_files'

extensions_to_modify = ['experimental.py',
                        'explain/algorithm/base.py',
                        'explain/algorithm/gnn_explainer.py',
                        'explain/algorithm/pg_explainer.py',
                        'explain/explainer.py',
                        'explain/explanation.py',
                        'explain/metric/faithfulness.py',
                        'explain/metric/fidelity.py',
                        'nn/conv/message_passing.py',
                        'utils/subgraph.py']

if __name__ == '__main__':
    print('Modifying files...')
    for full_extension in extensions_to_modify:
        shortened_extension = full_extension.split('/')[-1]
        modified_file_path = os.path.join(modified_files_dir,shortened_extension)
        destination_path   = os.path.join(dir_to_modify,full_extension)
        assert os.path.exists(modified_file_path)
        assert os.path.exists(destination_path)
        print(destination_path)
        shutil.copyfile(modified_file_path, destination_path)