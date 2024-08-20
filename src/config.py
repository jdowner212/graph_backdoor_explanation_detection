# path will be changed by following setup instructions
root_dir = '/path/to/graph_backdoor_explanation_detection'


src_dir     = f'{root_dir}/src'
data_dir    = f'{root_dir}/data'
explain_dir = f'{root_dir}/explainer_results'
train_dir   = f'{root_dir}/training_results'
train_dir_cln = f'{root_dir}/training_results_clean'
adapt_gen_dir     = f'{root_dir}/generators_adaptive_trigger'
adapt_surrogate_models = f'{root_dir}/surrogate_models'




data_shape_dict = {'MUTAG': {'num_classes': 2, 'num_node_features': 5, 'max_degree':4}, 
                   'AIDS': {'num_classes': 2, 'num_node_features': 7, 'max_degree':6}, 
                   'PROTEINS': {'num_classes': 2, 'num_node_features': 26, 'max_degree':25},
                   'IMDB-BINARY': {'num_classes':2, 'num_node_features':136, 'max_degree':135},
                   'COLLAB': {'num_classes':3, 'num_node_features':492, 'max_degree':491},
                   'REDDIT-BINARY': {'num_classes':3, 'num_node_features':3063, 'max_degree':3062},
                   'DBLP': {'num_classes':2, 'num_node_features':36, 'max_degree':35}}




hyp_dict_backdoor = \
    {'MUTAG':
        {0:
            {
                'A': dict(model_type='gcn',   epochs=600,   dropout=0,   lr=0.0001,   weight_decay=0,   batchsize=200,  hidden_channels=128,   num_layers=4,   num_conv_layers=3,    batchnorm=[False, False, False],   balanced=True),
                'B': dict(model_type='gcn',   epochs=350,   dropout=0.5,   lr=0.00005,   weight_decay=0.001,   batchsize=200,  hidden_channels=256,   num_layers=3,   num_conv_layers=4,   batchnorm=[True, True, True, True],   balanced=True)
            },
        1:
            {
                'A': dict(model_type='gcn',  epochs=300,  dropout=0,  lr=0.0001,  weight_decay=0,  batchsize=200,hidden_channels=128, num_layers=2, num_conv_layers=4, batchnorm=[False, False, False, True], balanced=True),
                'B': dict(model_type='gin2',   epochs=400,   dropout=0,   lr=0.00005,   weight_decay=0.05,   batchsize=200,  hidden_channels=80,   num_layers=2,  num_conv_layers=4,   batchnorm=[True, True, True, False],  balanced=True)
            }
        },

    'AIDS':
        {0:
            {
                'A': dict(model_type='gin',   epochs=300,   dropout=0.5,   lr=0.0001,   weight_decay=1e-4,   batchsize=250,  hidden_channels=48,   num_conv_layers=None,   batchnorm=None,   balanced=True),
                'B': dict(model_type='gin',   epochs=150,   dropout=0,   lr=1e-4,   weight_decay=1e-5,  batchsize=250, hidden_channels=32, num_conv_layers=None,  batchnorm=None,  balanced=True)},
        1:
            {
                'A': dict(model_type='gin', epochs=300, num_heads=2,dropout_forward=0.5, dropout_gat=0.6, dropout=0.5, ratio=0.5,lr=0.0001, weight_decay=0.1, batchsize=250,hidden_channels=32,  num_layers=2,num_conv_layers=2, batchnorm=[True,True], balanced=True),
                'B': dict(model_type='gin',  epochs=200,  dropout=0.3,  lr=1e-4,  weight_decay=0.1,  batchsize=250, hidden_channels=16,  num_layers=2, num_conv_layers=None,  batchnorm=None,  balanced=True)
            }
        },
    'PROTEINS':
        {0:
            {
                'A': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=650, hidden_channels=256,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True],balanced=True),
                'B': dict(model_type='gcn_plain',  epochs=150,  dropout=0.95,  lr=0.0005,  weight_decay=0.0005, batchsize=217,  hidden_channels=256,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True], balanced=True),
                'C': dict(model_type='gin2', epochs=200, dropout=0.99, lr=0.0005, weight_decay=0.005, batchsize=217, hidden_channels=256, num_conv_layers=3, num_layers=2,  batchnorm=[True, True, True, True], balanced=True),
            },
        1:
            {
                'A': dict(model_type='carate',    ratio=0.9,   epochs=200,   dropout=0.95,   lr=0.001,   weight_decay=0.0005,  batchsize=217,   hidden_channels=128,   num_heads=3,   dropout_forward=0.5,   dropout_gat = 0.5,   num_layers=3,   batchnorm=[True, True, True, True],   balanced=True),
                'B': dict(model_type='gcn_plain',   epochs=150,   dropout=0.95,   lr=0.0005,   weight_decay=0.0005,  batchsize=217,   hidden_channels=256,   num_layers=2,   batchnorm=[True, True, True, True],   balanced=True),
                'C': dict(model_type='gin',   epochs=300,   dropout=0.8,   lr=0.00005,   weight_decay=0.0005,   batchsize=217,  hidden_channels=128,   batchnorm=None,   balanced=False)
            }
        },

    'IMDB-BINARY':
        {0:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=400,  lr=0.001,  weight_decay=1e-2, batchsize=250, hidden_channels=64,  num_layers=2, balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=250, hidden_channels=256,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
            },
        1:
            {
                'A': dict(model_type='graphlevelgnn',    epochs=400,    lr=0.001,    weight_decay=1e-2,   batchsize=250,    hidden_channels=64,   num_layers=2,   balanced=True,  c_out=2),
                'B': dict(model_type='gin3',    epochs=200,    dropout=0.99,    lr=0.0003,    weight_decay=0.0,   batchsize=250,   hidden_channels=256,   num_heads=3,  dropout_forward=0,  dropout_gat = 0,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True],  balanced=True),
            }
        },

    'DBLP':
        {0:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=200,  lr=0.01,  weight_decay=0,batchsize=334, hidden_channels=32,  num_layers=2,balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=334, hidden_channels=32,  num_heads=3,dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
                'C': dict(model_type='gcn',  epochs=200,  dropout=0.5,  lr=0.001,  weight_decay=0.0, batchsize=334, hidden_channels=32,  num_heads=3,dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
                'D': dict(model_type='gcn',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=334, hidden_channels=32,  num_heads=3,dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),

            },
        1:
            {
                'A': dict(model_type='graphlevelgnn',   epochs=200,   lr=0.01,   weight_decay=0,  batchsize=334,   hidden_channels=32,   num_layers=2,   balanced=True,  c_out=2),
                'B': dict(model_type='gin3',   epochs=200,   dropout=0.99,   lr=0.0003,   weight_decay=0.0,  batchsize=334,  hidden_channels=32,   num_heads=3, dropout_forward=0,  dropout_gat = 0,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True],  balanced=True),
                'C': dict(model_type='gcn',   epochs=200,   dropout=0.5,   lr=0.001,   weight_decay=0.0,  batchsize=334,  hidden_channels=32,   num_heads=3, dropout_forward=0,  dropout_gat = 0,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True],  balanced=True),
                'D': dict(model_type='gcn',   epochs=200,   dropout=0.99,   lr=0.0003,   weight_decay=0.0,  batchsize=334,  hidden_channels=32,   num_heads=3, dropout_forward=0,  dropout_gat = 0,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True],  balanced=True),

            }
        },
}


hyp_dict_backdoor_adaptive = \
    {'MUTAG':
        {0:
            {
                'A': dict(model_type='gcn',  epochs=600,  dropout=0,  lr=0.001,  weight_decay=0,  batchsize=200,hidden_channels=128,  num_layers=4,  num_conv_layers=3, batchnorm=[False, False, False],  balanced=True),
                'B': dict(model_type='gcn', epochs=350, dropout=0.5,  lr=0.0005, weight_decay=0.001, batchsize=200,hidden_channels=256, num_layers=3, num_conv_layers=4, batchnorm=[True, True, True, True], balanced=True)
            },
        1:
            {
                'A': dict(model_type='gcn',   epochs=300,  dropout=0,  lr=0.0001,   weight_decay=0,   batchsize=200,  hidden_channels=128,   num_layers=2,   num_conv_layers=4,   batchnorm=[False, False, False, True],   balanced=True),
                'B': dict(model_type='gin2',   epochs=400,   dropout=0,  lr=0.00005,   weight_decay=0.05,  batchsize=200, hidden_channels=80,  num_layers=2,  num_conv_layers=4,   batchnorm=[True, True, True, False], balanced=True)
            }
        },
    'AIDS':
        {0:
            {
                'A': dict(model_type='gin', epochs=100,  dropout=0.5,  lr=0.0001,  weight_decay=1e-4,  batchsize=250, hidden_channels=48,  num_conv_layers=None,  batchnorm=None,  balanced=True),
                'B': dict(model_type='gin',  epochs=150,  dropout=0,  lr=1e-4,  weight_decay=1e-5,  batchsize=250, hidden_channels=32, num_conv_layers=None,  batchnorm=None,  balanced=True)},
            1:
            {
                'A': dict(model_type='gin',  epochs=300,  num_heads=2, dropout_forward=0.5,  dropout_gat=0.6,  dropout=0.5,  ratio=0.5, lr=0.0001,  weight_decay=0.1,  batchsize=250, hidden_channels=32,   num_layers=2, num_conv_layers=2,  batchnorm=[True,True],  balanced=True),'B': dict(model_type='gin',  epochs=200,  dropout=0.3,  lr=1e-4,  weight_decay=0.1,  batchsize=250, hidden_channels=16,  num_layers=2, num_conv_layers=None,  batchnorm=None,  balanced=True)
            }
        },
    'PROTEINS':
        {0:
            {
                'A': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=650, hidden_channels=256,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
                'B': dict(model_type='gcn_plain',  epochs=150,  dropout=0.95,  lr=0.0005,  weight_decay=0.0005, batchsize=217,  hidden_channels=256,  num_conv_layers=4,  num_layers=2,  batchnorm=[True, True, True, True], balanced=True),
                'C': dict(model_type='gin2', epochs=200, dropout=0.99, lr=0.0005, weight_decay=0.005, batchsize=650, hidden_channels=256, num_conv_layers=3, num_layers=2,  batchnorm=[True, True, True, True], balanced=True),
            },
        1:
            {
                'A': dict(model_type='carate',  ratio=0.9,  epochs=200,   dropout=0.95,  lr=0.001,  weight_decay=0.0005, batchsize=217,  hidden_channels=128,  num_heads=3,  dropout_forward=0.5,  dropout_gat = 0.5,  num_layers=3,  batchnorm=[True, True, True, True],  balanced=True),
                'B': dict(model_type='gcn_plain',   epochs=150,   dropout=0.95,   lr=0.0005,   weight_decay=0.0005,  batchsize=217,   hidden_channels=256,   num_layers=2,   batchnorm=[True, True, True, True],   balanced=True),
                'C': dict(model_type='gin',   epochs=300,   dropout=0.8,   lr=0.00005,   weight_decay=0.0005,   batchsize=217,  hidden_channels=128,   batchnorm=None,  balanced=False)
            }
        },
   'IMDB-BINARY':
        {0:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=200,  lr=0.001,  weight_decay=1e-2, batchsize=250, hidden_channels=64,  num_layers=2, balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,   weight_decay=0.0,  batchsize=250, hidden_channels=256,  num_heads=3,  dropout_forward=0, dropout_gat = 0,  num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
            },
        1:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=200,  lr=0.001,  weight_decay=1e-2, batchsize=250,  hidden_channels=64,  num_layers=2,  balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=200,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=250, hidden_channels=256,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
            }
        },

    'DBLP':
        {0:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=400,  lr=0.001,  weight_decay=1e-2, batchsize=417, hidden_channels=16,  num_layers=2, balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=400,  dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=417, hidden_channels=32,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
            },
        1:
            {
                'A': dict(model_type='graphlevelgnn',  epochs=100, lr=0.001,  weight_decay=1e-2, batchsize=417,  hidden_channels=16,  num_layers=2,  balanced=True, c_out=2),
                'B': dict(model_type='gin3',  epochs=100, dropout=0.99,  lr=0.0003,  weight_decay=0.0, batchsize=417, hidden_channels=32,  num_heads=3, dropout_forward=0, dropout_gat = 0, num_conv_layers=4, num_layers=2, batchnorm=[True, True, True, True], balanced=True),
            }
        },
    }




surrogate_hyperparams_initial = {'MUTAG':       {'c_hidden': 128, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.5,'max_epochs':200},
                                'AIDS':         {'c_hidden': 128, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.5,'max_epochs':200},
                                'PROTEINS':     {'c_hidden': 128, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.5,'max_epochs':200},
                                'IMDB-BINARY':  {'c_hidden': 128, 'lr': 1e-4, 'weight_decay': 0, 'num_layers':3, 'batch_size':256,'dp_rate_linear':0.8,'max_epochs':400},                                
                                'DBLP':         {'c_hidden': 512, 'lr': 1e-4, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.8,'max_epochs':200}}

surrogate_hyperparams_looping = { 'MUTAG':      {'c_hidden': 128, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.5,'max_epochs':200},
                                  'AIDS':       {'c_hidden': 96, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':128,'dp_rate_linear':0.5,'max_epochs':100},
                                  'PROTEINS':   {'c_hidden': 96, 'lr': 1e-2, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.5,'max_epochs':100},
                                  'IMDB-BINARY':{'c_hidden': 128, 'lr': 1e-4, 'weight_decay': 0, 'num_layers':3, 'batch_size':256,'dp_rate_linear':0.8,'max_epochs':100},                                    
                                  'DBLP':       {'c_hidden': 512, 'lr': 1e-4, 'weight_decay': 0, 'num_layers':2, 'batch_size':64,'dp_rate_linear':0.8,'max_epochs':50}}



generator_hyperparam_dicts =    {
                                'MUTAG':        [{'generator_class': 'heavy',   'epochs': 20,  'T': 100,    'lr_Ma': 2, 'lr_gen': 0.05, 'weight_decay': 0, 'hidden_dim':16,     'depth':2, 'dropout_prob':  0,  'batch_size':24,    'max_num_edges': 2},
                                                 {'generator_class': 'heavy',   'epochs': 20,  'T': 100,    'lr_Ma': 2, 'lr_gen': 0.05, 'weight_decay': 0, 'hidden_dim':16,     'depth':2, 'dropout_prob':  0,  'batch_size':12,    'max_num_edges': 1}],
                                'AIDS':         [{'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 1, 'lr_gen': 0.005,'weight_decay': 0, 'hidden_dim':64,     'depth':2, 'dropout_prob':  0,  'batch_size':32,    'max_num_edges': 16},
                                                 {'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 1, 'lr_gen': 0.005,'weight_decay': 0, 'hidden_dim':128,    'depth':2, 'dropout_prob':  0,  'batch_size':32,    'max_num_edges': 16}],
                                'PROTEINS':     [{'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 5, 'lr_gen': 0.01, 'weight_decay': 0, 'hidden_dim':16,     'depth':3, 'dropout_prob':  0,  'batch_size':16,    'max_num_edges': 16},
                                                 {'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 5, 'lr_gen': 0.01, 'weight_decay': 0, 'hidden_dim':16,     'depth':3, 'dropout_prob':  0,  'batch_size':16,    'max_num_edges': 16}],
                                'IMDB-BINARY':  [{'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 5, 'lr_gen': 0.05, 'weight_decay': 0, 'hidden_dim':128,    'depth':2, 'dropout_prob':  0,  'batch_size':32,    'max_num_edges': 16},
                                                 {'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 5, 'lr_gen': 0.05, 'weight_decay': 0, 'hidden_dim':80,     'depth':1, 'dropout_prob':  0,  'batch_size':16,    'max_num_edges': 16}],
                                'DBLP':         [{'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 15,'lr_gen': 0.01, 'weight_decay': 0, 'hidden_dim':16,     'depth':3, 'dropout_prob':  0,  'batch_size':128,   'max_num_edges': 16},
                                                 {'generator_class': 'heavy',   'epochs': 20,  'T': 50,     'lr_Ma': 15,'lr_gen': 0.01, 'weight_decay': 0, 'hidden_dim':16,     'depth':3, 'dropout_prob':  0,  'batch_size':128,   'max_num_edges': 16}]
                                }
                                    


metric_plot_info_dict = {'loss_max':                    {'DF': 'Max Loss',              'Title': 'Maximum Loss',                                        'x_label': 'loss',              'inequality': 'less'},
                         'loss_min':                    {'DF': 'Min Loss',              'Title': 'Minimum Loss',                                        'x_label': 'loss',              'inequality': 'less'},
                         'elbow':                       {'DF': 'Elbow',                 'Title': 'Elbow',                                               'x_label': 'epoch',             'inequality': 'less'},
                         'curv':                        {'DF': 'Curv',                  'Title': 'Curvature',                                           'x_label': 'curvature',         'inequality': 'more'},
                         'es':                          {'DF': 'Expl. Score',           'Title': 'Explainability',                                      'x_label': 'score',             'inequality': 'more'},
                         'unfaith':                     {'DF': 'Unfaith',               'Title': 'Unfaithfulness',                                      'x_label': 'unfaithfulness',    'inequality': 'less'},
                         'connectivity':                {'DF': 'Connectivity',          'Title': 'Connectivity',                                        'x_label': 'connectivity',      'inequality': 'more'},
                         'pred_conf':                   {'DF': 'Prediction Conf',       'Title': 'Prediction Confidence',                               'x_label': 'confidence',        'inequality': 'more'},
                         'node_deg_var':                {'DF': 'Node Degree Var',       'Title': 'Node Degree Variance',                                'x_label': 'variance',          'inequality': 'more'},
                         'mask_feat_var':               {'DF': 'Mask Feature Var',      'Title': 'Subgraph Node Degree Variance',                       'x_label': 'variance',          'inequality': 'more'},
                         'loss_max_dist':               {'DF': 'Max Loss Dist',         'Title': 'Maximum Loss',                                        'x_label': 't-score',           'inequality': 'more'},
                         'loss_min_dist':               {'DF': 'Min Loss Dist',         'Title': 'Minimum Loss',                                        'x_label': 't-score',           'inequality': 'more'},
                         'elbow_dist':                  {'DF': 'Elbow Dist',            'Title': 'Elbow',                                               'x_label': 't-score',           'inequality': 'more'},
                         'curv_dist':                   {'DF': 'Curv Dist',             'Title': 'Curvature',                                           'x_label': 't-score',           'inequality': 'more'},
                         'es_dist':                     {'DF': 'Expl. Score Dist',      'Title': 'Explainability',                                      'x_label': 't-score',           'inequality': 'more'},
                         'unfaith_dist':                {'DF': 'Unfaith Dist',          'Title': 'Unfaithfulness',                                      'x_label': 't-score',           'inequality': 'more'},
                         'connectivity_dist':           {'DF': 'Connectivity Dist',     'Title': 'Connectivity',                                        'x_label': 't_score',           'inequality': 'more'},
                         'pred_conf_dist':              {'DF': 'Prediction Conf Dist',  'Title': 'Prediction Confidence',                               'x_label': 't_score',           'inequality': 'more'},
                         'node_deg_var_dist':           {'DF': 'Node Degree Var Dist',  'Title': 'Node Degree Variance',                                'x_label': 't-score',           'inequality': 'more'},
                         'mask_feat_var_dist':          {'DF': 'Mask Feature Var Dist', 'Title': 'Subgraph Node Degree Variance',                       'x_label': 't-score',           'inequality': 'more'},
                         }



def get_info(name):
    g = globals()
    return g.get(name, g.get(name, {}))