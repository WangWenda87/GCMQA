import configargparse

def buildParser():
	parser = configargparse.ArgParser(default_config_files=['settings.conf'])

	# Data source
	parser.add('--name',          default='three_loss_mean_CPU',                     help='Name of folder where result is saved')
	parser.add('--father_path', default='/home/u2021103648/workspace/dataQA/', help='father path')
	parser.add('--clean_pdb_dir', default='/home/u2021103648/workspace/dataQA/clean_pdb/',          help='clean pdb folder')
	parser.add('--protein_gt_list', default='/home/u2021103648/workspace/dataQA/gt_950.csv',          help='list where all protein native pdb files exist')
	parser.add('--protein_pred_list', default='/home/u2021103648/workspace/dataQA/pred_950.csv',          help='list where all protein predicted pdb files exist')
	parser.add('--dgl_dir',     default='/home/u2021103648/workspace/dataQA/dgl/',  help='Destination directory for dgls')
	parser.add('--save_dir',    default='/home/u2021103648/workspace/dataQA/results/',      help='Destination directory for results')
	parser.add('--pretrained',                                      help='Path to pretrained model')
	parser.add('--avg_sample',  default=500,                        help='Normalizer sample count for calculating mean and std of target', type=int)

	# Training setup
	parser.add('--seed',        default=123,                       help='Seed for random number generation',   type=int)
	parser.add('--epochs',      default=10,                        help='Number of epochs',                    type=int)
	parser.add('--batch_size',  default=1,                          help='Batch size for training',             type=int)
	parser.add('--train',       default=0.7,                        help='Fraction of training data',           type=float)
	parser.add('--val',         default=0.1,                       help='Fraction of validation data',         type=float)
	parser.add('--test',        default=0.2,                       help='Fraction of test data',               type=float)
	parser.add('--testing',                                         help='If only testing the model',           action='store_true')

	# Optimizer setup
	parser.add('--lr',          default=0.001,                      help='Learning rate',                       type=float)

	# Model setup

	parser.add('--h_size',      default=256,                        help='hidden size',                         type=int)
	parser.add('--n_b_size',    default=168,                        help='node basic feature embedding dict dimension',      type=int)
	parser.add('--n_c_size',    default=128,                        help='node contact feature embedding dict dimension',    type=int)
	parser.add('--e_c_size',    default=128,                        help='edge contact feature embedding dict dimension',    type=int)
	parser.add('--n_bias',      default=64,                         help='node bias encoding medium dimension', type=int)
	parser.add('--e_bias',      default=64,                         help='edge bias encoding embedding dimension',           type=int)
	parser.add('--ffn_size',    default=64,                         help='ffn size',                            type=int)
	parser.add('--dropout',     default=0.2,                        help='dropout rate',                        type=int)
	parser.add('--a_dropout',   default=0.1,                        help='attention dropout rate',              type=int)
	parser.add('--num_heads',   default=8,                          help='multi head nums',                     type=int)
	parser.add('--l_mlp1',      default=256,                        help='plddt readout 1',                     type=int)
	parser.add('--l_mlp2',      default=128,                        help='plddt readout 2',                     type=int)
	parser.add('--s_mlp1',      default=256,                        help='score readout 1',                     type=int)
	parser.add('--s_mlp2',      default=128,                        help='score readout 2',                     type=int)
	parser.add('--e_o_h_s',     default=512,                        help='deviation readout',                   type=int)

	parser.add('--n_layers',    default=12,                         help='number of graph-attention layers',    type=int)
	# Other features

	parser.add('--save_checkpoints',    default=True,               help='Stores checkpoints if true',                      action='store_true')
	parser.add('--print_freq',          default=1,                 help='Frequency of printing updates between epochs',    type=int)
	parser.add('--workers',             default=2,                 help='Number of workers for data loading',              type=int)

	return parser

# print(buildParser().parse_args())
