from model import GCQA
from data import get_train_val_test_loader, ComplexDataset
from loss import EvaluateMetrics
import config
from arguments import buildParser
import sys, time, csv, os, random, math, argparse, gc
import numpy as np
import torch as t
import wandb
from scripts.utils import remove_noca, fix_pdb

os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"

def randomSeed(random_seed):
    if random_seed is not None:
        t.manual_seed(random_seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(random_seed)
            
def get_cuda(inputs, targets) : 
    
    in_size = len(inputs)
    tar_size = len(targets)
    assert in_size == tar_size, 'The size of input and target is not same'

    for i in range(in_size) : 
        index = t.tensor(i)
        input_var = inputs[index]
        target = targets[index]
        label = EvaluateMetrics(input_var, target).metric()
        lddt = label['lddt']
        interface = label['interface score']
        mDQ = label['mean DockQ']
        dev = label['deviation map']
        label_var = [lddt, interface, mDQ, dev]
        
    return input_var, label_var
        
            
class AverageMeter(object):
	def __init__(self, is_tensor=False, dimensions=None):
		if is_tensor and dimensions is None:
			print('Bad definition of AverageMeter!')
			sys.exit(1)
		self.is_tensor = is_tensor
		self.dimensions = dimensions
		self.reset()

	def reset(self):
		self.count = 0
		if self.is_tensor:
			self.val = t.zeros(self.dimensions, device=config.device)
			self.avg = t.zeros(self.dimensions, device=config.device)
			self.sum = t.zeros(self.dimensions, device=config.device)
		else:
			self.val = 0
			self.avg = 0
			self.sum = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def main() : 
    
    global args, dataset, savepath
    
    parser  = buildParser()
    args    = parser.parse_args()
    print('Torch Device being used: ', config.device)
    #t.device("cuda:0")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    savepath = args.save_dir + str(args.name) + '/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    randomSeed(args.seed)
    
    assert os.path.exists(args.protein_pred_list) and os.path.exists(args.protein_gt_list), '{} or {} does not exist!'.format(args.protein_pred_list, args.protein_gt_list)
    
    with open(args.protein_pred_list) as _list:
        reader = csv.reader(_list)
        all_pdbs = [row[0] for row in reader]
    
    data_len     = len(all_pdbs)
    indices     = list(range(data_len))
    random.shuffle(indices)

    train_size  = math.floor(args.train * data_len)
    val_size    = math.floor(args.val * data_len)
    test_size   = math.floor(args.test * data_len)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="gcmqa_cpu_three_mean_eps",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "Graphormer",
        "dataset": "ComplexData"
        }
    )
    print("wandb prepared")
    
    if val_size == 0:
        print('No protein directory given for validation!! Please recheck the split ratios, ignore if this is intended.')
    if test_size == 0:
        print('No protein directory given for testing!! Please recheck the split ratios, ignore if this is intended.')  
    
    test_datas   = all_pdbs[:test_size]
    train_datas  = all_pdbs[test_size:test_size + train_size]
    val_datas    = all_pdbs[test_size + train_size:test_size + train_size + val_size]
    print('Testing on {} protein directories:'.format(len(test_datas)))


    dataset = ComplexDataset(args.protein_gt_list, args.protein_pred_list, args.father_path, args.clean_pdb_dir, random_seed=args.seed)

    print('Dataset length: ', len(dataset))
    
    kwargs = {
        'input_list'        : args.protein_pred_list,
        'dgl_folder'        : args.dgl_dir,
        'n_layers'          : args.n_layers,
        'batch_size'        : args.batch_size,
        'h_size'            : args.h_size,
        'n_b_size'          : args.n_b_size,
        'n_c_size'          : args.n_c_size,
        'e_c_size'          : args.e_c_size,
        'n_bias_size'       : args.n_bias,
        'e_bias_size'       : args.e_bias,
        'ffn_size'          : args.ffn_size,
        'dropout'           : args.dropout,
        'a_dropout'         : args.a_dropout,
        'num_heads'         : args.num_heads,
        'l_mlp1'            : args.l_mlp1,
        'l_mlp2'            : args.l_mlp2,
        's_mlp1'            : args.l_mlp1,
        's_mlp2'            : args.l_mlp2,
        'e_out_hidden_size' : args.e_o_h_s
    }
    
    print("Let's use", t.cuda.device_count(), "GPUs and Data Parallel Model.")
    
    model = GCQA(**kwargs)
    # model = t.nn.DataParallel(model)
    #model.cuda()
    print(next(model.parameters()).device)
    
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, args.train, args.val, args.test, batch_size = args.batch_size, num_workers = args.workers, pin_memory = False)

    try:
            print('Training data    : ', len(train_loader.sampler))
            print('Validation data  : ', len(val_loader.sampler))
            print('Testing data     : ', len(test_loader.sampler))
    except Exception as e:
            print('\nException Cause: {}'.format(e.args[0]))
    
    for epoch in range(args.epochs) : 
        [train_loss, train_local, train_interface, train_global, train_dev] = Model(train_loader, model, epoch=epoch)
        [val_loss, val_local, val_interface, val_global, val_dev] = Model(val_loader, model, epoch=epoch, evaluation=True)
        
        if (val_local != val_local) or (val_global != val_global) or (val_interface != val_interface) or (val_dev != val_dev):
            print('Exit due to NaN')
            sys.exit(1)
            
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        best_local = val_local
        best_interface = val_interface
        best_global = val_global
        best_dev = val_dev
        
        if args.save_checkpoints:
            model.modules.save({
            'epoch'             : epoch,
            'state_dict'        : model.modules.state_dict(),
            'best_local'        : best_local,
            'best_interface'    : best_interface,
            'best_global'       : best_global,
            'best_dev'          : best_dev,
            'optimizer'         : model.modules.optimizer.state_dict(),
            'args'              : vars(args)
            }, is_best, savepath)
            
    if args.save_checkpoints and len(test_loader):
        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = t.load(savepath + 'model_best.pth.tar')
        model.modules.load_state_dict(best_checkpoint['state_dict'])
        [test_loss, test_local, test_interface, test_global, test_dev] = Model(test_loader, model, epoch=epoch, testing=True)

        
def Model(data_loader, model, epoch=None, evaluation=False, testing=False) : 
    
    batch_time          = AverageMeter()
    data_time           = AverageMeter()
    losses              = AverageMeter()
    batch_local         = AverageMeter()
    batch_interface     = AverageMeter()
    batch_global        = AverageMeter()
    batch_dev           = AverageMeter()
    
    if testing:
        test_preds = []
    
    end = time.time()
                            
    for i, (input_pdb, refer_gt) in enumerate(data_loader) : 
        
        batch_size = len(input_pdb)
        data_time.update(time.time() - end)
        
        remove_noca(input_pdb[0], input_pdb[0])
        fix_pdb(input_pdb[0], input_pdb[0])
        remove_noca(refer_gt[0], refer_gt[0])
        fix_pdb(refer_gt[0], refer_gt[0])
        print(input_pdb[0])
        _label = EvaluateMetrics(input_pdb[0], refer_gt[0]).metric()
        #_input, _label = get_cuda(input_pdb, refer_gt)
        
        if not evaluation and not testing:
            # Switch to train mode
            model.train()

            scores, deviation_map = model(input_pdb[0])
            assert deviation_map.size(0)  == _label['deviation map'].size(0) ,  "The dimension of predicted scores do not match with native labels"
            model.fit(scores, deviation_map, _label)
        
        else:

            with t.no_grad():

                model.eval()
                scores, deviation_map = model(input_pdb[0])
                assert deviation_map.size(0)  == _label['deviation map'].size(0) ,  "The dimension of predicted scores do not match with native labels"
                model.fit(scores, deviation_map, _label, pred=True)
                
        losses.update(model.loss.item(), batch_size)
        batch_local.update(model.sub_loss[0].item(), batch_size)
        batch_interface.update(model.sub_loss[1].item(), batch_size)
        batch_global.update(model.sub_loss[2].item(), batch_size)
        batch_dev.update(model.sub_loss[3].item(), batch_size)
        
        if testing : 
            test_local = scores['plddt']
            test_interface = scores['score'][0][0]
            test_global = scores['score'][0][1]
            test_dev = deviation_map

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if evaluation or testing:
                print('Test: [{0}][{1}]\t'
                        'Time {batch_time.val:.3f}\t'
                        'Loss {loss.val:.4f}\t'
                        'Local_loss {plddt.val:.4f}\t'
                        'Interface_loss {p_ics_ips.val:.4f}\t'
                        'Global_loss {pDockQ.val:.4f}\t'
                        'Deviation_loss {pdev.val:.4f}'.format(
                    epoch, input_pdb, batch_time=batch_time, loss=losses,
                    plddt=batch_local, p_ics_ips=batch_interface, pDockQ=batch_global, pdev=batch_dev))
            else:
                print('Epoch: [{0}]/\t'
                        'Time {batch_time.val:.3f}\t'
                        'Data {data_time.val:.3f}\t'
                        'Loss {loss.val:.4f}\t'
                        'Local_loss {plddt.val:.4f}\t'
                        'Interface_loss {p_ics_ips.val:.4f}\t'
                        'Global_loss {pDockQ.val:.4f}\t'
                        'Deviation_loss {pdev.val:.4f}'.format(
                    epoch, input_pdb, batch_time=batch_time, data_time=data_time, loss=losses,
                    plddt=batch_local, p_ics_ips=batch_interface, pDockQ=batch_global, pdev=batch_dev))

        wandb.log({"loss" : losses.val, "local-loss" : batch_local.val, "global-loss" : batch_global.val, "interface-loss" : batch_interface.val, "deviation-loss" : batch_dev.val})
        
        gc.collect()
        t.cuda.empty_cache()
        
        # if i % args.print_freq == 0:
        #     t.cuda.empty_cache()

    if testing:
        star_label = '**'
        with open(savepath + 'test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for input_pdb, test_local, test_interface, test_global in zip(input_pdb,
                                                                    test_local,
                                                                    test_interface,
                                                                    test_global):
                writer.writerow((input_pdb, test_local, test_interface, test_global))
    elif evaluation:
        star_label = 'validation'
    else:
        star_label = 'training'

    print(' {star} local {local.avg:.3f} interface {interface.avg:.3f} global {_global.avg:.3f} loss {loss.avg:.3f}'.format(
            star=star_label, local=batch_local.avg, interface=batch_interface, _global=batch_global, loss=losses))

    return losses.avg, batch_local.avg, batch_interface.avg, batch_global.avg, batch_dev.avg

if __name__ == '__main__':
    start = time.time()
    main()
    wandb.finish()
    print('Time taken: ', time.time() - start)
