import argparse
import yaml
import json
import os
from random import choices
from secrets import choice

from utils.other_utils import bool_flag, check_path



def add_general_arguments(parser):
    # Config
    parser.add_argument("--config", default="./config/default.yaml", help="Hyper-parameters")
    # Debug
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    # Wandb
    parser.add_argument('--use_wandb', default=False, type=bool_flag, help='whether to use wandb')
    parser.add_argument('--log_interval', default=500, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    # Run 
    parser.add_argument('--run_name', default='debug', help='the current experiment name')
    parser.add_argument('--run_idx', default=0, type=int, help='the index of the run')
    parser.add_argument('--gpu_idx', default=0, type=str, help='GPU index')
    parser.add_argument('--seed', default=None, type=int, help='random seed')

    # Path
    parser.add_argument('--save_dir', default=None, help='model relevant output directory')
    parser.add_argument('--save_model', default=True, type=bool_flag,
                        help="whether to save the latest model checkpoints or not.")
    parser.add_argument('--load_pretrained_model_path', default=None, type=str)
    parser.add_argument('--load_model_path', default=None, type=str)
    parser.add_argument('--save_check', default=False, help='whether to save checkpoint ')
    parser.add_argument("--resume_id", default=None, type=str,
                        help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")
    parser.add_argument('--continue_train_from_check_path', default=None,
                        help='path of checkpoint to continue training')

def add_data_arguments(parser):
    
    parser.add_argument('--dataset', default='csqa', help='dataset name')
    parser.add_argument('--inhouse', default=False, type=bool_flag, nargs='?', const=True, help='run in-house setting')
    
    parser.add_argument('--max_input_len', default=128, type=int, help='max input length')
    parser.add_argument("--input_format", default='each_option', type=str, choices=['each_option','all_option'], help='The input format')
    parser.add_argument("--is_skip_last_batch", default=False, type=bool_flag, help='If skip the last mini-batch')

    # preprocessing options
    parser.add_argument('--few_shot', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether use few shot setting')
    parser.add_argument('--subsample', default=1.0, type=float, help='few shot ratio')

def add_model_arguments(parser):

    parser.add_argument('--experiment_model', default='lm_only', type=str, help='experiment model, such as qagnn, gnncounter ...')
    parser.add_argument('--pretrain_model', default='roberta-large', help='pretrain_model type')
    parser.add_argument('--pretrain_model_layer', default=-1, type=int, help='pretrain_model layer ID to use as features')

    # CET
    parser.add_argument('--is_CET', default=False, type=bool_flag, help='if using colliding effect')
    parser.add_argument('--CET_W0', default=0.9, type=float, help='the weight for anchor samples')
    parser.add_argument('--CET_topk', default=5, type=int, help='the number for reference answers')
    parser.add_argument('--CET_sim_thres',default=1.00, type=float, help='the minimum similarity for selecting KNN [0,1]')

    # NoisyTune
    parser.add_argument('--is_NoisyTune', default=False, type=bool_flag, help='if using noisy tune')
    parser.add_argument('--NoisyTune_lambda', default=0.15, type=float, help='the magnitude of the noisy')

    # ChildTuning
    parser.add_argument('--ChildTuning_mode', default='ChildTuning-D', type=str, choices=['ChildTuning-D','ChildTuning-F'], help='if using ChildTuning')
    parser.add_argument('--ChildTuning_reserve_p', default=0.3, type=float, help='ChildTuning hyper-parameter')

    # Re-Init
    parser.add_argument('--is_ReInit', default=False, type=bool_flag, help='if using re-initialization')
    parser.add_argument('--ReInit_topk_layer', default=3, type=int, help='the number of layer for re-initialization')

    # Mixout
    parser.add_argument('--is_Mixout', default=False, type=bool_flag, help='if using Mixout')
    parser.add_argument('--Mixout_prob', default=0.9, type=float, help='the probability of replacing modules')

    # Batch Spectral Shrinkage (BSS)
    parser.add_argument('--is_BSS',  default=False, type=bool_flag, help='if using bss')
    parser.add_argument('--BSS_weight',  default=0.001, type=float, help='the weight for bss term')

    # R3F
    parser.add_argument('--is_R3F', default=False, type=bool_flag, help='if using r3f')
    parser.add_argument('--R3F_eps', default=1e-5, type=float)
    parser.add_argument('--R3F_lambda', default=1.0, type=float)
    parser.add_argument('--R3F_noise_type', default='uniform', type=str)
    

def add_optimization_arguments(parser):
    # optimization
    parser.add_argument('--n_epochs', default=200, type=int, help='total number of training epochs to perform.')
    parser.add_argument('--accumulate_batch_size', default=128, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--final_fc_lr', default=1e-2, type=float, help='the learning rate for the final FC layer')
    parser.add_argument('--max_epochs_before_stop', default=10, type=int, help='stop training if dev does not increase for N epochs')
    parser.add_argument('--warmup_steps', type=float, default=150)

    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam', 'childtuningadamw','RecAdam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='l2 weight decay strength')
    
def get_args(is_save=False):
    """A helper function that handles the arguments for the whole experiment"""
    parser = argparse.ArgumentParser(description='Configurations for Commonsense QA')
    add_general_arguments(parser)
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_optimization_arguments(parser)
    args = parser.parse_args()

    # Get Settings
    with open(args.config) as f:
        config = yaml.safe_load(f)
        for k, v in config.items():
            if k=='use_wandb':
                print('')
            if k in args.__dict__.keys():
                old_v = args.__dict__[k]
                if isinstance(old_v, bool):
                    if isinstance(v, bool):
                        args.__setattr__(k, v)
                    else:
                        if v.lower() in ('no', 'false', 'f', 'n', '0'):
                            new_v = False
                        else:
                            new_v = True
                        args.__setattr__(k, bool(new_v))
                elif isinstance(old_v, float):
                    args.__setattr__(k, float(v))
                elif isinstance(old_v, int):
                    args.__setattr__(k, int(v))
                elif isinstance(old_v, str):
                    args.__setattr__(k, str(v))
                elif old_v is None:
                    args.__setattr__(k, v)
                else:
                    raise Exception('Invalid data type {old_v}')

    args.__setattr__('dataset_dir', 'data/%s/%s'%(args.dataset,'in_house' if args.inhouse else 'official'))
    args.__setattr__('train_statements','%s/train.jsonl'%(args.dataset_dir))
    args.__setattr__('dev_statements', '%s/dev.jsonl'%(args.dataset_dir))
    args.__setattr__('test_statements', '%s/test.jsonl'%(args.dataset_dir))

    # Set Save Paths
    args.__setattr__('save_dir','./save_models/%s/%s/%s/%s'%(
                    args.run_name,
                    'few_shot_%.2f'%(float(args.subsample)) if args.few_shot else 'full_set',
                    args.dataset,
                    'run_%s'%args.run_idx))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save Settings
    if is_save:
        param_dict = dict(vars(args))
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as fout:
            json.dump(param_dict, fout, indent=4)

    return args