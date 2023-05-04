import numpy as np
import random
from tqdm import tqdm, trange
import os
# Specify CUDA_VISIBLE_DEVICES in the command, 
# e.g., CUDA_VISIBLE_DEVICES=0,1 nohup bash exp_on_b7server_0.sh
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import time
import json
import warnings
warnings.filterwarnings('ignore')
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast

from utils.parser_utils import get_args
from utils.logger_utils import get_logger
from utils.other_utils import *
from utils.optimization_utils import *
from utils.mixout_utils import *
from modeling.bert_models import *

def evaluate_accuracy(dev_loader, model):
    n_corrects_acm_eval, n_samples_acm_eval = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        num_batch = len(dev_loader)
        for batch_idx in tqdm(list(range(num_batch)),total=num_batch,desc='Evaluation'):

            input_data = dev_loader[batch_idx]
            labels = input_data['example_label']

            logits = model.predict(input_data)

            bs = logits.shape[0]
            n_corrects = n_corrects = (logits.argmax(1) == labels).sum().item()
            n_corrects_acm_eval += n_corrects
            n_samples_acm_eval += bs

    ave_acc_eval = n_corrects_acm_eval / n_samples_acm_eval
    return ave_acc_eval

def set_random_seed(seed):
    if not seed is None:
        logger.info("Fix random seed")
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.info("Use Random Seed")

def set_wandb(args):
    wandb_mode = "online" if args.use_wandb and (not args.debug) else "disabled" 
    resume = (args.continue_train_from_check_path is not None) and (args.resume_id != "None" and args.resume_id is not None)
    args.wandb_id = args.resume_id if resume else wandb.util.generate_id()
    args.hf_version = transformers.__version__
    wandb_log = wandb.init(mode=wandb_mode, entity="your-entity", project="your-project", config=args, name=args.run_name, resume="allow", id=args.wandb_id, settings=wandb.Settings(start_method="fork"))
    logger.info('{0:>30}: {1}'.format("wandb id", args.wandb_id))
    return wandb_log

def main(args):
    set_random_seed(args.seed)
    print_system_info()
    print_basic_info(args)
    wandb_log = set_wandb(args)
    train(args,wandb_log)

def train(args, wandb_log):
    logger.info('=' * 71)
    logger.info('Start Training')
    logger.info('=' * 71)

    ###################################################################################################
    #   Get available GPU devices                                                                     #
    ###################################################################################################
    assert torch.cuda.is_available() and torch.cuda.device_count()>=1, 'No gpu avaliable!'

    is_data_parellel = False
    # Note: Only using the pre-defined gpu_idx when debug; Otherwise, use CUDA_VISIBLE_DEVICES to specify the devices
    if (not args.use_wandb) and (args.gpu_idx is not None):
        gpu_idx = args.gpu_idx
        if isinstance(gpu_idx,int) or (isinstance(gpu_idx,str) and gpu_idx.isdigit()):
            devices = torch.device(gpu_idx)
        else:
            raise Exception('Invalid gpu_idx {gpu_idx}')
    else:
        # logger.info('{0:>30}: {1}'.format('Visible GPU count',torch.cuda.device_count()))
        devices = torch.device(0)
        if torch.cuda.device_count()>1:
            is_data_parellel = True
    # logger.info('{0:>30}: {1}'.format('Using visible GPU',str(devices)))

    # for debug
    # devices = 'cpu'

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    logger.info("Build model")
    if 'bert' in args.pretrain_model:
        model = BERT_basic(args)
    else:
        raise Exception('Invalid pretrain_model name %s'%args.pretrain_model)

    # Re-Init
    if args.is_ReInit:
        # First: Obtain a fully randomly initialized pretrained model
        random_init_pretrain_model = deepcopy(model.pretrain_model)
        random_init_pretrain_model.apply(random_init_pretrain_model._init_weights) # using apply() to init each submodule recursively
        # Then: Set the top layers in the pretrained model 
        if hasattr(random_init_pretrain_model.config,'num_layers'):
            num_layers = random_init_pretrain_model.config.num_layers
        elif hasattr(random_init_pretrain_model.config,'num_hidden_layers'):
            num_layers = random_init_pretrain_model.config.num_hidden_layers 
        else:
            raise Exception('Cannot find number of layers in model.configs!!!')
        ignore_layers = [layer_i for layer_i in range(num_layers-args.ReInit_topk_layer)]
        reinit_lst = []
        
        for _name, _para in model.pretrain_model.named_parameters():
            # Word embedding don't need initialization
            if 'shared' in _name or 'embeddings' in _name:
                continue
            # for bert
            if 'layer.' in _name:
                start_idx = _name.find('layer.') +len('layer.')
                end_idx = _name.find('.', start_idx)
                layer_id = int(_name[start_idx:end_idx])
                if layer_id in ignore_layers:
                    continue
            
            model.pretrain_model.state_dict()[_name][:] = random_init_pretrain_model.state_dict()[_name][:]
            reinit_lst.append(_name)
        logger.info('Reinit modules: %s'%reinit_lst)
        del random_init_pretrain_model

    # NoisyTune
    if args.is_NoisyTune:
        for _name, _para in model.pretrain_model.named_parameters():
            model.pretrain_model.state_dict()[_name][:] += (torch.rand(_para.size())-0.5)*args.NoisyTune_lambda*torch.std(_para)

    # Mixout
    if args.is_Mixout:
        # use tuple to avoid OrderedDict warning
        for name, module in tuple(model.pretrain_model.named_modules()):
            if name:
                recursive_setattr(model.pretrain_model, name, replace_layer_for_mixout(module, mixout_prob=args.Mixout_prob))

    # DataParellel
    if is_data_parellel:
        logger.info('Using data parallel ...')
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    logger.info('Parameters statistics')
    params_statistic(model)

    ###################################################################################################
    #   Resume from checkpoint                                                                        #
    ###################################################################################################
    start_epoch=0
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt')
    if args.continue_train_from_check_path is not None and args.continue_train_from_check_path != 'None':
        logger.info("Resume from checkpoint %s"%args.continue_train_from_check_path)
        if torch.cuda.is_available():
            check = torch.load(args.continue_train_from_check_path)  
        else: 
            check = torch.load(args.continue_train_from_check_path,map_location=torch.device('cpu'))
        model_state_dict, _ = check
        model.load_state_dict(model_state_dict)
        model.train()

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    logger.info("Load dataset and dataloader")
    dataset = Basic_Dataloader(args, devices=devices)
    dev_loader = dataset.dev()
    test_loader = dataset.test()
    train_loader = dataset.train()

    ###################################################################################################
    #   Build Optimizer                                                                               #
    ###################################################################################################
    logger.info("Build optimizer")
    optimizer, scheduler = get_optimizer(model, args, dataset)

    # ChildTune
    if args.optim == 'childtuningadamw' and  args.ChildTuning_mode == 'ChildTuning-D':
        model = model.to(devices)
        gradient_mask = calculate_fisher(args, model, train_loader)
        optimizer.set_gradient_mask(gradient_mask)
        model = model.cpu()

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################
    model.train()
    freeze_net(model.pretrain_model)
    logger.info("Freeze model.pretrain_model")

    model.to(devices)

    # record variables
    dev_acc = 0
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, best_test_acc = 0.0, 0.0, 0.0
    total_loss_acm, n_corrects_acm, n_samples_acm = 0.0, 0.0, 0.0
    best_dev_acc = dev_acc

    is_finish = False
    accumulate_batch_num = args.accumulate_batch_size//args.batch_size
    for epoch_id in trange(start_epoch, args.n_epochs, desc="Epoch"):

        model.epoch_idx = epoch_id

        if is_finish:
            break

        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.pretrain_model)
            logger.info("Unfreeze model.pretrain_model")
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.pretrain_model)
            logger.info("Freeze model.pretrain_model")

        model.train()

        start_time = time.time()
        
        num_batch = len(train_loader)-1 if args.is_skip_last_batch else len(train_loader)

        for batch_id in tqdm(range(num_batch), total=num_batch, desc="Batch"):
            # load data for one batch
            input_data = train_loader.__getitem__(batch_id, is_skip_last_batch=args.is_skip_last_batch) 
            labels = input_data['example_label']
            bs = len(input_data['example_id'])
            
            if args.is_CET:
                loss, logits = model.compute_CET_loss(input_data, labels)
            elif args.is_BSS:
                loss, logits = model.compute_BSS_loss(input_data, labels)
            elif args.is_R3F:
                loss, logits = model.compute_R3F_loss(input_data, labels)
            else:
                loss, logits = model.compute_loss(input_data, labels)

            total_loss_acm += loss.item()*bs
            loss.requires_grad_(True)  
            loss.backward()

            n_corrects = (logits.detach().argmax(1) == labels).sum().item() if logits is not None else 0
            n_corrects_acm += n_corrects
            n_samples_acm += bs

            if (batch_id+1)%accumulate_batch_num==0 or batch_id==num_batch-1:
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                logger.info('| step {:5} | lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step+1, scheduler.get_last_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))

                if not args.debug:
                    wandb_log.log({"lr": scheduler.get_last_lr()[0], "train_loss": total_loss_acm / n_samples_acm, "train_acc": n_corrects_acm / n_samples_acm, "ms_per_batch": ms_per_batch}, step=global_step+1)

                total_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                start_time = time.time()

            global_step += 1

        if epoch_id%args.eval_interval==0:

            model.eval()
            dev_acc = evaluate_accuracy(dev_loader, model)

            test_acc = 0.0
            total_acc = []
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    num_batch = len(test_loader)
                    for batch_idx in tqdm(list(range(num_batch)),total=num_batch,desc='Testing'):
                        input_data = test_loader[batch_idx]
                        qids = input_data['example_id']
                        labels = input_data['example_label']
                        logits = model.predict(input_data)
                        predictions = logits.argmax(1) #[bsize, ]
                        # preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                        for i, (qid, label, pred) in enumerate(zip(qids, labels, predictions)):
                            acc = int(pred.item()==label.item())
                            f_preds.write('{},{}\n'.format(qid, chr(ord('A') + pred.item())))
                            f_preds.flush()
                            total_acc.append(acc)
            test_acc = float(sum(total_acc))/len(total_acc)

            best_test_acc = max(test_acc, best_test_acc)
            if epoch_id >= args.unfreeze_epoch:
                # update record variables
                if dev_acc >= best_dev_acc:
                    best_dev_acc = dev_acc
                    final_test_acc = test_acc
                    best_dev_epoch = epoch_id
                    if args.save_model:
                        model_path = os.path.join(args.save_dir, 'model.pt')
                        torch.save([model.state_dict(), args], model_path)
                        logger.info("model saved to %s"%model_path)
            else:
                best_dev_epoch = epoch_id

            logger.info('-' * 71)
            logger.info(
                '| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc,
                                                                                        test_acc))
            logger.info('| best_dev_epoch {:3} | best_dev_acc {:7.4f} | final_test_acc {:7.4f} |'.format(best_dev_epoch,
                                                                                                        best_dev_acc,
                                                                                        final_test_acc))
            logger.info('-' * 71)

            if not args.debug:
                wandb_log.log({"dev_acc": dev_acc, "dev_loss": dev_acc, "best_dev_acc": best_dev_acc,
                            "best_dev_epoch": best_dev_epoch}, step=global_step)
                if test_acc > 0:
                    wandb_log.log({"test_acc": test_acc, "test_loss": 0.0, "final_test_acc": final_test_acc},
                            step=global_step)

            if args.save_check:
                training_dict = {'epoch':epoch_id, 'loss':loss,
                                'model_state_dict':model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'scheduler_dict':scheduler.state_dict()}
                torch.save(training_dict, checkpoint_path)

            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                logger.info("After %d epoch no improving. Stop!"%(epoch_id-best_dev_epoch))
                logger.info("Best test accuracy: %s"%str(best_test_acc))
                logger.info("Final best test accuracy according to dev: %s"%str(final_test_acc))
                is_finish=True
                break
            model.train()

    ###################################################################################################
    #   Testing                                                                                       #
    ###################################################################################################
    if args.n_epochs <= 0:
        logger.info('n_epochs <= 0, start testing ...')
        model.eval()
        with torch.no_grad():
            dev_acc = evaluate_accuracy(dev_loader, model)
            test_acc = evaluate_accuracy(test_loader, model)
            logger.info( 'dev_acc {:7.4f} | test_acc {:7.4f}'.format(dev_acc, test_acc))


if __name__ == '__main__':
    args = get_args(is_save=True)
    logger = get_logger(args)
    main(args)
