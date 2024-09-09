import torch
import gc
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import DataParallel
import numpy as np

import datetime

from src.utils.metric_utils import evaluate_mrr
from src.utils.config_utils import read_config, dataset_from_config
from src.model.model import ANYCQ
from src.model.loss import reinforce_loss
from src.csp.cq_data import CQ_Data

from argparse import ArgumentParser
from tqdm import tqdm
import os


torch.multiprocessing.set_sharing_strategy('file_system')


def get_linear_scheduler():
    training_steps = config['epochs'] * len(train_loader)
    decay = config['lr_decay']
    lr_fn = lambda step: max(1.0 - ((1.0 - decay) * (step / training_steps)), decay)
    scheduler = LambdaLR(opt, lr_lambda=lr_fn)
    return scheduler


def save_opt_states(model_dir):
    torch.save(
        {
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        },
        os.path.join(model_dir, 'opt_state_dict.pt')
    )


def load_opt_states(model_dir):
    state_dicts = torch.load(os.path.join(model_dir, 'opt_state_dict.pt'))
    opt.load_state_dict(state_dicts['opt_state_dict'])
    scheduler.load_state_dict(state_dicts['sched_state_dict'])
    scaler.load_state_dict(state_dicts['scaler_state_dict'])
    return opt, scaler


def train_epoch():
    model.train()
    
    scores_list = []
    solved_list = []
    out_data = None
    loss = None

    for data in tqdm(train_loader, total=len(train_loader), disable=args.no_bar, desc=f'Training Epoch {epoch+1}'):

        opt.zero_grad()
        data.to(device)
        
        if device == 'cuda':
            with torch.cuda.amp.autocast():
                out_data = model(
                    data,
                    config['T_train'],
                    return_log_probs=True,
                    return_all_scores=True,
                    return_all_assignments=True
                )

                loss = reinforce_loss(out_data, config['loss_config'])
        else:
            out_data = model(
                data,
                config['T_train'],
                return_log_probs=True,
                return_all_scores=True,
                return_all_assignments=True
            )

            loss = reinforce_loss(out_data, config['loss_config'])

                

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        best_scores = out_data.best_cq_scores.cpu().view(-1)

        scores_list.append(best_scores)
        solved_list.append(best_scores > 0)

        if (model.global_step + 1) % args.checkpoint_steps == 0:
            with open('logs/'+exp_name+'.out', 'a') as f:
                f.write("\nSaving model after step "+str(model.global_step + 1)+"...\n")
            model.save_model(name=f'checkpoint_{model.global_step+1}')
            validate()
            model.train()

        model.global_step += 1


def validate():
    model.eval()

    dataset_config = config['val_data']

    with open('logs/'+exp_name+'.out', 'a') as f:
        f.write('Validating saved model:'+'\n')

    for i, dataset_name in enumerate(dataset_config["data_files"]):

        dataset_config['data_file'] = dataset_name
        dataset = dataset_from_config(dataset_config, device)

        indices = torch.randint(high=len(dataset), size=(config['val_batch_size'][i],))

        in_data = CQ_Data.collate([dataset[data_id.item()] for data_id in indices])
        in_data.to(device)

        cq_type = dataset.cq_types[0]
        with open('logs/'+exp_name+'.out', 'a') as f:
            f.write('-------------------\n'+cq_type+'\n')

        total_rec_rank_easy = 0
        total_rec_rank_hard = 0

        with torch.no_grad():
            out_data = model(
                in_data,
                config['T_val'],
                return_all_scores=True,
                return_all_assignments = True
            )

        scores = out_data.all_cq_scores.cpu()
        assgns = out_data.all_assignments.cpu()

        sorted_ids = [torch.unique_consecutive(assgns[in_data.pred_mask.cpu()][i][torch.argsort(scores[i], descending=True)]) for i in range(scores.shape[0])]

        correct_ans_easy = in_data.corr_ans_easy[0]
        correct_ans_hard = in_data.corr_ans_hard[0]
        
        correct_ans_easy = [torch.Tensor(answr).flatten().long() for answr in correct_ans_easy]
        correct_ans_hard = [torch.Tensor(answr).flatten().long() for answr in correct_ans_hard]

        rec_ranks_easy,	rec_ranks_hard = evaluate_mrr(sorted_ids, correct_ans_easy, correct_ans_hard)

        total_rec_rank_easy += rec_ranks_easy.mean()
        total_rec_rank_hard += rec_ranks_hard.mean()
                    
        with open('logs/'+exp_name+'.out', 'a') as f:
            f.write('Easy MRR (sample): '+ str((total_rec_rank_easy).item())+'\n')
            f.write('Hard MRR (sample): '+ str((total_rec_rank_hard).item())+'\n')

        in_data.to('cpu')
        out_data.to('cpu')
    
    model.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/comb/test', help="Model directory")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--checkpoint_steps", type=int, default=5000, help="Training steps between saving checkpoints")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--no_bar", action='store_true', default=False, help="Turn of tqdm bar")
    parser.add_argument("--no_PE", action='store_true', default=False, help="Disable PE labels")
    parser.add_argument("--from_last", action='store_true', default=False, help="Continue from existing last checkpoint")
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Pretrained Model directory")
    parser.add_argument("--config", type=str, help="path the config file")
    parser.add_argument("--exp_name", type=str, default="", help="name of the experiment")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.from_last:
        args.pretrained_dir = args.model_dir
        args.config = os.path.join(args.model_dir, 'config.json')

    config = read_config(args.config)

    if args.pretrained_dir is None:
        model = ANYCQ(args.model_dir, config, use_PE = False if args.no_PE else True)
    else:
        model = ANYCQ.load_model(args.pretrained_dir, f'last')
        model.model_dir = args.model_dir

    print('Using PE: ', model.use_PE)
    model.to(device)
    model.train()

    print("Using device: ", device)

    print("Creating dataset...")
    train_data = dataset_from_config(config['train_data'], device)
    print("Creating DataLoader...")
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        collate_fn=CQ_Data.collate,
        shuffle = True
    )

    print("Training data prepared!")

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_scheduler()
    scaler = torch.cuda.amp.GradScaler()

    if args.pretrained_dir is not None:
        opt, scaler = load_opt_states(args.pretrained_dir)

    logger = SummaryWriter(args.model_dir)
    best_unsat = np.float32('inf')
    best_solved = 0.0
    start_step = 0

    exp_name = args.exp_name+"".join(str(datetime.datetime.now()).split(' '))

    with open('logs/'+exp_name+'.out', 'w') as f:
        f.write('Starting training:\n\n')

    for epoch in range(config['epochs']):
        train_epoch()
        model.save_model(name='last')
        save_opt_states(model.model_dir)
        torch.cuda.empty_cache()