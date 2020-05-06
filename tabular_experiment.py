import os
import math
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.distributions as tdist
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
import datasets

from models.flow_model import Net
import pdb

# add device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info("Using device: {}".format(device))

seed = 52199
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate")
parser.add_argument('--batch_size', type=int, default=5000,
                    help="Batch size")
parser.add_argument('--total_datapoints', type=int, default=100,
                    help="Total number of data point for KDE")
parser.add_argument('--adjust_step', type=int, default=10000,
                    help="Decrease learning rate after a couple steps")
parser.add_argument('--epoch', type=int, default=1000,
                    help="Number of epochs")
parser.add_argument('--process_size', type=int, default=100,
                    help="Process size")
parser.add_argument('--layer', type=int, default=5,
                    help="Total number of Gaussianization layers")
parser.add_argument('--usehouseholder', action='store_true',
                    help='Train rotation matrix using householder reflection or not')
parser.add_argument('--multidim_kernel', action='store_true',
                    help='Use multi bandwidth kernel')
parser.add_argument('--test_interval', type=int, default=5,
                    help="Test interval")
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--kde_num', type=int, default=1, help='Stacking multiple KDE before each rotation layer')
parser.add_argument(
    '--dataset',
    default='POWER',
    help='POWER | GAS | HEPMASS | MINIBOONE | BSDS300 | MOONS')
args = parser.parse_args()


def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
    log_jacob = log_jacob.sum()
    loss = -(log_probs + log_jacob)

    if size_average:
        loss /= u.size(0)
    return loss


if __name__ == '__main__':
    process_size = args.process_size
    epoch = args.epoch
    total_layer = args.layer
    total_datapoints = args.total_datapoints
    print("Total layer {}, total KDE datapoints {}".format(total_layer, total_datapoints))

    best_loss = 1e10
    n_vals_without_improvement = 0
    cum = 0
    dataset = getattr(datasets, args.dataset)()

    dimension = dataset.n_dims
    train_tensor = torch.from_numpy(dataset.trn.x)
    val_tensor = torch.from_numpy(dataset.val.x)
    test_tensor = torch.from_numpy(dataset.tst.x)

    print("Loading dataset {}".format(args.dataset))
    net = Net(total_datapoints, total_layer, dimension, args.kde_num, multidim_kernel=args.multidim_kernel,
              usehouseholder=args.usehouseholder).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0., amsgrad=True)
    base_iter = batch_iter(train_tensor, batch_size=total_datapoints, shuffle=False)
    DATA = next(base_iter)
    DATA = DATA.to(device)
    DATA = DATA.view(DATA.shape[0], -1)
    step = 0

    for e in range(epoch):
        train_loss = 0
        total = 0
        for i, data in enumerate(batch_iter(train_tensor, batch_size=args.batch_size, shuffle=True)):
            net.train()
            if (step + 1) % args.adjust_step == 0:
                print("Adjusting learning rate, divide lr by 2")
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 2.
            step += 1
            total += data.shape[0]
            data = data.to(device)
            data = data.view(data.shape[0], -1)

            data, log_det, _ = net.forward(data, DATA, args.process_size)
            train_loss_r = flow_loss(data, log_det)
            optimizer.zero_grad()
            train_loss_r.backward()
            optimizer.step()

            train_loss += train_loss_r * data.shape[0]
            if i % 10 == 0:
                print("Epoch {} batch {} current loss {:.2f}({:.2f})".format(e, i,
                                                                train_loss_r.item(), train_loss.item()/total))
            if i >= 10: #remove this line!
                break
        print("Total training points {}. Epoch {} train loss {}".format(total, e, train_loss/total))

        net.eval()
        # Evaluate
        with torch.no_grad():
            val_loss = 0
            val_total = 0
            for val_data in batch_iter(val_tensor, shuffle=False):
                val_total += val_data.shape[0]
                val_data = val_data.to(device)
                val_data = val_data.view(val_data.shape[0], -1)
                val_data, val_log_det, _ = net.forward(val_data, DATA, args.process_size)
                val_loss_r = flow_loss(val_data, val_log_det)
                val_loss += val_loss_r * val_data.shape[0]

            if val_loss/val_total < best_loss:
                best_loss = val_loss/val_total
                # Save checkpoint
                print('Saving checkpoint...')
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                os.makedirs('ckpts', exist_ok=True)
                torch.save(state, 'ckpts/tabular_{}_{}.pth.tar'.format(args.dataset, seed))
                n_vals_without_improvement = 0
                cum = 0
            else:
                n_vals_without_improvement += 1
                cum += 1
            print("Val loss {} (val loss withput improvement {}/{}({}/{}))".format(val_loss/val_total,
                                                                            n_vals_without_improvement,
                                                                            args.early_stopping, cum, 100))
        if n_vals_without_improvement > args.early_stopping:
            print("Adjusting learning rate, divide lr by 2")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2.
            n_vals_without_improvement = 0

        # Test
        with torch.no_grad():
            test_total = 0
            test_loss = 0
            pbar = tqdm(total=len(test_tensor))
            pbar.set_description('Test')
            for batch_idx, data in enumerate(batch_iter(test_tensor, shuffle=False)):
                test_total += data.shape[0]
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                data, log_det, _ = net.forward(data, DATA, args.process_size)
                test_loss_r = flow_loss(data, log_det)
                test_loss += test_loss_r * data.shape[0]
                pbar.set_postfix(loss=test_loss.item() / test_total)
                pbar.update(data.shape[0])
            pbar.close()
            print("Now computing total loss. {} test datapoints".format(test_total))
            print("Epoch {} {} Total test loss {}".format(e, args.dataset, test_loss / test_total))

    # Test
    net.eval()
    with torch.no_grad():
        test_total = 0
        test_loss = 0
        checkpoint = torch.load('ckpts/tabular_{}_{}.pth.tar'.format(args.dataset, seed))
        net.load_state_dict(checkpoint['net'])
        pbar = tqdm(total=len(test_tensor))
        pbar.set_description('Test')
        for batch_idx, data in enumerate(batch_iter(test_tensor, shuffle=False)):
            test_total += data.shape[0]
            data = data.to(device)
            data = data.view(data.shape[0], -1)
            data, log_det, _ = net.forward(data, DATA, args.process_size)
            test_loss_r = flow_loss(data, log_det)
            test_loss += test_loss_r * data.shape[0]
            pbar.set_postfix(loss=test_loss.item() / test_total)
            pbar.update(data.shape[0])
        pbar.close()
        print("Now computing total loss. {} test datapoints".format(test_total))
        print("{} total test loss {}".format(args.dataset, test_loss / test_total))
