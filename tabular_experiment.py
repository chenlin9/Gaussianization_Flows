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

import pdb

hs_min = 1e-7
# add device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info("Using device: {}".format(device))

normal_dist = tdist.Normal(0, 1)
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
                    help="Decrease learning rate after steps")
parser.add_argument('--epoch', type=int, default=1000,
                    help="Number of epochs")
parser.add_argument('--process_size', type=int, default=100,
                    help="Process size")
parser.add_argument('--layer', type=int, default=5,
                    help="Total number of Gaussianization")
parser.add_argument('--kde_type', type=str, default='Logistic',
                    help="KDE type")
parser.add_argument('--rotation_type', type=str, default='PCA',
                    help="Rotation matrix type: PCA, random, ICA")
parser.add_argument('--usehouseholder', action='store_true',
                    help='Train rotation matrix using householder reflection or not')
parser.add_argument('--usepatch', action='store_true',
                    help='Train rotation matrix using patch method or not')
parser.add_argument('--multidim_kernel', action='store_true',
                    help='Use multi bandwidth kernel')
parser.add_argument('--test_interval', type=int, default=5,
                    help="Test interval")
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--kde_num', type=int, default=1)
parser.add_argument('--semi_learning', action='store_true',
                    help='Train rotation matrix using householder reflection or not')
parser.add_argument('--subset', type=float,  default=2000,
                    help='Subset tabular set only')

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


def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
    log_jacob = log_jacob.sum()
    loss = -(log_probs + log_jacob)

    if size_average:
        loss /= u.size(0)
    return loss

# Each layer each block
class rbig_block(nn.Module):
    def __init__(self, layer, dimension, datapoint_num, semi_learning=args.semi_learning, multidim_kernel=True,
                 usehouseholder=False, usepatch=False, need_rotation=True):
        super().__init__()
        self.init = False
        self.householder_iter = dimension #min(dimension, 10)
        self.layer = layer
        self.dimension = dimension
        self.datapoint_num = datapoint_num
        self.usehouseholder = usehouseholder
        self.usepatch = usepatch
        self.need_rotation = need_rotation
        bandwidth = (4. * np.sqrt(math.pi) / ((math.pi ** 4) * datapoint_num)) ** 0.2

        if not semi_learning:
            self.datapoints = nn.Parameter(torch.randn(datapoint_num, self.dimension))
        else:
            self.datapoints = torch.randn(datapoint_num, self.dimension).to(device)

        self.kde_weights = torch.zeros(datapoint_num, self.dimension).to(device)
        if multidim_kernel:
            self.log_hs = nn.Parameter(
                torch.ones(datapoint_num, dimension) * np.log(bandwidth)
            )
        else:
            self.log_hs = nn.Parameter(
                torch.ones(1, dimension) * np.log(bandwidth)
            )

        if usehouseholder:
            self.vs = nn.Parameter(
                torch.randn(self.householder_iter, dimension)
            )
        else:
            self.register_buffer('matrix', torch.ones(dimension, dimension))

    def logistic_kernel_log_cdf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)

        log_cdfs = - F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                   self.kde_weights[:, None, :] - torch.logsumexp(self.kde_weights, dim=0, keepdim=True)[:, None, :]
        log_cdf = torch.logsumexp(log_cdfs, dim=0)
        return log_cdf

    def logistic_kernel_log_sf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)

        log_sfs = -(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :] - \
                  F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                  self.kde_weights[:, None, :] - torch.logsumexp(self.kde_weights, dim=0, keepdim=True)[:, None, :]
        log_sf = torch.logsumexp(log_sfs, dim=0)
        return log_sf

    def logistic_kernel_pdf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        log_hs = torch.max(self.log_hs, torch.ones_like(hs) * np.log(hs_min))

        log_pdfs = -(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :] - log_hs[:, None, :] - \
                   2. * F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                   self.kde_weights[:, None, :] - torch.logsumexp(self.kde_weights, dim=0, keepdim=True)[:, None, :]
        log_pdf = torch.logsumexp(log_pdfs, dim=0)
        pdf = torch.exp(log_pdf)
        return pdf

    def logistic_kernel_cdf(self, x, datapoints):
        # Using bandwidth formula
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)

        log_cdfs = - F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) \
                   + self.kde_weights[:, None, :] - torch.logsumexp(self.kde_weights, dim=0, keepdim=True)[:, None, :]
        log_cdf = torch.logsumexp(log_cdfs, dim=0)
        cdf = torch.exp(log_cdf)
        return cdf

    def compute_householder_matrix(self):
        Q = torch.eye(self.dimension, device=device)
        for i in range(self.householder_iter):
            v = self.vs[i].reshape(-1, 1)
            v = v / v.norm()
            Qi = torch.eye(self.dimension, device=device) - 2 * torch.mm(v, v.permute(1, 0))
            Q = torch.mm(Q, Qi)
        return Q

    # compute inverse normal CDF
    def inverse_normal_cdf(self, x):
        mask_bound = 0.5e-7
        datapoints = self.datapoints

        cdf_l = self.logistic_kernel_cdf(x, datapoints)
        log_cdf_l = self.logistic_kernel_log_cdf(x, datapoints)  # log(CDF)
        log_sf_l = self.logistic_kernel_log_sf(x, datapoints)  # log(1-CDF)

        # Approximate Gaussian CDF
        # inv(CDF) ~ np.sqrt(-2 * np.log(1-x)) #right, x -> 1
        # inv(CDF) ~ -np.sqrt(-2 * np.log(x)) #left, x -> 0
        # 1) Step1: invert good CDF
        cdf_mask = ((cdf_l > mask_bound) & (cdf_l < 1 - (mask_bound))).float()
        # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
        cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
        inverse_l = normal_dist.icdf(cdf_l_good)

        # 2) Step2: invert BAD large CDF
        cdf_mask_right = (cdf_l >= 1. - (mask_bound)).float()
        # Keep large bad CDF, mask the good and small bad CDF values to 0.
        cdf_l_bad_right_log = log_sf_l * cdf_mask_right
        inverse_l += torch.sqrt(-2. * cdf_l_bad_right_log)

        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf_l <= mask_bound).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 1.
        cdf_l_bad_left_log = log_cdf_l * cdf_mask_left
        inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log))
        return inverse_l

    def sampling(self, z, verbose=False, lower=-1e3, upper=1e3):  # lower=-1e5, upper=1e5
        if self.usepatch:  # need to change the pattern back
            self.matrix = self.patch_matrix()
        elif self.usehouseholder:
            self.matrix = self.compute_householder_matrix()

        if self.need_rotation:
            if self.usehouseholder:
                z = torch.mm(z, self.matrix.permute(1, 0))  # uncomment

            if self.usepatch:
                batch_size = z.shape[0]
                z = z.reshape(batch_size * 49, 1, 16)
                matrix = self.matrix.unsqueeze(dim=0).expand(batch_size, self.matrix.shape[0],
                                                             self.matrix.shape[1], self.matrix.shape[2])
                matrix = matrix.permute(0, 1, 3, 2)  # take transpose
                matrix = matrix.reshape(z.shape[0], matrix.shape[-2], matrix.shape[-1])  # batch_size*49, 16, 16
                z = torch.matmul(z, matrix)  # patch z: batch_size*49, 1, 16
                z = z.reshape(batch_size, -1)  # patch s: batch, 28*28

        iteration = int(np.log2(upper - lower) + np.log2(1e6))
        upper = torch.tensor(upper).repeat(*z.shape).to(device)
        lower = torch.tensor(lower).repeat(*z.shape).to(device)
        for i in range(iteration):
            mid = (upper + lower) / 2.
            inverse_mid = self.inverse_normal_cdf(mid)
            right_part = (inverse_mid < z).float()
            left_part = 1. - right_part

            correct_part = (close(inverse_mid, z, rtol=1e-6, atol=0)).float()
            lower = (1. - correct_part) * (right_part * mid + left_part * lower) + correct_part * mid
            upper = (1. - correct_part) * (right_part * upper + left_part * mid) + correct_part * mid

        if verbose:
            print("Average error {}".format(torch.sum(upper - lower) / np.prod(x.shape)))
        # print(mid.max(), mid.min())
        return mid


    def forward(self, inputs, process_size=args.process_size):
        # Parameters:
        # x: inpute data, with shape (batch_size, dimension)
        # log_det: accumulated log_det
        # cur_datapoints: the datapoints used to fit KDE
        # process_size: batch size for generating new datapoints
        [x, log_det, cur_datapoints] = inputs

        if not self.usehouseholder:
            if not self.init:
                self.datapoints.data = cur_datapoints
                self.init = True

                if self.need_rotation:
                    self.matrix, _, _ = torch.svd(
                        torch.mm(cur_datapoints.permute(1, 0), cur_datapoints))
                    rotation_matrix = self.matrix
            else:
                if self.need_rotation:
                    rotation_matrix = self.matrix
        else:
            if not self.init:
                self.datapoints.data = cur_datapoints
                self.init = True
            if self.need_rotation:
                rotation_matrix = self.compute_householder_matrix()

        # self.datapoints = self.c * cur_datapoints + self.delta
        total_datapoints = self.datapoints.shape[0]
        #############################################################################################
        # Compute inverse CDF
        #############################################################################################

        cdf_l = self.logistic_kernel_cdf(x, self.datapoints)
        log_cdf_l = self.logistic_kernel_log_cdf(x, self.datapoints)  # log(CDF)
        log_sf_l = self.logistic_kernel_log_sf(x, self.datapoints)  # log(1-CDF)

        cdf_mask = ((cdf_l > 0.5e-7) & (cdf_l < 1 - (0.5e-7))).float()
        # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
        cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
        inverse_l = normal_dist.icdf(cdf_l_good)

        # with torch.no_grad():  # if remove this line, gradient is nan
        # 2) Step2: invert BAD large CDF
        cdf_mask_right = (cdf_l >= 1. - (0.5e-7)).float()
        # Keep large bad CDF, mask the good and small bad CDF values to 0.
        cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
        inverse_l += torch.sqrt(-2. * cdf_l_bad_right_log) * cdf_mask_right
        # if (inverse_l != inverse_l).any():
        #     breakpoint()

        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf_l <= 0.5e-7).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 1.
        cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)  # add mask to avoid sqrt(0)
        inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log)) * cdf_mask_left

        #############################################################################################
        ######################################
        # Modified log_det
        # Using bandwidth formula
        ######################################
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        log_hs = torch.max(self.log_hs, torch.ones_like(hs) * np.log(hs_min))
        # n = self.datapoints.shape[0]
        # log_pdfs = -(x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :] - self.log_hs[:, None, :] - \
        #            2. * F.softplus(-(x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :]) - np.log(n)
        log_pdfs = -(x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :] - log_hs[:, None, :] - \
                   2. * F.softplus(-(x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :]) + \
                   self.kde_weights[:, None, :] - torch.logsumexp(self.kde_weights, dim=0, keepdim=True)[:, None, :]

        log_pdf = torch.logsumexp(log_pdfs, dim=0)
        log_gaussian_derivative_good = tdist.Normal(0, 1).log_prob(inverse_l) * cdf_mask
        log_gaussian_derivative_left = (torch.log(torch.sqrt(-2 * cdf_l_bad_left_log))
                                        - log_cdf_l) * cdf_mask_left
        log_gaussian_derivative_right = (torch.log(torch.sqrt(-2. * cdf_l_bad_right_log))
                                         - log_sf_l) * cdf_mask_right
        log_gaussian_derivative = log_gaussian_derivative_good + log_gaussian_derivative_left + log_gaussian_derivative_right
        log_det += (log_pdf - log_gaussian_derivative).sum(dim=-1)  # only keep batch size
        ######################################
        # End of changes
        ######################################
        if self.need_rotation:
            x = torch.mm(inverse_l, rotation_matrix)
        else:
            x = inverse_l

        # update cur_data
        with torch.no_grad(): #MAYBE REMOVE THIS NEED TO DOUBLE CHECK THIS (RETAIN GRAPH?)
            #Since this has bug in image version
            update_data_arrays = []
            if total_datapoints % process_size != 0:
                pdb.set_trace()

            assert (total_datapoints % process_size == 0), "Process_size does not divide total_datapoints!"
            for b in range(total_datapoints // process_size):
                # return x, log_det, None  #remove this line when layer > 1
                # if b % 20 == 0 and b>0:
                #     print("Generating new datapoints: {0}/{1}".format(b, total_datapoints // process_size))
                cur_data_batch = self.datapoints[process_size * b: process_size * (b + 1), :]
                cdf_data = self.logistic_kernel_cdf(cur_data_batch, self.datapoints)
                log_cdf_data = self.logistic_kernel_log_cdf(cur_data_batch, self.datapoints)
                log_sf_data = self.logistic_kernel_log_sf(cur_data_batch, self.datapoints)

                # 1) Step1: invert good CDF
                cdf_mask_data = ((cdf_data > 0.5e-7) & (cdf_data < 1. - (0.5e-7))).float()
                # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
                cdf_data_good = cdf_data * cdf_mask_data + 0.5 * (1. - cdf_mask_data)
                inverse_data = normal_dist.icdf(cdf_data_good)
                # 2) Step2: invert BAD large CDF
                cdf_mask_right_data = (cdf_data >= 1. - (0.5e-7)).float()
                # keep large bad CDF, mask the good and small bad cdf values to 0.
                cdf_data_bad_right_log = log_sf_data * cdf_mask_right_data
                inverse_data += torch.sqrt(-2. * cdf_data_bad_right_log)

                # 3) Step3: invert BAD small CDF
                cdf_mask_left_data = (cdf_data <= 0.5e-7).float()
                # keep small bad CDF, mask the good and large bad CDF values to 1.
                cdf_data_bad_left_log = log_cdf_data * cdf_mask_left_data
                inverse_data += (-torch.sqrt(-2 * cdf_data_bad_left_log))
                if self.need_rotation:
                    cur_data_batch = torch.mm(inverse_data, rotation_matrix.data)
                else:
                    cur_data_batch = inverse_data
                update_data_arrays.append(cur_data_batch)
            cur_datapoints_update = torch.cat(update_data_arrays, dim=0)
            # print(self.log_hs)
        return x, log_det, cur_datapoints_update


# Trainable model
class Net(nn.Module):
    def __init__(self, datapoint_num, total_layer, dimension, kde_num=args.kde_num, usepatch=args.usepatch,
                 multidim_kernel=True, usehouseholder=False):
        super().__init__()
        self.total_layer = total_layer
        self.layers = nn.ModuleList()
        self.usepatch = usepatch

        for layer_num in range(total_layer):
            for i in range(kde_num-1):
                self.layers.append(rbig_block(layer_num, dimension, datapoint_num, multidim_kernel=multidim_kernel,
                                              usehouseholder=usehouseholder, need_rotation=False))
            self.layers.append(rbig_block(layer_num, dimension, datapoint_num, multidim_kernel=multidim_kernel,
                                          usehouseholder=usehouseholder))

    def forward(self, x, datapoints):
        # Input data x has shape batch_size, channel * image_size * image_size
        log_det = torch.zeros(x.shape[0], device=device)
        cur_datapoints = datapoints
        # cur_datapoints is only used for initialization
        for layer in self.layers:
            x, log_det, cur_datapoints = layer([x, log_det, cur_datapoints])
        return x, log_det, cur_datapoints


if __name__ == '__main__':
    process_size = args.process_size
    epoch = args.epoch
    total_layer = args.layer
    kde_type = args.kde_type
    rotation_type = args.rotation_type
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
    net = Net(total_datapoints, total_layer, dimension, multidim_kernel=args.multidim_kernel,
              usehouseholder=args.usehouseholder).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0., amsgrad=True)

    base_iter = batch_iter(train_tensor, batch_size=total_datapoints, shuffle=False)
    DATA = next(base_iter)
    DATA = DATA.to(device)
    DATA = DATA.view(DATA.shape[0], -1)
    step = 0

    for e in range(epoch):
        train_bpd = 0
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

            data, log_det, _ = net.forward(data, DATA)
            train_loss_r = flow_loss(data, log_det)
            optimizer.zero_grad()
            train_loss_r.backward()
            optimizer.step()

            train_bpd_r = (train_loss_r.item() * data.shape[0]) * (1 / (np.log(2) * np.prod(data.shape)))
            train_bpd += train_bpd_r * data.shape[0]
            train_loss += train_loss_r * data.shape[0]
            if i % 10 == 0:
                print("Epoch {} batch {} current loss {:.2f}({:.2f})".format(e, i,
                                                                train_loss_r.item(), train_loss.item()/total))

        print("Total training points {}. Epoch {} train loss {} bpd {}".format(total, e, train_loss/total, train_bpd/total))

        net.eval()
        with torch.no_grad():
            val_loss = 0
            val_total = 0
            for val_data in batch_iter(val_tensor, shuffle=False):
                val_total += val_data.shape[0]
                val_data = val_data.to(device)
                val_data = val_data.view(val_data.shape[0], -1)

                val_data, val_log_det, _ = net.forward(val_data, DATA)
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

        if cum > 100: #remove this line
            break

        if n_vals_without_improvement > args.early_stopping:
            print("Adjusting learning rate, divide lr by 2")
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/2.
            n_vals_without_improvement = 0

        with torch.no_grad():
            test_total = 0
            test_bpd = 0
            test_loss = 0

            pbar = tqdm(total=len(test_tensor))
            pbar.set_description('Test')
            for batch_idx, data in enumerate(batch_iter(test_tensor, shuffle=False)):
                test_total += data.shape[0]
                data = data.to(device)
                data = data.view(data.shape[0], -1)

                data, log_det, _ = net.forward(data, DATA)
                test_loss_r = flow_loss(data, log_det)
                test_bpd_r = (test_loss_r.item() * data.shape[0]) * (1 / (np.log(2) * np.prod(data.shape)))
                test_bpd += test_bpd_r * data.shape[0]
                test_loss += test_loss_r * data.shape[0]
                pbar.set_postfix(loss=test_loss.item() / test_total)
                pbar.update(data.shape[0])

            pbar.close()
            print("Now computing total loss. {} test datapoints".format(test_total))
            print("Epoch {} {} Total test loss {} bpd {}".format(e, args.dataset, test_loss / test_total,
                                                                 test_bpd / test_total))

    # Test
    with torch.no_grad():
        test_total = 0
        test_bpd = 0
        test_loss = 0

        checkpoint = torch.load('ckpts/tabular_{}_{}.pth.tar'.format(args.dataset, seed))
        net.load_state_dict(checkpoint['net'])
        pbar = tqdm(total=len(test_tensor))
        pbar.set_description('Test')
        for batch_idx, data in enumerate(batch_iter(test_tensor, shuffle=False)):
            test_total += data.shape[0]
            data = data.to(device)
            data = data.view(data.shape[0], -1)

            data, log_det, _ = net.forward(data, DATA)
            test_loss_r = flow_loss(data, log_det)
            test_bpd_r = (test_loss_r.item() * data.shape[0]) * (1 / (np.log(2) * np.prod(data.shape)))
            test_bpd += test_bpd_r * data.shape[0]
            test_loss += test_loss_r * data.shape[0]
            pbar.set_postfix(loss=test_loss.item() / test_total)
            pbar.update(data.shape[0])
        pbar.close()
        print("Now computing total loss. {} test datapoints".format(test_total))
        print("{} total test loss {} bpd {}".format(args.dataset, test_loss / test_total, test_bpd / test_total))
