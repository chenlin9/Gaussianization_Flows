import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid

import os
import math
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.distributions as tdist
import seaborn as sns
from scipy.stats import kde
from toy_distribution import *

sns.set()
import pdb

# add device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info("Using device: {}".format(device))


normal_dist = tdist.Normal(0, 1)

# set random seed
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate")
parser.add_argument('--householder_iter', type=int, default=2,
                    help="Number of householder iteration")
parser.add_argument('--adjust_step', type=int, default=1000,
                    help="Decrease learning rate after steps")
parser.add_argument('--batch_size', type=int, default=10000,
                    help="Batch size")
parser.add_argument('--total_datapoints', type=int, default=50,
                    help="Total number of data point for KDE")
parser.add_argument('--epoch', type=int, default=5000,
                    help="Number of epochs")
parser.add_argument('--process_size', type=int, default=10000,
                    help="Process size")
parser.add_argument('--total_layer', type=int, default=20,
                    help="Total number of Gaussianization")
parser.add_argument('--kde_type', type=str, default='Logistic',
                    help="KDE type")
parser.add_argument('--test_interval', type=int, default=5,
                    help="Test interval")
parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--dataset', default='swissroll', help='swissroll | sinewave | tenbyten | circles | rings | 8gaussians | pinwheel | checkerboard | line')
parser.add_argument('--test', action='store_true', help='Test mode')
parser.add_argument('--train', action='store_true', help='Train mode')
parser.add_argument('--checkpoint_name', type=str, default="toy", help='checkpoint_name')
parser.add_argument('--kde_num', type=int, default=1, help="multiple kde layers for Gaussianization")
args = parser.parse_args()

print("Using dataset {}".format(args.dataset))

sample_data = inf_train_gen(args.dataset, batch_size=args.batch_size)
MAX_X = sample_data[:,0].max()
MIN_X = sample_data[:,0].min()
MAX_Y = sample_data[:,1].max()
MIN_Y = sample_data[:,1].min()


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



def plot_density(net, DATA, image_name="trained_{}_model_density.png".format(args.dataset), npts=200):
    with torch.no_grad():
        plt.figure(figsize=(6,6))

        xside = np.linspace(MIN_X, MAX_X, npts)
        yside = np.linspace(MIN_Y, MAX_Y, npts)

        xx, yy = np.meshgrid(xside, yside)
        z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        z = torch.Tensor(z).to(device).float()
        data, log_det, _ = net.forward(z, DATA)

        log_probs = (-0.5 * data.pow(2) - 0.5 * np.log(2 * np.pi)).sum(dim=-1)
        log_jacob = log_det
        loss = -(log_probs + log_jacob).cpu().numpy().reshape(npts, npts)
        p = np.exp(-loss)
        print(p.max())
        plt.pcolormesh(xx, yy, p, cmap='viridis', vmin=0.0, vmax=47.0)
        dict = {}
        dict["args"] = args
        dict["xx"] = xx
        dict["yy"] = yy
        dict["p"] = p
        with open("toy_ckpt/{}.pkl".format(args.dataset), 'wb') as file:
            pickle.dump(dict, file, protocol=2)

        plt.xticks([]), plt.yticks([])
        plt.savefig("toy_dataset/{}".format(image_name), bbox_inches='tight', pad_inches=0, dpi=300)
        print("Density map generated! Written to image {}".format(image_name))
        plt.close()
    print("dataset density generated")



hs_min = 1e-8
weights_min = np.log(1e-30)
# Each layer each block
class rbig_block(nn.Module):
    def __init__(self, layer, dimension, datapoint_num, householder_iter=args.householder_iter,
                 multidim_kernel=True, usehouseholder=False, need_rotation=True):
        super().__init__()
        if usehouseholder:
            print("Use householder rotation")

        self.init = False
        self.layer = layer
        self.dimension = dimension
        self.datapoint_num = datapoint_num
        self.usehouseholder = usehouseholder
        self.householder_iter = datapoint_num
        self.need_rotation = need_rotation

        bandwidth = (4. * np.sqrt(math.pi) / ((math.pi ** 4) * datapoint_num)) ** 0.2
        self.delta = nn.Parameter(torch.zeros(datapoint_num, self.dimension))
        self.register_buffer('datapoints', None)
        self.kde_weights = torch.zeros(datapoint_num, self.dimension).to(device)
        if multidim_kernel:
            self.log_hs = nn.Parameter(
                torch.ones(datapoint_num, self.dimension) * np.log(bandwidth)
            )
        else:
            self.log_hs = nn.Parameter(
                torch.ones(1, self.dimension) * np.log(bandwidth)
            )

        if usehouseholder:
            self.vs = nn.Parameter(
                torch.randn(self.householder_iter, self.dimension)
            )
        else:
            self.register_buffer('matrix', torch.ones(self.dimension, self.dimension))


    def logistic_kernel_log_cdf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        # added mask
        kde_weights = self.kde_weights
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)

        log_cdfs = - F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                   kde_weights[:, None, :] - torch.logsumexp(kde_weights, dim=0, keepdim=True)[:, None, :]
        log_cdf = torch.logsumexp(log_cdfs, dim=0)
        return log_cdf



    def logistic_kernel_log_sf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        # added mask
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        kde_weights = self.kde_weights

        log_sfs = -(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :] - \
                  F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                  kde_weights[:, None, :] - torch.logsumexp(kde_weights, dim=0, keepdim=True)[:, None, :]
        log_sf = torch.logsumexp(log_sfs, dim=0)
        return log_sf

    def logistic_kernel_pdf(self, x, datapoints):
        hs = torch.exp(self.log_hs)
        # added mask
        log_hs = torch.max(self.log_hs, torch.ones_like(hs) * np.log(hs_min))
        kde_weights = self.kde_weights

        hs = torch.max(hs, torch.ones_like(hs) * hs_min)

        log_pdfs = -(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :] - log_hs[:, None, :] - \
                   2. * F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                   kde_weights[:, None, :] - torch.logsumexp(kde_weights, dim=0, keepdim=True)[:, None, :]
        log_pdf = torch.logsumexp(log_pdfs, dim=0)
        pdf = torch.exp(log_pdf)
        return pdf

    def logistic_kernel_cdf(self, x, datapoints):
        # Using bandwidth formula
        hs = torch.exp(self.log_hs)
        # added mask
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        kde_weights = self.kde_weights

        log_cdfs = - F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) \
                   + kde_weights[:, None, :] - torch.logsumexp(kde_weights, dim=0, keepdim=True)[:, None, :]
        log_cdf = torch.logsumexp(log_cdfs, dim=0)
        cdf = torch.exp(log_cdf)
        return cdf


    # self.vs: torch.randn(householder_iter, dimension)
    def compute_householder_matrix(self):
        Q = torch.eye(self.dimension, device=device)
        # print(self.vs[0])
        for i in range(self.householder_iter):
            v = self.vs[i].reshape(-1, 1)
            v = v / v.norm()
            Qi = torch.eye(self.dimension, device=device) - 2 * torch.mm(v, v.permute(1, 0))
            Q = torch.mm(Q, Qi)
        return Q


    # compute inverse normal CDF
    def inverse_normal_cdf(self, x):
        mask_bound = 0.5e-7
        datapoints = self.datapoints + self.delta

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
        # print(torch.sum(cdf_mask_right))
        cdf_l_bad_right_log = log_sf_l * cdf_mask_right
        inverse_l += torch.sqrt(-2. * cdf_l_bad_right_log)

        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf_l <= mask_bound).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 1.
        # print(torch.sum(cdf_mask_left))
        cdf_l_bad_left_log = log_cdf_l * cdf_mask_left
        inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log))
        return inverse_l

    def sampling(self, z, verbose=False, lower=-1e3, upper=1e3): #lower=-1e5, upper=1e5
        if self.usehouseholder:
            self.matrix = self.compute_householder_matrix()

        if self.need_rotation:
            if self.usehouseholder:
                z = torch.mm(z, self.matrix.permute(1, 0))  # uncomment

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
                self.init = True
                self.matrix, _, _ = torch.svd(
                    torch.mm(cur_datapoints.permute(1, 0), cur_datapoints))
                rotation_matrix = self.matrix
            else:
                rotation_matrix = self.matrix

        elif self.usehouseholder:
            rotation_matrix = self.compute_householder_matrix()


        self.datapoints = (cur_datapoints).to(device)
        datapoints = self.datapoints + self.delta
        #############################################################################################
        # Compute inverse CDF
        #############################################################################################
        cdf_l = self.logistic_kernel_cdf(x, datapoints)
        log_cdf_l = self.logistic_kernel_log_cdf(x, datapoints)  # log(CDF)
        log_sf_l = self.logistic_kernel_log_sf(x, datapoints)  # log(1-CDF)

        cdf_mask = ((cdf_l > 0.5e-7) & (cdf_l < 1 - (0.5e-7))).float()
        # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
        cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
        inverse_l = normal_dist.icdf(cdf_l_good)

        # 2) Step2: invert BAD large CDF
        cdf_mask_right = (cdf_l >= 1. - (0.5e-7)).float()
        # Keep large bad CDF, mask the good and small bad CDF values to 0.
        cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
        inverse_l += torch.sqrt(-2. * cdf_l_bad_right_log) * cdf_mask_right


        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf_l <= 0.5e-7).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 1.

        cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)  # add mask to avoid sqrt(0)
        inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log)) * cdf_mask_left

        #############################################################################################
        # log_det += self.compute_log_det(x, inverse_l, self.datapoints) #original one
        ######################################
        # Modified log_det
        # Using bandwidth formula
        ######################################
        # n = self.datapoints.shape[0]
        hs = torch.exp(self.log_hs)
        # added mask
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        log_hs = torch.max(self.log_hs, torch.ones_like(hs) * np.log(hs_min))

        ## logistic
        kde_weights = self.kde_weights #torch.max(self.kde_weights, torch.ones_like(self.kde_weights) * weights_min)
        log_pdfs = -(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :] - log_hs[:, None, :] - \
                   2. * F.softplus(-(x[None, ...] - datapoints[:, None, :]) / hs[:, None, :]) + \
                   kde_weights[:, None, :] - torch.logsumexp(kde_weights, dim=0, keepdim=True)[:, None, :]
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
            if self.usehouseholder:
                x = torch.mm(inverse_l, rotation_matrix)  # uncomment householder
        else:
            x = inverse_l  # remove

        # update cur_data
        with torch.no_grad():  # MAYBE REMOVE THIS
            inverse_data = self.inverse_normal_cdf(datapoints)

            if self.need_rotation:
                if self.usehouseholder:
                    cur_data_batch = torch.mm(inverse_data, rotation_matrix.data)  # remove this line householder
            else:
                cur_data_batch = inverse_data

        return x, log_det, cur_data_batch


# Training end to end
class Net(nn.Module):
    def __init__(self, datapoint_num, total_layer, dimension,
                 multidim_kernel=True, usehouseholder=False):
        super().__init__()
        self.total_layer = total_layer
        self.layers = nn.ModuleList()

        for layer_num in range(total_layer):
            self.layers.append(
                rbig_block(layer_num, dimension, datapoint_num, multidim_kernel=multidim_kernel,
                           usehouseholder=usehouseholder)) #change it back


    def forward(self, x, datapoints):
        # Input data x has shape batch_size, dimension
        log_det = torch.zeros(x.shape[0], device=device)
        cur_datapoints = datapoints

        for index, layer in enumerate(self.layers):
            x, log_det, cur_datapoints = layer([x, log_det, cur_datapoints])
        return x, log_det, cur_datapoints


def train():
    best_loss = 1e10
    n_vals_without_improvement = 0

    print("Total layer {}".format(total_layer))
    net = Net(total_datapoints, total_layer, dimension, multidim_kernel=True,
              usehouseholder=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    DATA = torch.tensor(inf_train_gen(args.dataset, batch_size=total_datapoints)).float().to(device)

    step = 0
    for e in range(args.epoch):
        train_loss = 0
        total = 0
        data = torch.tensor(inf_train_gen(args.dataset, batch_size=batch_size)).float().to(device)
        step += 1
        if (step + 1) % args.adjust_step == 0:
            print("Adjusting learning rate, divide lr by 2")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2.

        total += data.shape[0]
        data, log_det, _ = net.forward(data, DATA)
        train_loss_r = flow_loss(data, log_det)

        ###################################
        # added sampling
        ###################################
        if e % 10 == 0:
            plot_density(net, DATA)

        optimizer.zero_grad()
        train_loss_r.backward()
        optimizer.step()

        train_loss += train_loss_r * data.shape[0]
        if e % 10 == 0:
            print("{} Epoch [{}/{}] loss {}".format(dataset, e, args.epoch, train_loss_r))


        if train_loss / total < best_loss:
            best_loss = train_loss / total
            n_vals_without_improvement = 0
            # Save checkpoint
            print('Saving checkpoint...')
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'data': DATA,
                'total_layer': args.total_layer,
            }
            torch.save(state, 'toy_ckpt/{}_{}.pth.tar'.format(checkpoint_name, dataset))
        else:
            n_vals_without_improvement += 1

        print("Epoch {} Total train loss {} (val loss withput improvement {}/{})".format(e, train_loss / total,
                                                                                        n_vals_without_improvement,
                                                                                        args.early_stopping))

        if n_vals_without_improvement > args.early_stopping:
            print("Adjusting learning rate, divide lr by 2")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2.
            n_vals_without_improvement = 0

def test():
    print("Total layer {}".format(total_layer))
    net = Net(total_datapoints, total_layer, dimension, multidim_kernel=True,
              usehouseholder=True).to(device)

    checkpoint = torch.load('toy_ckpt/{}_{}.pth.tar'.format(checkpoint_name, dataset))
    DATA = checkpoint['data']
    print(DATA.shape)
    data = torch.tensor(inf_train_gen(args.dataset, batch_size=batch_size)).float().to(device)
    net.forward(data, DATA)
    net.load_state_dict(checkpoint['net'])
    plot_density(net, DATA)


if __name__ == '__main__':
    dimension = 2
    dataset = args.dataset
    batch_size = args.batch_size
    epoch = args.epoch
    total_layer = args.total_layer
    lr = args.lr
    process_size = args.process_size
    checkpoint_name = args.checkpoint_name
    total_datapoints = args.total_datapoints
    print("Start training")
    train()
    test()
