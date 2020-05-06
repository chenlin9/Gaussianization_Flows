import numpy as np
import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import pdb

hs_min = 1e-7
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
normal_dist = tdist.Normal(0, 1)

def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal

class rbig_block(nn.Module):
    def __init__(self, layer, dimension, datapoint_num, householder_iter=0, semi_learning=False, multidim_kernel=True,
                 usehouseholder=False, need_rotation=True):
        super().__init__()
        self.init = False
        if householder_iter == 0:
            self.householder_iter = dimension #min(dimension, 10)
        else:
            self.householder_iter = householder_iter
        self.layer = layer
        self.dimension = dimension
        self.datapoint_num = datapoint_num
        self.usehouseholder = usehouseholder
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


    def forward(self, inputs, process_size):
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

        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf_l <= 0.5e-7).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 1.
        cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)  # add mask to avoid sqrt(0)
        inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log)) * cdf_mask_left

        #############################################################################################
        hs = torch.exp(self.log_hs)
        hs = torch.max(hs, torch.ones_like(hs) * hs_min)
        log_hs = torch.max(self.log_hs, torch.ones_like(hs) * np.log(hs_min))
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

        if self.need_rotation:
            x = torch.mm(inverse_l, rotation_matrix)
        else:
            x = inverse_l

        # update cur_data
        with torch.no_grad():
            update_data_arrays = []
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
        return x, log_det, cur_datapoints_update


class Net(nn.Module):
    def __init__(self, datapoint_num, total_layer, dimension, kde_num, householder_iter=0,
                 multidim_kernel=True, usehouseholder=False):
        super().__init__()
        self.total_layer = total_layer
        self.layers = nn.ModuleList()

        for layer_num in range(total_layer):
            for i in range(kde_num-1):
                self.layers.append(rbig_block(layer_num, dimension, datapoint_num, householder_iter=householder_iter, multidim_kernel=multidim_kernel,
                                              usehouseholder=usehouseholder, need_rotation=False))
            self.layers.append(rbig_block(layer_num, dimension, datapoint_num, householder_iter=householder_iter, multidim_kernel=multidim_kernel,
                                          usehouseholder=usehouseholder))

    def forward(self, x, datapoints, process_size):
        # Input data x has shape batch_size, channel * image_size * image_size
        log_det = torch.zeros(x.shape[0], device=device)
        cur_datapoints = datapoints
        # cur_datapoints is only used for initialization
        for layer in self.layers:
            x, log_det, cur_datapoints = layer([x, log_det, cur_datapoints], process_size)
        return x, log_det, cur_datapoints


    def sampling(self, datapoints, x, process_size, sample_num=100):
        with torch.no_grad():
            print("Start sampling")
            datapoints_array = []
            cur_datapoints = datapoints
            datapoints_array.append(cur_datapoints)

            for i in range(sample_num // process_size):
                for l, layer in reversed(list(enumerate(self.layers))):
                    x[i * process_size: (i + 1) * process_size, :] = layer.sampling(
                        x[i * process_size: (i + 1) * process_size, :])
        return x

