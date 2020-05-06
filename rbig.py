import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from utils.rbig_util import *

device = 'cuda'

# fix random seed
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    dataset = 'MNIST'
    total_datapoints = 10000
    process_size = 100
    layer = 10

    rotation_type = "PCA"
    print("Loading dataset {}".format(dataset))
    normal_distribution = tdist.Normal(0, 1)

    if dataset == 'MNIST':
        channel = 1
        image_size = 28
        lambd = 1e-5
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        train_dataset = MNIST(os.path.join('datasets', 'mnist'), train=True, download=True,
                              transform=transform)
        test_dataset = MNIST(os.path.join('datasets', 'mnist'), train=False, download=True,
                             transform=transform)
    elif dataset == 'FMNIST':
        channel = 1
        image_size = 28
        lambd = 1e-6
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        train_dataset = FashionMNIST(os.path.join('datasets', 'fmnist'), train=True, download=True,
                              transform=transform)
        test_dataset = FashionMNIST(os.path.join('datasets', 'fmnist'), train=False, download=True,
                             transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=total_datapoints, shuffle=True,
                              num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True,
                             num_workers=4, drop_last=False)

    train_iter = iter(train_loader)
    DATA, _ = next(train_iter)
    DATA = DATA.to(device) * 255. / 256.
    DATA += torch.rand_like(DATA) / 256.
    DATA = logit_transform(DATA, lambd)
    DATA = DATA.view(DATA.shape[0], -1)

    test_bpd = 0
    test_loss = 0
    total = 0
    print("Total layer {}".format(layer))
    rotation_matrices = []
    with torch.no_grad():
        data_anchors = [DATA]
        bandwidths = []
        vectors = []
        for batch_idx, (data, _) in enumerate(test_loader):
            total += data.shape[0]
            data = data.to(device) * 255. / 256.
            data += torch.rand_like(data) / 256.
            data = logit_transform(data, lambd)
            log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                data.shape) * np.log(1 - 2 * lambd)
            data = data.view(data.shape[0], -1)
            log_det = torch.zeros(data.shape[0]).to(device)

            # Pass the data through the first l-1 gaussian layer
            for prev_l in range(layer):
                # initialize rotation matrix
                if batch_idx == 0:
                    bandwidth = generate_bandwidth(data_anchors[prev_l])
                    bandwidths.append(bandwidth)
                    rotation_matrix = generate_orthogonal_matrix(data_anchors[prev_l], type=rotation_type)
                    rotation_matrices.append(rotation_matrix.to(device))
                    vector = 2 * (torch.rand((1, data.shape[-1])) - 0.5).to(device)
                    vectors.append(vector)

                inverse_l, cdf_mask, [log_cdf_l, cdf_mask_left], [log_sf_l, cdf_mask_right] \
                    = logistic_inverse_normal_cdf(data, bandwidth=bandwidth, datapoints=data_anchors[prev_l])
                log_det += compute_log_det(data, inverse_l, data_anchors[prev_l], cdf_mask,
                                                 log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right, h=bandwidth)
                data = torch.mm(inverse_l, rotation_matrices[prev_l])

                # Update cur_data
                if batch_idx == 0:
                    cur_data = data_anchors[prev_l]
                    update_data_arrays = []
                    assert (total_datapoints % process_size == 0), "Process_size does not divide total_datapoints!"
                    for b in range(total_datapoints // process_size):
                        if b % 20 == 0:
                            print("Layer {0} generating new datapoints: {1}/{2}".format(prev_l, b,
                                                                                        total_datapoints // process_size))
                        cur_data_batch = cur_data[process_size * b: process_size * (b + 1), :]
                        inverse_data, _, _, _ = logistic_inverse_normal_cdf(cur_data_batch, bandwidth=bandwidth,
                                                                         datapoints=data_anchors[prev_l])
                        cur_data_batch = torch.mm(inverse_data, rotation_matrices[prev_l])

                        update_data_arrays.append(cur_data_batch)
                    cur_data = torch.cat(update_data_arrays, dim=0)
                    data_anchors.append(cur_data[:cur_data.shape[0]])

            test_loss_r = flow_loss(data, log_det)
            test_bpd_r = (test_loss_r.item() * data.shape[0] - log_det_logit) * (
                    1 / (np.log(2) * np.prod(data.shape))) + 8

            test_bpd += test_bpd_r * data.shape[0]
            test_loss += test_loss_r * data.shape[0]
            if batch_idx % 100 == 0:
                print("Batch {} loss {} bpd {}".format(batch_idx, test_loss_r, test_bpd_r))

        print("Total loss {} bpd {}".format(test_loss / total, test_bpd / total))
        sampling(rotation_matrices, data_anchors, bandwidths, image_name='images/RBIG_samples_{}.png'.format(dataset),
                       channel=channel, image_size=image_size, process_size=10)
