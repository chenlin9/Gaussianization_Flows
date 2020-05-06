import os
import torch
import math
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.distributions as tdist

# set random seed
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda'
normal_distribution = tdist.Normal(0, 1)


def generate_bandwidth(datapoints):
    total_datapoints = datapoints.shape[0]
    bandwidth = torch.std(datapoints, dim=0, keepdim=True) * (
                4. * np.sqrt(math.pi) / ((math.pi ** 4) * total_datapoints)) ** (0.2)
    return bandwidth


def logit_transform(image, lambd=1e-5):
    image = lambd + (1 - 2 * lambd) * image
    return torch.log(image) - torch.log1p(-image)


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
    log_jacob = log_jacob.sum()
    loss = -(log_probs + log_jacob)
    if size_average:
        loss /= u.size(0)
    return loss


def sigmoid_transform(samples, lambd=1e-5):
    samples = torch.sigmoid(samples)
    samples = (samples - lambd) / (1 - 2 * lambd)
    return samples


def logistic_kernel_pdf(x, datapoint, h):
    n = datapoint.shape[0]
    log_pdfs = -(x[None, ...] - datapoint[:, None, :]) / h[:, None, :] - torch.log(h[:, None, :]) - \
               2. * F.softplus(-(x[None, ...] - datapoint[:, None, :]) / h[:, None, :]) - np.log(n)
    log_pdf = torch.logsumexp(log_pdfs, dim=0)
    pdf = torch.exp(log_pdf)
    return pdf


def logistic_kernel_log_pdf(x, datapoint, h):
    n = datapoint.shape[0]
    log_pdfs = -(x[None, ...] - datapoint[:, None, :]) / h[:, None, :] - torch.log(h[:, None, :]) - \
               2. * F.softplus(-(x[None, ...] - datapoint[:, None, :]) / h[:, None, :]) - np.log(n)
    log_pdf = torch.logsumexp(log_pdfs, dim=0)
    return log_pdf


def logistic_kernel_cdf(x, datapoint, h):
    n = datapoint.shape[0]
    log_cdfs = - F.softplus(-(x[None, ...] - datapoint[:, None, :]) / h[None, ...]) - np.log(n)
    log_cdf = torch.logsumexp(log_cdfs, dim=0)
    cdf = torch.exp(log_cdf)
    return cdf


# return log(CDF)
def logistic_kernel_log_cdf(x, datapoint, h):
    n = datapoint.shape[0]
    log_cdfs = - F.softplus(-(x[None, ...] - datapoint[:, None, :]) / h[None, ...]) - np.log(n)
    log_cdf = torch.logsumexp(log_cdfs, dim=0)
    return log_cdf


# return log(1-CDF)
def logistic_kernel_log_sf(x, datapoint, h):
    n = datapoint.shape[0]
    log_sfs = - F.softplus((x[None, ...] - datapoint[:, None, :]) / h[None, ...]) - np.log(n)
    log_sf = torch.logsumexp(log_sfs, dim=0)
    return log_sf


def compute_log_det(x, gaussianalized_x, datapoints, cdf_mask, log_cdf_l, cdf_mask_left,
                    log_sf_l, cdf_mask_right, h, squeeze=True):
    n = datapoints.shape[0]
    log_pdfs = -(x[None, ...] - datapoints[:, None, :]) / h[None, ...] - torch.log(h[None, ...]) - \
               2 * F.softplus(-(x[None, ...] - datapoints[:, None, :]) / h[None, ...]) - np.log(n)
    log_pdf = torch.logsumexp(log_pdfs, dim=0)

    log_gaussian_derivative_good = tdist.Normal(0, 1).log_prob(gaussianalized_x) * cdf_mask
    cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
    cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)
    log_gaussian_derivative_left = (torch.log(torch.sqrt(-2 * cdf_l_bad_left_log))
                                    - log_cdf_l) * cdf_mask_left
    log_gaussian_derivative_right = (torch.log(torch.sqrt(-2. * cdf_l_bad_right_log))
                                     - log_sf_l) * cdf_mask_right
    log_gaussian_derivative = log_gaussian_derivative_good + log_gaussian_derivative_left + log_gaussian_derivative_right

    if squeeze:
        log_det = (log_pdf - log_gaussian_derivative).sum(dim=-1)  # only keep batch size
    else:
        log_det = (log_pdf - log_gaussian_derivative)  # keep original size
    return log_det


def close(a, b, rtol=1e-5, atol=1e-4):
    equal = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return equal


# compute inverse normal CDF
def logistic_inverse_normal_cdf(x, bandwidth, datapoints):
    mask_bound = 0.5e-7
    cdf_l = logistic_kernel_cdf(x, datapoints, h=bandwidth)
    log_cdf_l = logistic_kernel_log_cdf(x, datapoints, h=bandwidth)  # log(CDF)
    log_sf_l = logistic_kernel_log_sf(x, datapoints, h=bandwidth)  # log(1-CDF)

    # Approximate Gaussian CDF
    # inv(CDF) ~ np.sqrt(-2 * np.log(1-x)) #right, x -> 1
    # inv(CDF) ~ -np.sqrt(-2 * np.log(x)) #left, x -> 0
    # 1) Step1: invert good CDF
    cdf_mask = ((cdf_l > mask_bound) & (cdf_l < 1 - (mask_bound))).float()
    # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
    cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
    inverse_l = normal_distribution.icdf(cdf_l_good)

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
    return inverse_l, cdf_mask, [log_cdf_l, cdf_mask_left], [log_sf_l, cdf_mask_right]


# Inverting the function using bisection, but with respect to the composed function
def invert_bisection_combo(x, rotation_matrix, datapoints, bandwidth, verbose=False, lower=-1e5, upper=1e5):
    z = torch.mm(x, rotation_matrix.permute(1, 0))
    iteration = int(np.log2(upper - lower) + np.log2(1e6))

    upper = torch.tensor(upper).repeat(*z.shape).to(device)
    lower = torch.tensor(lower).repeat(*z.shape).to(device)
    for i in range(iteration):
        mid = (upper + lower) / 2.
        inverse_mid, _, _, _ = logistic_inverse_normal_cdf(mid, bandwidth, datapoints)
        right_part = (inverse_mid < z).float()
        left_part = 1. - right_part

        correct_part = (close(inverse_mid, z, rtol=1e-6, atol=0)).float()
        lower = (1. - correct_part) * (right_part * mid + left_part * lower) + correct_part * mid
        upper = (1. - correct_part) * (right_part * upper + left_part * mid) + correct_part * mid

    if verbose:
        print("Average error {}".format(torch.sum(upper - lower) / np.prod(x.shape)))
    return mid


def sampling(rotation_matrices, data_anchors, bandwidths, image_name='RBIG_samples.png',
             sample_num=100, channel=1, image_size=28, process_size=10):
    print("Start sampling")
    x = torch.randn(sample_num, channel * image_size * image_size).to(device)
    for i in range(sample_num // process_size):
        for l in range(len(rotation_matrices) - 1, -1, -1):
            print("Sampling layer {}".format(l))
            x[i * process_size: (i + 1) * process_size, :] = invert_bisection_combo(
                    x[i * process_size: (i + 1) * process_size, :], rotation_matrices[l], data_anchors[l].to(device),
                    bandwidths[l])

        x[i * process_size: (i + 1) * process_size, :] = sigmoid_transform(
            x[i * process_size: (i + 1) * process_size, :])
    x = x.reshape(sample_num, channel, image_size, image_size)
    images_concat = torchvision.utils.make_grid(x, nrow=int(sample_num ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, image_name)


def generate_orthogonal_matrix(data_anchor, type="PCA", verbose=False):
    if verbose:
        print("Generating {} matrix.".format(type))
    if type == "random":
        # Random orthogonal matrix
        random_mat = torch.randn(data_anchor.shape[-1], data_anchor.shape[-1])
        rotation_matrix, _ = torch.qr(random_mat)
    elif type == "PCA":
        # PCA
        rotation_matrix, _, _ = torch.svd(
            torch.mm(data_anchor.permute(1, 0), data_anchor))

    if verbose:
        print("{} matrix generated!".format(type))
    return rotation_matrix
