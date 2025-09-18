import os
import shutil
import torch
import logging
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    torch.distributed.destroy_process_group()

def is_main_process():
    return torch.distributed.get_rank() == 0


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def seed_torch(seed=728):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**30 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    # return
    # return
    # dir_path = os.path.join(args.store_root, args.store_name)
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    # torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        torch.save(state, filename.replace('pth.tar', 'best.pth.tar'))
        # shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))



def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def calibrate_mean_var_bayes(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    mu_mu_1, mu_var_1 = torch.tensor_split(m1, 2)
    var_mu_1, _ = torch.tensor_split(v1, 2)
    mu_mu_2, mu_var_2 = torch.tensor_split(m2, 2)
    var_mu_2, _ = torch.tensor_split(v2, 2)
    
    if torch.sum(var_mu_1) < 1e-10:
        return matrix
    
    matrix_mu, matrix_var = torch.tensor_split(matrix, 2, dim=1)

    
    if (var_mu_1 == 0.).any():
        valid = (var_mu_1 != 0.)
        factor = torch.clamp(var_mu_2[valid] / var_mu_1[valid], clip_min, clip_max)
        matrix_mu = (matrix_mu[:, valid] - mu_mu_1[valid]) * torch.sqrt(factor) + mu_mu_2[valid]
        matrix_var = (matrix_var[:, valid] + mu_var_1[valid]) * torch.sqrt(factor) + mu_var_2[valid]
#         assert (matrix_var<=0).sum() == 0.0, valid
        matrix = torch.cat((matrix_mu, matrix_var), dim=1)
        return matrix
    
    
    factor = torch.clamp(var_mu_2 / var_mu_1, clip_min, clip_max)
    
    matrix_mu = (matrix_mu - mu_mu_1) * torch.sqrt(factor) + mu_mu_2
    matrix_var = (matrix_var + mu_var_1) * torch.sqrt(factor) + mu_var_2

    matrix = torch.cat((matrix_mu, matrix_var), dim=1)
    return matrix


def get_lds_kernel_window(kernel, ks, sigma, bins):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    window_l, window_r = [], []
    mid_window = [kernel_window[half_ks]] if bins == 1 else [kernel_window[half_ks] / bins] * bins
    for i in range(half_ks):
        tmp_l = kernel_window[i]
        tmp_r = kernel_window[i - half_ks]
        tmp_window_l = [tmp_l] if bins == 1 else [tmp_l / bins] * bins
        tmp_window_r = [tmp_r] if bins == 1 else [tmp_r / bins] * bins
        window_l += tmp_window_l
        window_r += tmp_window_r
    kernel_window_bins = window_l + mid_window + window_r

    print(f'LDS kernel window{kernel_window}')
    print(f'LDS kernel window{kernel_window_bins}')

    kernel_window = kernel_window_bins
    return kernel_window



#####################################
def per_label_frobenius_norm(features, labels):
    """
    features: Tensor of shape (N, D)
    labels: Tensor of shape (N,)
    Returns: dict {label: avg Frobenius norm}
    """
    features = features.view(features.size(0), -1)  # Ensure shape (N, D)
    labels = labels.view(-1)  # Ensure shape (N,)

    unique_labels = labels.unique()
    frob_norms = {}

    for label in unique_labels:
        mask = labels == label
        feats = features[mask]  # (n_c, D)
        if feats.size(0) == 0:
            continue
        norms = torch.norm(feats, p='fro', dim=1)  # L2 norm per row
        avg = norms.mean().item()
        frob_norms[int(label.item())] = avg

    frob_norm = {key  : frob_norms[key] for key in sorted(frob_norms.keys())}

    return frob_norm



#####################################
def cal_per_label_Frob(model, train_loader):
    model.eval()
    feature, label = [], []
    with torch.no_grad():
        for idx, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            #
            # pred, none, none, encoding
            _,_, _, z_pred = model(x)
            feature.append(z_pred.cpu())
            label.append(y)
        features = torch.cat(feature, dim=0)
        labels = torch.cat(label)
    frob_norm = per_label_frobenius_norm(features, labels)
    return frob_norm



#####################################
def cal_per_label_mae(model, train_loader):
    """
    #output: Tensor of shape (N, 1)
    #target: Tensor of shape (N,) with M unique labels
    Returns: dict mapping each label to its MAE
    """
    output, target = [], []
    with torch.no_grad():
        for idx, (x, y, _) in enumerate(train_loader):
            x = x.to(device)
            # pred, none, none, encoding
            y_pred, _, _, _ = model(x)
            #print(y_pred)
            target.extend(y.squeeze(-1).tolist())
            output.extend(y_pred.cpu().squeeze(-1).tolist())
            
        N = len(target)
        #print(f'N is {N}')
        output = torch.tensor(output).reshape(N,)  # (N,)
        target = torch.tensor(target).reshape(N,)  # (N,)

        unique_labels = target.unique()
        mae_dict = {}

        for label in unique_labels:
            mask = target == label
            if mask.sum() == 0:
                continue
            pred_subset = output[mask]
            true_subset = target[mask].float()
            mae = torch.abs(pred_subset - true_subset).mean()
            mae_dict[int(label.item())] = mae.item()
    #
    return mae_dict


def cal_MAE_and_Frobs(model_regression, train_loader, test_loader):
    per_label_MAE_train = cal_per_label_mae(model_regression, train_loader)
    print('===============train key MAE============='+'\n')
    k_train = [k for k in per_label_MAE_train.keys()]
    print(f'k_train is {k_train}')
    v_train = [per_label_MAE_train[k] for k in per_label_MAE_train.keys()]
    print(f'v_train is {v_train}')
    print('===============train MAE============='+'\n')
    per_label_MAE_test = cal_per_label_mae(model_regression, test_loader)
    print('===============test key MAE============='+'\n')
    k_test = [k for k in per_label_MAE_test.keys()]
    print(f'k_test is {k_test}')
    v_test = [per_label_MAE_test[k] for k in per_label_MAE_test.keys()]
    print(f'v_test is {v_test}')
    print('===============test MAE============='+'\n')
    #
    per_label_Frobs_train = cal_per_label_Frob(model_regression, train_loader)
    per_label_Frobs_test = cal_per_label_Frob(model_regression, test_loader)
    k_frobs_train = [k for k in per_label_Frobs_train.keys()]
    k_frobs_test = [k for k in per_label_Frobs_test.keys()]
    v_frobs_train = [per_label_Frobs_train[k] for k in per_label_Frobs_train.keys()]
    v_frobs_test = [per_label_Frobs_train[k] for k in per_label_Frobs_test.keys()]
    print('===============train frobs key============='+'\n')
    print(f'k_frobs_train is {k_frobs_train}')
    print('===============train frobs============='+'\n')
    print(f'v_frobs_train is {v_frobs_train}')
    print('===============test frobs key============='+'\n')
    print(f'k_frobs_test is {k_frobs_test}')
    print('===============test frobs============='+'\n')
    print(f'v_frobs_test is {v_frobs_test}')

    ####
    df_train = pd.DataFrame({
        "train MAE labels" : k_train,
        "train MAE" : v_train,
        "train Frobs" : v_frobs_train,
    })

    df_test = pd.DataFrame({
        "test MSE labels" : k_test,
        "test MAE" : v_test,
        "test Frobs" : v_frobs_test
    })


    df_train.to_csv("VIR_train.csv", index=False)
    df_test.to_csv("VIR_test.csv", index=False)

    return