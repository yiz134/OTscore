import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import utils.network as network
from torch.utils.data import DataLoader
from utils.data_list import ImageList_idx
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import math
from utils.ot_score_utils import *


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
def data_load(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_src = open(args.s_dset_path).readlines()
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(), datadir=args.datadir)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), datadir=args.datadir)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    dsets["source"] = ImageList_idx(txt_src, transform=image_test(), datadir=args.datadir)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)
    return dset_loaders

def gmm(all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + args.epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += args.epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def evaluation(loader, netF, netB, netC, args, cnt, ot_scorer=None):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1].cuda()
        idx = data[2]
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            all_idx = idx
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            all_idx = torch.cat((all_idx, idx), 0)

            
    _, predict = torch.max(all_output, 1)
    if args.dset=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output = nn.Softmax(dim=1)(all_output)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    aff = all_output.float()
    uniform = torch.ones(len(all_fea),args.class_num)/args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    pi = gamma.sum(dim=0)
    mu = torch.matmul(gamma.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)
    zz, gamma = gmm((all_fea), pi, mu, gamma)
    pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if args.dset=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
    
    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()
    PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    JMDS = (LPG * PPL)

    print("Computing otscore")


    args.otscorer.tar_labels = pred_label
    args.otscorer.compute_semi_discrete_OT()
    ot_score = args.otscorer.compute_ot_score(indices=all_idx)
    ot_min = ot_score.min()
    ot_max = ot_score.max()
    ot_score = (ot_score - ot_min) / (ot_max - ot_min)
    sample_weight = 2 * ot_score * JMDS

    if args.dset=='VISDA-C':
        return aff, sample_weight, aacc/100
    return aff, sample_weight, accuracy
    
def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)

def mixup(x, c_batch, t_batch, netF, netB, netC, args):
    lam = (torch.from_numpy(np.random.beta(0.2, 0.2, [len(x)]))).float().cuda()
    t_batch = torch.eye(args.class_num).cuda()[t_batch.argmax(dim=1)]
    shuffle_idx = torch.randperm(len(x))
    mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = netC(netB(netF(mixed_x)))
    return KLLoss(mixed_outputs, mixed_t, mixed_c, args)

def train_target(args):
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type="wn", class_num = args.class_num, bottleneck_dim=256).cuda()
    
    modelpath = args.output_dir_src + '/source_F_{}.pt'.format(args.seed)
    print('modelpath: {}'.format(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B_{}.pt'.format(args.seed)
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_{}.pt'.format(args.seed)
    netC.load_state_dict(torch.load(modelpath))
        
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1.0}]
    
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    
    crop_size = 224
    augment1 = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
            
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    cnt = 0

    dset_loaders = data_load(args)
    
    epochs = []
    accuracies = []
    
    netF.eval()
    netB.eval()
    netC.eval()
    args.current_epoch = 0
    with torch.no_grad():
        src_mean_features, src_labels = compute_cls_mean_features_BFC(netB, netF, netC, dset_loaders["source"], ratio=1.0)
        tar_features, tar_pseudo_labels = extract_features_BFC(netB, netF, netC, dset_loaders["test"])
        args.otscorer = OT_SCORE(src_mean_features, src_labels, tar_features, tar_pseudo_labels)
        args.otscorer.compute_semi_discrete_OT()
        soft_pseudo_label, coeff, accuracy = evaluation(
            dset_loaders["test"], netF, netB, netC, args, cnt
        )
        epochs.append(cnt)
        accuracies.append(np.round(accuracy*100, 2))
    netF.train()
    netB.train()
    netC.train()
    
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // (args.interval)
    iter_num = 0
    args.current_epoch = 0
    print('\nTraining start\n')

    iter_test = iter(dset_loaders["target"])
    for iter_num in tqdm(range(0, max_iter)):
        args.current_epoch = iter_num / len(dset_loaders["target"])
        try:
            inputs_test, label, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)
        pred = soft_pseudo_label[tar_idx]
        pred_label = pred.argmax(dim=1)
        
        coeff, pred = map(torch.autograd.Variable, (coeff, pred))
        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()
        coeff = coeff.cuda()
        pred = pred.cuda()
        pred_label = pred_label.cuda()
        
        target_mixup_kl_loss = mixup(images1, coeff[tar_idx], pred, netF, netB, netC, args)
        
        if iter_num < 0.0 * interval_iter + 1:
            target_mixup_kl_loss *= 1e-6
            
        optimizer.zero_grad()
        target_mixup_kl_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(args.name, iter_num, max_iter)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            
            netF.eval()
            netB.eval()
            netC.eval()
            
            cnt += 1
            with torch.no_grad():
                soft_pseudo_label, coeff, accuracy = evaluation(dset_loaders["test"], netF, netB, netC, args, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy*100, 2))

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))
            netF.train()
            netB.train()
            netC.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir, f'target_F_{args.seed}.pt'))
    torch.save(netB.state_dict(), osp.join(args.output_dir, f'target_B_{args.seed}.pt'))
    torch.save(netC.state_dict(), osp.join(args.output_dir, f'target_C_{args.seed}.pt'))
        
        
    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--max_epoch', type=int, default=1, help="max iterations")
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--output_src', type=str, default='output')
    parser.add_argument('--datadir', type=str, default='datadir')


    args = parser.parse_args()


    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'image_CLEF':
        args.names = ['c', 'i', 'p']
        args.class_num = 12
    if args.dset == 'domainnet':
        args.names = ['clipart', 'real', 'painting', 'sketch']
        args.class_num = 126
        
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    if args.s < 0 or args.s >= len(args.names):
        raise ValueError(f"Invalid source index s={args.s}, valid range: [0, {len(args.names)-1}]")
    if args.t < 0 or args.t >= len(args.names):
        raise ValueError(f"Invalid target index t={args.t}, valid range: [0, {len(args.names)-1}]")
    if args.s == args.t:
        raise ValueError("Source and target must be different (s != t).")

    start = time.time()
    print("source:", args.names[args.s])
    print("target:", args.names[args.t])
    args.s_dset_path = osp.join(args.datadir, args.names[args.s] + '_list.txt')
    args.t_dset_path = osp.join(args.datadir, args.names[args.t] + '_list.txt')
    args.test_dset_path = osp.join(args.datadir, args.names[args.t] + '_list.txt')

    args.output_dir_src = osp.join(args.output_src, args.dset, args.names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.dset, args.names[args.s][0].upper()+args.names[args.t][0].upper())
    args.name = args.names[args.s][0].upper()+args.names[args.t][0].upper()

    os.makedirs(args.output_dir_src, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = osp.join(args.output_dir, f'target_F_{args.seed}.pt')
    log_path = osp.join(args.output_dir, f'log{random.randint(100000, 999999)}.txt')
    if not osp.exists(ckpt_path):
        args.out_file = open(log_path, 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
    else:
        print('{} exists. Overwriting and retraining...'.format(log_path))
        args.out_file = open(log_path, 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)

    total_time = time.time() - start
    log_str = 'Consumed time : {} h {} m {}s'.format(total_time // 3600, (total_time // 60) % 60,
                                                     np.round(total_time % 60, 2))
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
