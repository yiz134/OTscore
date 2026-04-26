import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import utils.network as network
import utils.loss as loss
from torch.utils.data import DataLoader
from utils.data_list import ImageList
import random
from tqdm import tqdm
from utils.loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
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
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
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
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), datadir=args.datadir)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), datadir=args.datadir)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test(), datadir=args.datadir)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_source(args):
    dset_loaders = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="wn", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    iter_source = iter(dset_loaders["source_tr"])
    for iter_num in tqdm(range(0, max_iter)):
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F_{}.pt".format(args.seed)))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B_{}.pt".format(args.seed)))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C_{}.pt".format(args.seed)))

    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type="wn", class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F_{}.pt'.format(args.seed)
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B_{}.pt'.format(args.seed)
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_{}.pt'.format(args.seed)
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.dset=='VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format("val", args.name, acc) + '\n' + acc_list
    else:
        acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format("val", args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=1, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office-home', 'image_CLEF', 'domainnet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--datadir', type=str, default='datadir')
    parser.add_argument('--test_target', action='store_true',
                        help='Run cross-domain test_target evaluation')
    args = parser.parse_args()

    print("Start")

    dset_key = args.dset.lower()
    if dset_key == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    elif args.dset == 'image_CLEF':
        names = ['c', 'i', 'p']
        args.class_num = 12
    elif dset_key == 'domainnet':
        names = ['clipart', 'real', 'painting', 'sketch']
        args.class_num = 126
    else:
        raise ValueError(f"Unsupported dataset: {args.dset}")

    if args.s < 0 or args.s >= len(names):
        raise ValueError(f"Invalid source index s={args.s}, valid range: [0, {len(names)-1}]")
    if args.t < 0 or args.t >= len(names):
        raise ValueError(f"Invalid target index t={args.t}, valid range: [0, {len(names)-1}]")
    if args.s == args.t:
        raise ValueError("Source and target must be different (s != t).")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    print("domain=", names[args.s])
    args.s_dset_path = osp.join(args.datadir, names[args.s] + '_list.txt')
    args.test_dset_path = osp.join(args.datadir, names[args.t] + '_list.txt')

    args.output_dir_src = osp.join(args.output, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    os.makedirs(args.output_dir_src, exist_ok=True)

    if not osp.exists((args.output_dir_src + '/source_F_{}.pt'.format(args.seed))):
        args.out_file = open(osp.join(args.output_dir_src, 'log_{}.txt'.format(args.seed)), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)

        if args.test_target:
            args.out_file = open(osp.join(args.output_dir_src, 'log_test_{}.txt'.format(args.seed)), 'w')
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.s_dset_path = osp.join(args.datadir, names[args.s] + '_list.txt')
            args.test_dset_path = osp.join(args.datadir, names[args.t] + '_list.txt')
            test_target(args)
    else:
        if args.test_target:
            args.out_file = open(osp.join(args.output_dir_src, 'log_test_{}.txt'.format(args.seed)), 'w')
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.s_dset_path = osp.join(args.datadir, names[args.s] + '_list.txt')
            args.test_dset_path = osp.join(args.datadir, names[args.t] + '_list.txt')
            test_target(args)