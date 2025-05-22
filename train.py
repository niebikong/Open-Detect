import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataset
from data.splits import get_splits
from model import OpenDetectNet, train_model, validate_model
from utils import setup_seed, weight_init
from utils import reset_prototype

CUDA_LAUNCH_BLOCKING=1

datasets = [
    'mal',
    'USTC',
    'combined_USTC_mal'
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    USTC 10, 3, 2
    mal  10, 1, 6
    combined_USTC_mal 0, 1
    """
    parser.add_argument('--dset', default='USTC', help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initial_learning_rate')  # 0.001
    parser.add_argument('--batch_size', type=int, default=128)  # 32
    parser.add_argument('--epoch', type=int, default=100)  # 100
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--h', type=int, default=128, help='dimension of the hidden layer')  # 128
    parser.add_argument('--c', type=int, default=1, help='image channel')  #  default=3
    parser.add_argument('--temp_inter', type=float, default=1, help='temperature factor')  # 0.1
    parser.add_argument('--temp_intra', type=float, default=1, help='temperature factor')  # 1
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--arch', default='resnet18', help='net arch')

    parser.add_argument('--split', type=int, default=10, help='unknown splits')
    
    parser.add_argument('--lamda', type=float, default=0.005, help='balance param between gen & dis')


    args, _ = parser.parse_known_args()
    setup_seed(2022)  # 2022, 2024
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % args.gpu
    if not os.path.exists('./save_model/'):
        os.makedirs('./save_model/')

    known_classes, unknown_classes, known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)
    args.num_classes = len(known_classes)
    train_set = get_dataset(known_dataset, True, known_classes, 'reindex')
    val_set = get_dataset(known_dataset, False, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, False, unknown_classes, 'open')
    print(len(val_set), len(open_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last = False)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, drop_last = False)
    open_loader = DataLoader(open_set, batch_size=10, shuffle=False, num_workers=4, drop_last = False)
    
    open_detect_net = OpenDetectNet(args.arch, args.c, args.h, args.num_classes, args.temp_inter, args.temp_intra)
    open_detect_net = open_detect_net.cuda()
    open_detect_net.apply(weight_init)

    optimizer = optim.Adam(open_detect_net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    print(args)

    best = 0
    for epoch in range(args.epoch):
        train_model(open_detect_net, args, train_loader, epoch, optimizer)
        if epoch in [50, 80]:
            open_detect_net.prototypes = reset_prototype(open_detect_net, train_loader)
        val_acc = validate_model(open_detect_net, args, val_loader, epoch)
        scheduler.step()

        # save model
        if val_acc > best:
            torch.save(open_detect_net, './save_model/{}_split_{}.pt'.format(args.dset, args.split))
            best = val_acc
        
    
    print('Finished')