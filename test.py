import argparse
import os
from model import VAE,loss_function,Classifier
from dataloader import data_set_class
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Logger,AverageMeter,one_hot_encoding,sklearn_auroc

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--in_data', default='./data/train/mnist', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--out_data', default='./data/test/fmnist', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--save_path', default='./result', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--model_epoch',default=200, type=int)
parser.add_argument('--model_path', default='./result101', type=str, help='modelpath')
parser.add_argument('--seed',default=False,type=bool,help='Set seed')
parser.add_argument('--gpu_id', default='0', type=str, help='devices')
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed == True:
        torch.manual_seed(5)
        print('Manual seed')

    network = Classifier().cuda()

    _, test_loader_gen,test_loader_fmnist= data_set_class(batch_size=1000, data_set=args.in_data, test_dataset=args.out_data)

    auroc_logger = Logger(os.path.join(args.save_path, 'auroc_logger.log'))
    for i in range(args.model_epoch):
        network.load_state_dict(torch.load(os.path.join(args.model_path, 'classifier', 'model_{0}.pth'.format(i+1))))
        test_classi(network,i, test_loader_gen, test_loader_fmnist, auroc_logger)


def test_classi(model,model_epoch, test_loader_gen, test_loader_out, auroc_logger):
    model.eval()
    gen, out = AverageMeter(),AverageMeter()

    for k, (data, target) in enumerate(test_loader_gen):
        data = data.cuda()
        target = np.array(target)
        in_ind = np.where(target != 10)[0]
        out_ind = np.delete(np.arange(len(target)), in_ind, None)
        in_output = torch.max(model(data[in_ind]), axis=1)[0]
        out_output = torch.max(model(data[out_ind]), axis=1)[0]
        auroc = sklearn_auroc(in_output, out_output)
        gen.update(auroc[0].item(), data.shape[0])


    for m, (data, target) in enumerate(test_loader_out):
        data = data.cuda()
        target = np.array(target)
        in_ind = np.where(target != 10)[0]
        out_ind = np.delete(np.arange(len(target)), in_ind, None)
        in_output = torch.max(model(data[in_ind]), axis=1)[0]
        out_output = torch.max(model(data[out_ind]), axis=1)[0]
        auroc = sklearn_auroc(in_output, out_output)
        out.update(auroc[0].item(), data.shape[0])

    auroc_logger.write([args.in_data,model_epoch,'==> gen_ood : ',gen.avg, args.out_data,': ',out.avg])
    print('\nTest ========== Epoch : {0} ood_auroc : {auroc:.4f} gen_ood_auroc : {genauroc: .4f} \n'.format(model_epoch,auroc=out.avg,genauroc=gen.avg))

if __name__ == "__main__":
    main()
