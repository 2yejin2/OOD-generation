import argparse
import os
import time
import torch.nn as nn
from model import VAE,loss_function,Classifier
import utils
from dataloader import data_set_vae
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import Logger, AverageMeter, one_hot_encoding, draw_curve_loss
import json

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs',default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--gpu_id', default='3', type=str, help='devices')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--opt', default='rmsprop', type=str, help='optimizer')
parser.add_argument('--momentum', default=0.0, type=float, help='optimizer momentum')
parser.add_argument('--in_data', default='./data/train/mnist', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--save_path', default='./result', type=str, help='savepath')
parser.add_argument('--seed',default=False,type=bool,help='Set seed')
args = parser.parse_args()


def main():
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'vae'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed==True:
        torch.manual_seed(5)
        print('Manual seed')

    argparse_dict=vars(args)
    with open(os.path.join(args.save_path, 'vae','configuration.json'), 'w') as f:
        json.dump(argparse_dict, f, indent=2)

    network = VAE().cuda()
    in_train_loader, in_vali_loader = data_set_vae(batch_size=args.batch_size, data_set= args.in_data)
    cudnn.benchmark = True

    if args.opt == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=args.lr, momentum=args.momentum)
    if args.opt == 'ada':
        optimizer = optim.Adadelta(network.parameters(), lr=args.lr, weight_decay=0.001)

    train_logger = Logger(os.path.join(save_path,'vae', 'train_loss.log'))
    vali_logger = Logger(os.path.join(save_path,'vae', 'vali_loss.log'))

    for epoch in range(1, args.epochs + 1):
        train(network, in_train_loader, optimizer, epoch, args.epochs, train_logger)
        vali(network, in_vali_loader, epoch, args.epochs, vali_logger)
        torch.save(network.state_dict(), '{0}{1}{2}_{3}.pth'.format(save_path, '/vae/', 'model', epoch))

    draw_curve_loss(os.path.join(save_path, 'vae'), train_logger, vali_logger)

    print("Process complete")


def train(model, in_train_loader, optimizer, epoch, num_epoch,train_logger):
    train_loss = AverageMeter()
    model.train()
    start = time.time()
    for i, (data, target) in enumerate(in_train_loader):
        data, target = data.cuda(), target.cuda()
        target=one_hot_encoding(target)
        recon_x,mu,logvar,z = model(data,target)
        # recon_xnorm = (recon_x - recon_x.min()) / (recon_x.max()-recon_x.min())
        # loss = loss_function(recon_xnorm,data,mu,logvar).cuda()
        loss_function= nn.MSELoss()
        loss=loss_function(recon_x, data)
        train_loss.update(loss.item(), data.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            end = time.time() - start
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss.avg:.4f} Time : {time:.4f}'.format(
                        epoch, num_epoch, i, len(in_train_loader), loss=train_loss, time=end))
            start = time.time()
    train_logger.write([epoch,train_loss.avg])


def vali(model,vali_loader,epoch,num_epoch,vali_logger):
  model.eval()
  vali_loss = AverageMeter()
  for i,(data,target) in enumerate(vali_loader):
        data, target = data.cuda(), target.cuda()
        target = one_hot_encoding(target)
        recon_x, mu, logvar, z = model(data, target)
        # recon_xnorm = (recon_x - recon_x.min()) / (recon_x.max()-recon_x.min())
        # vali_lss = loss_function(recon_xnorm, data, mu, logvar).cuda()
        vali_lss_function= nn.MSELoss()
        vali_lss=vali_lss_function(recon_x, data)
        vali_loss.update(vali_lss.item(), data.shape[0])

  print('\nValidation ========== Epoch : {0}/{1} Validation Loss : {loss.avg:.4f} \n'.format(epoch,num_epoch,loss=vali_loss))
  vali_logger.write([epoch,vali_loss.avg])


if __name__ == "__main__":
    main()
