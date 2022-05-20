import argparse
import os
import time
from model import VAE,loss_function,Classifier
from dataloader import data_set_class
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import Logger, AverageMeter, one_hot_encoding, draw_curve
import json

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs',default=200, type=int)
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--gpu_id', default='0', type=str, help='devices')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--opt', default='ada', type=str, help='optimizer')
parser.add_argument('--momentum', default=0.0, type=float, help='optimizer momentum')
parser.add_argument('--in_data', default='./data/train/mnist', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--save_path', default='./result', type=str, help='savepath')
parser.add_argument('--seed',default=False,type=bool,help='Set seed')
args = parser.parse_args()

def main():
    save_path = args.save_path
    if not os.path.exists(os.path.join(args.save_path,'classifier')):
        os.makedirs(os.path.join(args.save_path, 'classifier'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed==True:
        torch.manual_seed(5)
        print('Manual seed')

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'classifier','configuration.json'), 'w') as f:
        json.dump(argparse_dict, f, indent=2)

    network = Classifier().cuda()
    train_loader_gen, vali_loader_gen, _ = data_set_class(batch_size=args.batch_size, test_dataset='./data/test/fmnist', data_set=args.in_data)
    cudnn.benchmark = True
    weights = [1 / 6000,1 / 6000,1 / 6000,1 / 6000,1 / 6000,1 / 6000,1 / 6000, 1 /6000, 1 / 6000, 1 / 6000, 1 / 30000]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)
    if args.opt =='ada':
        optimizer =optim.Adadelta(network.parameters(), lr=args.lr, rho=0.9, weight_decay=0.0001)

    train_logger = Logger(os.path.join(save_path, 'classifier', 'train_logger.log'))
    vali_logger = Logger(os.path.join(save_path, 'classifier', 'vali_logger.log'))


    for epoch in range(1, args.epochs+1):
        train(network, train_loader_gen, optimizer , args.batch_size, epoch, args.epochs, train_logger, criterion=criterion)
        vali(network, vali_loader_gen, args.batch_size, epoch, args.epochs, vali_logger, criterion=criterion)
        torch.save(network.state_dict(), '{0}{1}{2}_{3}.pth'.format(save_path,'/classifier/', 'model', epoch))

    draw_curve(os.path.join(save_path,'classifier'), train_logger, vali_logger)

    print("Process complete")


def train(model, train_loader, optimizer,batch_size, epoch, num_epoch, train_logger, criterion):
    accuracy,train_loss=AverageMeter(), AverageMeter()
    correct=0
    model.train()
    start = time.time()
    for i, (data, target) in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().float()
        acc = correct / (batch_size * (i + 1)) * 100
        accuracy.update(acc.item(), data.shape[0])
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.shape[0])
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 10 == 0 and i != 0:
            end = time.time() - start
            print('Epoch : [{0}/{1}] [{2}/{3}]  Train Loss : {loss.avg:.4f} Time : {time:.4f}'.format(
                        epoch, num_epoch, i, len(train_loader), loss=train_loss, time=end))
            start = time.time()
    train_logger.write([epoch,train_loss.avg,accuracy.avg])


def vali(model,vali_loader,batch_size,epoch,num_epoch,vali_logger,criterion):
  model.eval()
  vali_accuracy,vali_loss = AverageMeter(),AverageMeter()
  correct = 0
  for k,(data,target) in enumerate(vali_loader):
      data, target = data.cuda(), target.cuda()
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum().float()
      acc = correct / (batch_size * (k + 1)) * 100
      vali_accuracy.update(acc.item(), data.shape[0])
      loss = criterion(output, target)
      vali_loss.update(loss.item(), data.shape[0])
  vali_logger.write([epoch, vali_loss.avg, vali_accuracy.avg])

  print('\nValidation ========== Epoch : {0}/{1} Val Loss : {loss.avg:.4f} Val Accuracy : {acc.avg:.4f} \n'.format(epoch,num_epoch,loss=vali_loss, acc=vali_accuracy))

if __name__ == "__main__":
    main()
