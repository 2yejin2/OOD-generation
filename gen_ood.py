import argparse
import time
from sympy import Matrix
import numpy as np
import torch
from model import VAE
from utils import one_hot_encoding,get_hyper_info,generate_nd_hyper_data
from random import *
import random
import os
from dataloader import data_set_vae
from multiprocessing import Pool
import multiprocessing
from functools import partial
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
parser.add_argument('--model_epoch', default=100, type=int, help='epoch')
parser.add_argument('--gpu_id', default='0', type=str, help='devices')
parser.add_argument('--model_path', default='./result5', type=str, help='savepath')
parser.add_argument('--data', default='./data/train/mnist', type=str, help='datasets : mnist / fmnist')
parser.add_argument('--dataset', default='train', type=str, help='train / test')
args = parser.parse_args()


if not os.path.exists(os.path.join(args.data,'ood')):
    os.makedirs(os.path.join(args.data,'ood'))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

#load model
old_model=VAE().cuda()
old_model.load_state_dict(torch.load(os.path.join(args.model_path,'vae', 'model_{0}.pth'.format(args.model_epoch))))


############################# generate OOD samples #############################

def get_ood2(in_train_loader,old_model):

    for i, (data, target) in enumerate(in_train_loader):
        start = time.time()
        data,target = data.cuda(),one_hot_encoding(target.cuda()).float()
        recon_x, mu, logvar, z = old_model(data, target)
        target = target.cpu().numpy()
        maha_info=[]

        for n in range(len(target[0])):
            maha_info.append(get_hyper_info(target, z, n))
        Z = np.empty((target.shape[0], 8))

        for w in range(target.shape[1]):
            inds = np.where((target[:, w] == 1))[0]
            a, _ = generate_nd_hyper_data(maha_info[w][0], maha_info[w][1], maha_info[w][2], 8, inds.shape[0])
            Z[inds, :] = a

        Z=torch.FloatTensor(Z).cuda()
        target=torch.FloatTensor(target).cuda()
        output = old_model.decoder(Z, target)
        output=output.detach().cpu()

        p=Pool(30)
        p.map(partial(save_image,dataset=args.dataset,i=i,type=2),[n for n in output])
        p.close()

        end = time.time() - start
        print('OOD_2 >>> Time : {time:.4f} min=============== Iteration : {iter:} =============='.format( time=end/60, iter=i))




def get_ood1(in_train_loader,old_model):
    for i, (data, target) in enumerate(in_train_loader):
        start = time.time()
        data,target = data.cuda(),one_hot_encoding(target.cuda()).float()
        recon_x, mu, logvar, z = old_model(data, target)
        z.requires_grad_(True).cuda()
        grad=torch.zeros([data.size(2)**2,data.size(0),8])

        for k in range(recon_x.size(1)):
            grad[k]=torch.autograd.grad(outputs=recon_x[:,k].sum(),inputs=z,retain_graph=True,create_graph=True)[0].data
        grad=np.transpose(grad,axes=(1,0,2))

        p=Pool()
        nullspace=p.map(nullspace_,[n for n in grad])
        p.close()

        epsilon = torch.empty(recon_x.size(0)).uniform_(0.1, 28.0)
        rand_nums = torch.empty(recon_x.size(0)).uniform_(-1, 1)

        for m in range(recon_x.size(0)):
            epsilon_n = torch.empty(1, nullspace[m].shape[0]).uniform_(-1, 1)
            vec = torch.sum(epsilon_n * nullspace[m].T, dim=1)
            vec = vec / torch.sqrt(torch.sum(vec ** 2))
            vec = torch.FloatTensor(vec).cuda()
            recon_x[m] = recon_x[m] + (rand_nums[m] * epsilon[m] * vec).view(-1, 28, 28)
            recon_x[(recon_x > 1.0)] = 1.0
            recon_x[(recon_x < 0.0)] = 0.0
        recon_x=recon_x.detach().cpu()

        p=Pool(30)
        p.map(partial(save_image,dataset=args.dataset,i=i,type=1),[n for n in recon_x])
        p.close()

        end=time.time()-start
        print('OOD_1 >>> Time : {time:.4f} min=============== Iteration : {iter:} =============='.format( time=end/60, iter=i))

######################### multiprocessing function ##########################

def nullspace_(grad_x):
    nullspace=Matrix(grad_x.T).nullspace()
    nullspace = torch.FloatTensor(nullspace)
    return nullspace

def save_image(output, dataset, i,type):
    if not os.path.exists(os.path.join(args.data, 'ood',dataset)):
        os.makedirs(os.path.join(args.data, 'ood',dataset,'ood'))
    process=multiprocessing.current_process()
    if dataset=='train':
        index=random.randint(0,8)
        vutils.save_image(output,os.path.join(args.data,'ood',dataset,'ood')+'/{0}.{1}_{2}_{3}.png'.format(process.pid,index,i,type))
    elif dataset=='test':
        index=random.randint(0,1000)
        vutils.save_image(output,os.path.join(args.data,'ood',dataset,'ood')+'/{0}.{1}_{2}_{3}.png'.format(process.pid,index,i,type))

#############################################################################

if args.dataset=='train':
    in_train_loader,_=data_set_vae(batch_size=args.batch_size,data_set=args.data)
    get_ood1(in_train_loader,old_model)
    get_ood2(in_train_loader,old_model)

elif args.dataset=='test':
    _,in_test_loader=data_set_vae(batch_size=args.batch_size,data_set=args.data)
    get_ood1(in_test_loader, old_model)
    get_ood2(in_test_loader, old_model)
  
