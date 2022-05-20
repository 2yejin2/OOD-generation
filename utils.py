import torch
import numpy as np
from sklearn import metrics
from collections import Iterable
from matplotlib import pyplot as plt

number_of_classes=10

def generate_nd_sphere_data(r, dims, n):
    # number of points
    X = np.zeros((n, dims))
    for i in range(dims):
        X[:, i] = np.random.normal(0, 1, n)
    D = np.sqrt(np.sum(np.square(X), axis=-1))
    X = X*1.0*r[:, np.newaxis]/D[:, None]
    y = np.ones((n, 1)) * (number_of_classes-1)
    #Y = np_utils.to_categorical(Y, 3)
    return X, y

def get_hyper_info(target,z,n):
    inds = np.where(target[:, n] == 1)[0]
    z_n = z[inds].detach().cpu().numpy()
    m = np.mean(z_n, axis=0)
    cov = np.cov(z_n.T)
    cov_inv = np.linalg.pinv(cov)
    maha_dist = []
    for k in range(z_n.shape[0]):
        maha_dist.append(get_mahalanobis_dist(m, cov_inv, z_n[k]))
    maha_dist.sort()
    dist_95 = maha_dist[int(0.95 * len(maha_dist))]
    return [m,cov,dist_95]

def one_hot_encoding(data_y):
    data_y=data_y.cpu()
    data_y=data_y.numpy()
    cls = set(data_y)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, data_y)))
    one_hot=torch.from_numpy(one_hot).float()
    return one_hot.cuda()

def get_mahalanobis_dist(m, cov_inv, z):
    return np.sqrt((z - m).dot(cov_inv).dot((z - m).T))


def generate_nd_hyper_data(m, Cov, r, dims, n):
    # generate for randorm radii between r and r+0.5r
    rs = np.random.uniform(r, r + 0.5 * r, n)
    X, y = generate_nd_sphere_data(rs, dims, n)
    Cov_inv = Cov
    C = np.zeros_like(Cov)

    for i in range(C.shape[0]):
        for j in range(i + 1):
            s1 = 0
            for k in range(j):
                s1 += C[i, j] * C[j, k]
            s2 = 0
            for k in range(j):
                s2 += C[j, k] * C[j, k]
            C[i, j] = (Cov[i, j] - s1) / np.sqrt(np.abs(Cov[j, j] - s2))
    for i in range(X.shape[0]):
        X[i, :] = np.dot(C, X[i, :]) + m
    return X, y


def auroc(in_dist, out_dist):

    Y1 = out_dist
    X1 = in_dist
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))

        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr

    return aurocBase


def sklearn_auroc(p_in, p_out):
    y = list(torch.ones(p_in.shape[0]))
    y.extend(torch.zeros(p_out.shape[0]))
    pred = list(torch.cat((p_in.detach().cpu(), p_out.detach().cpu()), axis=0))
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

    idx_95 = (np.abs(tpr - 0.95)).argmin()
    idx_80 = (np.abs(tpr - 0.80)).argmin()
    return metrics.auc(fpr, tpr), fpr[idx_95], fpr[idx_80]


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass


class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try:
            return len(self.read())
        except:
            return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.', v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def draw_curve(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss,accuracy = zip(*train_logger)
        _,test_loss,test_accuracy = zip(*test_logger)

        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch, test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()

        plt.plot(epoch,accuracy,color='blue',label="Train Accuracy")
        plt.plot(epoch,test_accuracy, color='red', label="Test Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Curve")
        plt.legend()
        plt.savefig(work_dir + '/accuracy_curve.png')
        plt.close()

def draw_curve_loss(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss= zip(*train_logger)
        _,test_loss = zip(*test_logger)
        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch,test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()


