from __future__ import print_function
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from folder import CustomImageFolder

def data_set_class(batch_size, data_set,test_dataset):

        if data_set == './data/train/mnist':
            train_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            dataset_test_in = datasets.ImageFolder(root='./data/train/mnist/test', transform=test_transforms)
            dataset_train_in = datasets.ImageFolder(root='./data/train/mnist/train', transform=train_transforms)
            dataset_test_out_gen = CustomImageFolder(root='./data/train/mnist/ood/test', transform=test_transforms)
            dataset_train_out=CustomImageFolder(root='./data/train/mnist/ood/train', transform=train_transforms)

            if test_dataset=='./data/test/fmnist':
                dataset_test_out_oth = CustomImageFolder(root='./data/test/fmnist/test',transform=transforms.Compose(
                    [transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))

        elif data_set == './data/train/fmnist':
            train_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            test_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))])

            dataset_test_in = datasets.ImageFolder(root='./data/train/fmnist/test', transform=test_transforms)
            dataset_train_in = datasets.ImageFolder(root='./data/train/fmnist/train', transform=train_transforms)
            dataset_test_out_gen = CustomImageFolder(root='./data/train/fmnist/ood/test', transform=test_transforms)
            dataset_train_out=CustomImageFolder(root='./data/train/fmnist/ood/train', transform=train_transforms)

            if test_dataset=='./data/test/mnist':
                dataset_test_out_oth = CustomImageFolder(root='./data/test/mnist/test',transform=transforms.Compose(
                    [transforms.Grayscale(),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


        test_loader = DataLoader(dataset_test_in+dataset_test_out_gen, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
        train_loader = DataLoader(dataset_train_in+dataset_train_out, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
        test_loader_out = DataLoader(dataset_test_out_oth+dataset_test_in, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)

        return train_loader, test_loader, test_loader_out

def data_set_vae(batch_size, data_set):
        if data_set == './data/train/mnist':
            train_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transforms = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_train = datasets.ImageFolder(root='./data/train/mnist/train',transform=train_transforms)
            dataset_test = datasets.ImageFolder(root='./data/train/mnist/test',transform=test_transforms)

        elif data_set == './data/train/fmnist':
            train_transforms = transforms.Compose(
                [transforms.Grayscale(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
            test_transforms = transforms.Compose(
                [transforms.Grayscale(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
            dataset_train = datasets.ImageFolder(root='./data/train/fmnist/train', transform=train_transforms)
            dataset_test =datasets.ImageFolder(root='./data/train/fmnist/test',transform=test_transforms)

        in_train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
        in_test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)

        return in_train_loader, in_test_loader