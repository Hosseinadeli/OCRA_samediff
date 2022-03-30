# --------------------
# Data
# --------------------

import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch

def fetch_dataloader(args, batch_size, train=True, train_val_split='none', download=True):
    """
    load dataset depending on the task
    currently implemented tasks:
        -svhn
        -cifar10
        -mnist
        -multimnist, multimnist_cluttered 
    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
        -train_val_split: 
            'none', load entire train dataset
            'train', load first 90% as train dataset
            'val', load last 10% as val dataset
            'train-val', load 90% train, 10% val dataset
    """
    kwargs = {'num_workers': 0, 'pin_memory': False} if args.device.type == 'cuda' else {}
    
    if args.task == 'svhn': 
        data_root = args.data_dir + '/svhn-data'
        #kwargs.pop('input_size', None)

        if args.num_targets == 1:
            if train:
                train_loader = torch.utils.data.DataLoader(datasets.SVHN(
                        root=data_root, split='train', download=download,transform=T.Compose([T.ToTensor(),
                            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])),batch_size=batch_size, shuffle=True, **kwargs)
                return train_loader

            else:
                test_loader = torch.utils.data.DataLoader(datasets.SVHN(
                        root=data_root, split='test', download=download,transform=T.Compose([T.ToTensor(),
                            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])), batch_size=batch_size, shuffle=False, **kwargs)
                return test_loader

                
    elif args.task == 'cifar10': 

        data_root  = args.data_dir + '/cifar10-data'    
        #kwargs.pop('input_size', None)
        transform = T.Compose([T.ToTensor()])  # transforms.RandomCrop(size=32, padding=shift_pixels),
        
        if train: 
            trainset = datasets.CIFAR10(root=data_root, train=True,download=download, transform=transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, **kwargs)
            return train_loader
        
        else: 
            testset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, **kwargs)
            return test_loader

    elif (args.task == 'mnist_ctrv') or (args.task == 'mnist'): 
        transforms = T.Compose([T.ToTensor()])
        dataset = datasets.MNIST(root=args.data_dir, train=train, download=download, transform=transforms)

        if train: 
            train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=train, drop_last=True, **kwargs)
            val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
            return train_dataloader, val_dataloader
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    
    else:
        
        # 2 mnist digits overlapping 
        if args.task == 'multimnist':
            data_root = args.data_dir + 'multimnist/'
#             train_datafile = 'mnist_overlap4pix_nodup_20fold_36_train.pt'
#             test_datafile = 'mnist_overlap4pix_nodup_20fold_36_test.pt'
            train_datafile = 'mnist_overlap4pix_nodup_50fold_36_train.npz'
            test_datafile = 'mnist_overlap4pix_nodup_50fold_36_test.npz'
        
        # 2 mnist digits on a 100*100 canvas with 6 pieces of clutter 
        elif args.task == 'multimnist_cluttered': 
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_cluttered2o6c_3fold_100_trainval.pt'
            test_datafile = 'mnist_cluttered2o6c_3fold_100_test.pt'
            
        # 3 mnist digits without category duplicate on a 100*100 canvas
        elif args.task == 'multimnist-3o':
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_100x100_multi3_nodup_train.pt'
            test_datafile = 'mnist_100x100_multi3_nodup_test.pt'
                    
        # 1 mnist digit on a 100*100 canvas with 8 pieces of clutter 
        elif args.task == 'mnist_cluttered_100x100': 
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_100x100_cluttered1o8c_train.pt'
            test_datafile = 'mnist_100x100_cluttered1o8c_test.pt'  
            
        # 1 mnist digit on a 60*60 canvas with 4 pieces of clutter
        elif args.task == 'mnist_cluttered_60x60': 
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_60x60_cluttered1o4c_train.pt'
            test_datafile = 'mnist_60x60_cluttered1o4c_test.pt'            
            
        elif args.task == 'multisvhn': 
            data_root = args.data_dir + '/svhn-data/'
            train_datafile = 'multisvhn_train_14classes.pt'
            test_datafile = 'multisvhn_test_14classes.pt'
            
        # SVRT      
        elif args.task == 'svrt_task1': 
            data_root = args.data_dir + 'svrt_dataset/'
            
            dataset_names_train = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]
        
            #dataset_names_val = dataset_names_train
        
            dataset_names_val = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]        
                            
            dataset_names_test = [
                            'svrt_task1',
                            'svrt1_irregular',
                            'svrt1_regular',
                            'svrt1_open',
                            'svrt1_wider_line',
                            'svrt1_scrambled',
                            'svrt1_filled',
                            'svrt1_lines',
                            'svrt1_arrows',
                            # 'svrt1_rectangles',
                            # 'svrt1_straight_lines',
                            # 'svrt1_connected_squares',
                            # 'svrt1_connected_circles'
                            ]
                            
            train_size_each = int(54000/len(dataset_names_train))  # 54000
            val_size_each = int(10800/len(dataset_names_train))
            test_size_each = int(5400/len(dataset_names_test))
            multi_task = 0 # this means include the datasets that are in the test set but put their same-diff error to zero
            
            tensor_train_ims = []
            tensor_train_ys = []
            loss_w_train = []
            # if (0): # only use the orig svrt task 1 as auxilary data
                # train_datafile = _1110_train.pt'
                # test_datafile = 'svrt_task1_1110_test.pt' 
                # tensor_trainval_ims, tensor_trainval_ys = torch.load(data_root+train_datafile)
                # tensor_train_ims.append(tensor_trainval_ims[:10000])
                # tensor_train_ys.append(tensor_trainval_ys[:10000])    

                
            if train: # load train dataset

                # Train
                for d in range(len(dataset_names_train)):
                    
                    train_datafile = dataset_names_train[d] + '_64_train.pt'
                    tensor_trainval_ims, tensor_trainval_ys = torch.load(data_root+train_datafile)
                    
                    tensor_trainval_ims = tensor_trainval_ims[:train_size_each, -args.num_classes:]
                    tensor_trainval_ys = tensor_trainval_ys[:train_size_each, -args.num_classes:]

                    if(multi_task) and (dataset_names_train[d] in dataset_names_test):

                        loss_w = torch.zeros(len(tensor_trainval_ims),1)
                        
                    else: 
                        loss_w = torch.ones(len(tensor_trainval_ims),1)
                      
                    
                    if(multi_task):
                        loss_w = torch.cat((torch.ones(len(loss_w),1), torch.ones(len(loss_w),1), 3*loss_w, 3*loss_w), dim=1)
                    else:
                        loss_w = torch.cat((torch.ones(len(loss_w),args.num_classes-2), 2*loss_w, 2*loss_w), dim=1)
                        
                        
                    tensor_train_ims.append(tensor_trainval_ims)
                    tensor_train_ys.append(tensor_trainval_ys)  
                    loss_w_train.append(loss_w)
                      
                tensor_train_ims = torch.cat(tensor_train_ims)
                tensor_train_ys = torch.cat(tensor_train_ys)    
                loss_w_train = torch.cat(loss_w_train)    
            
                idx = np.random.permutation(len(tensor_train_ims))
                tensor_train_ims, tensor_train_ys, loss_w_train = tensor_train_ims[idx], tensor_train_ys[idx], loss_w_train[idx]
            

                # Validation 
                tensor_val_ims = []
                tensor_val_ys = []
                for d in range(len(dataset_names_val)):
                    
                    train_datafile = dataset_names_val[d] + '_64_train.pt'
                    tensor_trainval_ims, tensor_trainval_ys = torch.load(data_root+train_datafile)
                    
                    tensor_trainval_ims = tensor_trainval_ims[-val_size_each:, -args.num_classes:]
                    tensor_trainval_ys = tensor_trainval_ys[-val_size_each:, -args.num_classes:]

                    # if(multi_task):

                        # if dataset_names[d] in dataset_names_test:
                            # loss_w = torch.zeros(len(tensor_trainval_ims),1)
                        
                        # else: 
                            # loss_w = torch.ones(len(tensor_trainval_ims),1)
                            
                    # loss_w = torch.cat((torch.ones(len(loss_w),1), (torch.ones(len(loss_w),1), 3*loss_w, 3*loss_w), dim=1)
                    # loss_w_train.append(loss_w)
                    tensor_val_ims.append(tensor_trainval_ims)
                    tensor_val_ys.append(tensor_trainval_ys)  
                    
                      
                tensor_val_ims = torch.cat(tensor_val_ims)
                tensor_val_ys = torch.cat(tensor_val_ys)       
            
                idx_val = np.random.permutation(len(tensor_val_ims))
                tensor_val_ims, tensor_val_ys = tensor_val_ims[idx_val], tensor_val_ys[idx_val]


                train_dataset = TensorDataset(tensor_train_ims, tensor_train_ys, loss_w_train) # create your datset
                val_dataset = TensorDataset(tensor_val_ims, tensor_val_ys) # create your datset
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                return train_dataloader, val_dataloader
                
                
                
                
            # load test dataset
            elif not train: 


                tensor_ims = []
                tensor_ys = []
                for d in range(len(dataset_names_test)):
                    
                    test_datafile = dataset_names_test[d] + '_64_test.pt'
                    tensor_test_ims, tensor_test_ys = torch.load(data_root+test_datafile)
                    
                    tensor_test_ims = tensor_test_ims[:test_size_each, -args.num_classes:]
                    tensor_test_ys = tensor_test_ys[:test_size_each, -args.num_classes:]

                    # if(multi_task):

                        # if dataset_names[d] in dataset_names_test:
                            # loss_w = torch.zeros(len(tensor_trainval_ims),1)
                        
                        # else: 
                            # loss_w = torch.ones(len(tensor_trainval_ims),1)
                            
                    # loss_w = torch.cat((torch.ones(len(loss_w),1), (torch.ones(len(loss_w),1), 3*loss_w, 3*loss_w), dim=1)
                    # loss_w_train.append(loss_w)
                    tensor_ims.append(tensor_test_ims)
                    tensor_ys.append(tensor_test_ys)  
                    
                      
                tensor_test_ims = torch.cat(tensor_ims)
                tensor_test_ys = torch.cat(tensor_ys)       
            
                idx_test = np.random.permutation(len(tensor_test_ims))
                tensor_test_ims, tensor_test_ys = tensor_test_ims[idx_test], tensor_test_ys[idx_test]

                test_dataset = TensorDataset(tensor_test_ims,tensor_test_ys) # create your datset
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                
                return test_dataloader