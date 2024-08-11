import os
import torch
import logging
import argparse
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.datasets as datasets

from Code.Train.Train import learn
from Code.Model.VIT import get_model
from Code.Utils.Logging import setup_logging
from Code.Utils.GpuCheck import check_available_gpus
from Code.DataSet.Preprocess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset


def train(rank, num_gpus, train_dir, test_dir, preporcess_dir, weight_path,
         model_type = 'vit_base_patch16_224',
         preprocess_local = False, 
         batch_size = 128, img_size = 224, num_patches = 196, embed_dim = 768, 
         to_size = (8, 8, 3),
         num_classes = 1000, use_qdt = True):
    
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank}')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    if use_qdt:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])

        if (preprocess_local == False): 
            train_set = ImageNetDataset(root_dir=train_dir, transform=transform, 
                                    img_size = img_size, to_size = to_size, 
                                    num_patches = num_patches)
            
            test_set = ImageNetDataset(root_dir=test_dir, transform=transform, 
                                    img_size = img_size, to_size = to_size, 
                                    num_patches = num_patches)
        else:
            preprocess_image(train_dir, preporcess_dir + f"{num_patches}_{to_size}/train/", 
                            img_size = img_size, 
                            to_size = to_size, fixed_length = num_patches)

            preprocess_image(test_dir,  preporcess_dir + f"{num_patches}_{to_size}/test/", 
                            img_size = img_size, to_size = to_size, fixed_length = num_patches)
            
            train_set = ImageNetDataset(root_dir= preporcess_dir + f"{num_patches}_{to_size}/train/", 
                                        transform=transform, preprocess_local = True, img_size = img_size, 
                                    to_size = to_size, num_patches = num_patches)
            
            test_set = ImageNetDataset(root_dir= preporcess_dir + f"{num_patches}_{to_size}/test/", 
                                    transform=transform, img_size = img_size, to_size = to_size, 
                                    num_patches = num_patches)

    else:
        train_set = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        test_set = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]))

    weight_path = weight_path + f'img_size_{img_size}_num_patches_{num_patches}_use_qdt_{use_qdt}.pth'

    model = get_model(model_type, num_classes, num_patches, embed_dim, to_size, use_qdt)
    model = model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    
    if (num_gpus > 1):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_dl = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_dl = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler)

        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model = learn(model = model, train_dl = train_dl, test_dl = test_dl, weight_path = weight_path,
                num_epochs = 200, optimizer = optimizer, 
                criterion = criterion, scheduler = scheduler,
                rank = rank, device = device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patchify dataset.')
    parser.add_argument('--batch_size', type=int,  default=256, help='Batch Size.')
    parser.add_argument('--img_size', type=int,  default=224, help='Image Size.')
    parser.add_argument('--num_patches', type=int,  default=196, help='Number of Patches.')
    args = parser.parse_args()

    batch_size = args.batch_size
    img_size =  args.img_size
    num_patches = args.num_patches
    use_qdt = False

    train_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
    test_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/val/"

    preporcess_dir = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/"
    weight_path = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Weight/"
    log_path = f"/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Log/img_size_{img_size}_num_patches_{num_patches}_use_qdt_{use_qdt}.log"

    setup_logging(log_path)

    num_gpus = check_available_gpus()

    if (num_gpus > 1):
        args.world_size = int(os.environ['SLURM_NTASKS'])

        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
        os.environ['MASTER_PORT'] = "29500"
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['RANK'] = os.environ['SLURM_PROCID']

        print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], 
                                                        os.environ['MASTER_PORT'], 
                                                        os.environ['WORLD_SIZE'], 
                                                        os.environ['RANK'],
                                                        local_rank))
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=args.world_size,                              
            rank=int(os.environ['RANK'])                                               
        )

        print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

        print(f"Start running basic DDP example on rank {local_rank}.")
        device_id = local_rank % torch.cuda.device_count()
        train(device_id, num_gpus, train_dir, test_dir, preporcess_dir, weight_path, 
            batch_size = batch_size, img_size = img_size, num_patches = num_patches, use_qdt = use_qdt)

        dist.destroy_process_group()
    else:
        torch.cuda.empty_cache()
        train(0, num_gpus, train_dir, test_dir, preporcess_dir, weight_path, 
            batch_size = batch_size, img_size = img_size, num_patches = num_patches)