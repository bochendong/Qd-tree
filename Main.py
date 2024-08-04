import os
import torch
import logging
import argparse
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Code.Train.Train import learn
from Code.Model.VIT import get_model
from Code.Utils.Logging import setup_logging
from Code.Utils.GpuCheck import check_available_gpus
from Code.DataSet.Preprocess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset


def train(rank, num_gpus, root_dir, preporcess_dir, weight_path,
         model_type = 'vit_base_patch16_224',
         preprocess_local = False, 
         batch_size = 128, img_size = 224, num_patches = 196, embed_dim = 768, 
         to_size = (8, 8, 3),
         num_classes = 1000):
    
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank}')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preporcess_dir = preporcess_dir + f"{num_patches}_{to_size}/train/"
    weight_path = weight_path + f'img_size_{img_size}_num_patches_{num_patches}.pth'

    if (preprocess_local == False): 
        dataset = ImageNetDataset(root_dir=root_dir, transform=transform, 
                                  img_size = img_size, to_size = to_size, 
                                  num_patches = num_patches)
    else:
        logging.info('-' * 8 + f"Device {rank}, Preprocess images to local" + '-' * 8)
        preprocess_image(root_dir, preporcess_dir, img_size = img_size, 
                         to_size = to_size, fixed_length = num_patches)
        
        dataset = ImageNetDataset(root_dir=preporcess_dir, transform=transform, 
                                  preprocess_local = True, img_size = img_size, 
                                  to_size = to_size, num_patches = num_patches)

    model = get_model(model_type, num_classes, num_patches, embed_dim, to_size)
    model = model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    
    if (num_gpus > 1):
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        logging.info('-' * 8 + f"Device {rank} Dataloader created" + '-' * 8)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model = learn(model = model, dataloader = dataloader, weight_path = weight_path,
                num_epochs = 200, optimizer = optimizer, 
                criterion = criterion, scheduler = scheduler,
                rank = rank, device = device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patchify dataset.')
    parser.add_argument('--batch_size', type=int,  default=128, help='Batch Size.')
    parser.add_argument('--img_size', type=int,  default=224, help='Image Size.')
    parser.add_argument('--num_patches', type=int,  default=196, help='Number of Patches.')
    args = parser.parse_args()

    batch_size = args.batch_size
    img_size =  args.img_size
    num_patches = args.num_patches

    root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
    preporcess_dir = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/"
    weight_path = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Weight/"
    log_path = f"/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Log/img_size_{img_size}_num_patches_{num_patches}.log"

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
        train(device_id, num_gpus, root_dir, preporcess_dir, weight_path, 
            batch_size = batch_size, img_size = img_size, num_patches = num_patches)

        dist.destroy_process_group()
    else:
        torch.cuda.empty_cache()
        train(0, num_gpus, root_dir, preporcess_dir, weight_path, 
            batch_size = batch_size, img_size = img_size, num_patches = num_patches)