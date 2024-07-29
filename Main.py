import os
import torch
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
import logging
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Code.DataSet.Preprocess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset
from Code.Model.VIT import get_model
from Code.Train.Train import learn
from Code.Utils.Logging import setup_logging

def check_available_gpus():
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available.")
        else:
            print(f"Number of available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        return num_gpus
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure PyTorch is installed with ROCm support.")

def init_process(rank, num_gpus, root_dir, preporcess_dir, preprocess_local, 
                 batch_size, weight_path,
                 train_fn, log_path, backend='nccl'):

    setup_logging(log_path)
    
    logging.info('-' * 8 + f"Init Process at Rank {rank}" + '-' * 8)
    
    os.environ['MASTER ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend, rank=rank, world_size=num_gpus)
    train_fn(rank, num_gpus, root_dir, preporcess_dir, 
             preprocess_local = preprocess_local, batch_size = batch_size, weight_path = weight_path)

def train(rank, num_gpus, root_dir, preporcess_dir, weight_path,
         model_type = 'vit_base_patch16_224',
         preprocess_local = False, 
         batch_size = 32, img_size = 224, num_patches = 196, embed_dim = 768, 
         to_size = (8, 8, 3),
         num_classes = 1000):
    
    torch.manual_seed(0)
    logging.info('-' * 8 + "Training Start" + '-' * 8)

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
        logging.info('-' * 8 + "Preprocess images to local" + '-' * 8)
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
        sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, sampler=sampler)
        model = DDP(model, device_ids=[rank])
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = learn(model = model, dataloader = dataloader, weight_path = weight_path,
                num_epochs = 10, optimizer = optimizer, criterion = criterion, device = device)


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
    logging.info('-' * 8 + "Logging File Created" + '-' * 8)

    num_gpus = check_available_gpus()
    mp.set_start_method('spawn')

    logging.info('-' * 8 + "Set Start Method Spawn" + '-' * 8)

    preprocess_local = False

    if (num_gpus > 1):
        processes = []
        for rank in range(num_gpus):
            p = torch.multiprocessing.Process(target=init_process, 
                                              args=(rank, num_gpus, 
                                                    root_dir, preporcess_dir, preprocess_local,
                                                    batch_size, weight_path,
                                                    train, log_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        train(0, num_gpus, root_dir, preporcess_dir, weight_path,
         model_type = 'vit_base_patch16_224',
         preprocess_local = False, 
         batch_size = 32, img_size = 224, num_patches = 196, embed_dim = 768, 
         to_size = (8, 8, 3),
         num_classes = 1000)






