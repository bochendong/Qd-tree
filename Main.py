import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Code.DataSet.Preprocess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset
from Code.Model.VIT import get_model
from Code.Train.Train import train

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
                 train_fn, backend='nccl'):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend, rank=rank, world_size=num_gpus)
    train_fn(rank, num_gpus, root_dir, preporcess_dir, 
             preprocess_local = preprocess_local, batch_size = batch_size, weight_path = weight_path)

def train(rank, num_gpus, root_dir, preporcess_dir, weight_path,
         model_type = 'vit_base_patch16_224',
         preprocess_local = True, 
         batch_size = 32, img_size = 224, num_patches = 196, embed_dim = 768, 
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

    # Preprocess the image while load the data
    if (preprocess_local == False): 
        dataset = ImageNetDataset(root_dir=root_dir, transform=transform, 
                                  img_size = img_size, to_size = to_size, 
                                  num_patches = num_patches)
        
    # Preprocess the image and save to local
    else:
        preprocess_image(root_dir, preporcess_dir, img_size = img_size, 
                         to_size = to_size, fixed_length = num_patches)
        
        dataset = ImageNetDataset(root_dir=preporcess_dir, transform=transform, 
                                  preprocess_local = True, img_size = img_size, 
                                  to_size = to_size, num_patches = num_patches)

    sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, sampler=sampler)

    if (model_type == 'vit_base_patch16_224'):
        model = get_model(model_type, num_classes, num_patches, embed_dim, to_size)
        model = model.to(device)
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        ddp_model = DDP(model, device_ids=[rank])
    else:
        print("Invalid model Type")
        return 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    
    # Vit Model Test
    model = train(model = ddp_model, dataloader = dataloader, weight_path = weight_path,
                num_epochs = 10, optimizer = optimizer, criterion = criterion, device = device)


if __name__ == "__main__":
    num_gpus = check_available_gpus()
    mp.set_start_method('spawn')
    root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
    preporcess_dir = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/"
    weight_path = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree/Weight/"

    preprocess_local = False
    batch_size = 512

    processes = []
    for rank in range(num_gpus):
        p = torch.multiprocessing.Process(target=init_process, 
                                          args=(rank, num_gpus, 
                                                root_dir, preporcess_dir, preprocess_local,
                                                batch_size, weight_path,
                                                train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()






