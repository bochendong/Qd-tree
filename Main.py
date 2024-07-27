import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader


from Code.DataSet.Proprecess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset
from Code.Model.VIT import get_model
from Code.Train.Train import train

def main(root_dir, preporcess_dir, 
         model_type = 'vit_base_patch16_224',
         preprocess_local = True, 
         batch_size = 32, img_size = 224, num_patches = 196, embed_dim = 768, 
         to_size = (8, 8, 3),
         num_classes = 1000):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preporcess_dir = preporcess_dir + f"{num_patches}_{to_size}/train/"

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if (model_type == 'vit_base_patch16_224'):
        model = get_model(model_type, num_classes, num_patches, embed_dim, to_size)
        model = model.to(device)
    else:
        print("Invalid model Type")
        return 

    # Processed Image Shape Test 
    for i, data in enumerate(dataloader):
        print("Processed Image Size:", data[0].size())
        print("Processed Label Size:", data[1].size())

        out = model(data[0])
        print("Model Output Test:", out.size())
        break

    # Vit Model Test

    '''
    model = get_vit_model('vit_base_patch16_224', len(dataset.classes))    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train(model = model, dataloader = dataloader, 
                num_epochs = 10, optimizer = optimizer, criterion = criterion, device = device)'''


root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
preporcess_dir = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/"
main(root_dir, preporcess_dir, preprocess_local = False)





