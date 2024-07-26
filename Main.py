from Code.DataSet.Proprecess import preprocess_image
from Code.DataSet.ImageNetDataSet import ImageNetDataset
from torchvision import transforms
from torch.utils.data import DataLoader

to_size = (8, 8, 3)
fixed_length = 1024

root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
preporcess_dir = f"/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/{fixed_length}_{to_size}/train/"


# preprocess_image(root_dir, preporcess_dir, to_size = to_size, fixed_length = fixed_length)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageNetDataset(root_dir=preporcess_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for i, data in enumerate(dataloader):
    print(data[0].size())
    print(data[1].size())                    


