import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2

# [Tr] Cuda kullanabilmek için
# [En]To be able to use Cuda
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)
# [Tr]glob.globu kullanark png uzantlı tüm dosyaları içeri aktarıldı.
# [En] Imported all files with png extension using glob.glob.

test_images = glob.glob("./archive/test/*.png")
train_images = glob.glob('./archive/train/*.png')

print(f"Test img : {len(test_images)}  , Train img : {len(train_images)}")
# [Tr] tensöre çevirme işlemi, compose metodunu kullanarak pek çok trasform işlemini yapmak mümkün
# [En]It is possible to perform several transform operations using the compose method.
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((255,255))#128x128 orginal size
])


# dataset and dataloader for train
# pythoch data load req.
class dataset(Dataset):
    def __init__(self, image_list, transform, device):
        self.image_list = image_list
        self.transform = transform
        # [Tr]Resimlerin isimlerinin sondan 6. karakteri ile sondan 4. kısımı resimleri labellamak için kullanıyorum
        # [En] I use the 6th character from the end of the names of the images and the 4th character from the end to label the images
        self.image_labels = [x[-6:-4] for x in self.image_list]
        for i, x in enumerate(self.image_labels):
            label = int(x[0])
            # [Tr]eğer "R" yi görürse classes olarak 6 ekleyerek sağ el için olan classes kısını yapıyoruz
            # örnek sol 1 yani L1=1 , sağ 1 R1 ise 1+6=7 index olark gösterildi toplamda 12 classes var
            # [En]If it sees "R", we add 6 as classes and make the classes part for the right hand
            # example left 1 i.e. L1=1, right 1 R1 is shown as index 1+6=7, there are 12 classes in total.
            if x[1] == 'R':
                label += 6
            self.image_labels[i] = label
        self.device = device
        self.image_labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        x = cv2.imread(self.image_list[index])
        x = self.transform(x).to("cuda")

        y = self.image_labels[index]
        y = torch.LongTensor([y,]).to(self.device)

        return x, y


# load train_images
train_data = dataset(train_images, transform, device)
train_data = DataLoader(train_data, batch_size=32, shuffle=True)

# load test_images
test_data = dataset(test_images, transform, device)
test_data = DataLoader(test_data, batch_size=32, shuffle=True)
