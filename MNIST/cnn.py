import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # generar barras de progreso
#%matplotlib inline  
# load data
df = pd.read_csv('sample_data/mnist_train_small.csv', header=None)
print(df.head())

print(len(df))

ix = random.randint(0, len(df)-1)
label, pixels = df.loc[ix][0], df.loc[ix][1:]
img = np.array(pixels).reshape((28,28))
print(label)
# plot on command line doesn't work
#plt.imshow(img)

labels, imgs = [], []  # Listas
for index, row in df.iterrows():
    label, pixels = row[0], row[1:]
    img = np.array(pixels).reshape((28,28))  # con CNN trabajamos con imágenes
    labels.append(label)
    imgs.append(img)
df2 = pd.DataFrame({'label': labels, 'img': imgs})
#df2 = df2[:1000]  # Escogemos solo los 1000 primeros. Hay que mirar que estan mezclados para no quedarse  solo con el mismo númreo
print(df2.head())

ix = random.randint(0, len(df2)-1)
#img = df2.loc[ix].img.reshape((28,28))
img = df2.loc[ix].img
label = df2.loc[ix].label
print(label)
# plot on command line doesn't work
#plt.imshow(img)


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df2, test_size=0.2, shuffle=True, stratify=df2.label)

print(len(train_df), len(val_df))

# plot on command line doesn't work
#val_df.plot.hist(bins=10)

# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
#platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
#cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
#accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

#pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision

import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
#print(device)


# create dataset
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
  def __init__(self, imgs, labels):
    super(MNISTDataset, self).__init__()
    self.imgs = imgs
    self.labels = labels
    
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, ix):
    img = self.imgs[ix]
    label = self.labels[ix]
    return torch.from_numpy(img).unsqueeze(0).float(), label


dataset = {
    'train': MNISTDataset(train_df.img.values, train_df.label.values),
    'val': MNISTDataset(val_df.img.values, val_df.label.values)
} 

print(len(dataset['train']), len(dataset['val']))

ix = random.randint(0, len(dataset['train'])-1)
img, label = dataset['train'][ix]
print("Shape and type of 'img'")
print(img.shape, img.dtype)
print(label)
# plot on bash doesn't work
#plt.imshow(img.squeeze())

# create model
import torch.nn as nn

def conv_block(in_f, out_f, k, p=0):
  return nn.Sequential(
    nn.Conv2d(in_f, out_f, k, padding=p),
    nn.BatchNorm2d(out_f),
    nn.ReLU(inplace=True)
  )

def conv_block_mp(in_f, out_f, k, p=0):
  return nn.Sequential(
    conv_block(in_f, out_f, k),
    conv_block(out_f, out_f, 3, p=1), 
    nn.MaxPool2d(2)
  )

def lin_block(in_f, out_f):
  return nn.Sequential(
      nn.Linear(in_f, out_f),
      nn.BatchNorm1d(out_f),
      nn.ReLU(inplace=True)
  )

class Net(nn.Module):
  def __init__(self, in_channels=1, num_classes=10):
    super(Net, self).__init__()
    
    self.conv1 = conv_block_mp(in_channels, 10, 5)
    self.conv2 = conv_block_mp(10, 20, 5)
    self.fc1 = lin_block(320, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

model = Net()
model.to(device)

input = torch.zeros((32, 1, 28, 28))
out = model(input.to(device))
print("Shape of output model")
print(out.shape)


from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, min_lr=0.0001, verbose=True)

bs = 32
dataloader = {
    'train': DataLoader(dataset['train'], batch_size=bs, shuffle=True, num_workers=4),
    'val': DataLoader(dataset['val'], batch_size=bs, shuffle=False, num_workers=4),
}


# train




# test
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.eval()

ix = random.randint(0, len(dataset['val'])-1)
img, label = dataset['val'][ix]
pred = model(img.unsqueeze(0).to(device)).cpu()
pred_label = torch.argmax(pred)
print('Ground Truth: {}, Prediction: {}'.format(label, pred_label))
plt.imshow(img.squeeze(0))


### Opencv

import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
from collections import deque
import numpy as np

coordinates={'x1':0,'y1':0,'x2':300,'y2':300}
cnn_input_size=28
num=[]


cap=cv2.VideoCapture(0)
model.eval()
while True:
    _,frame= cap.read()
    
    #preprocessing of the image
    #frame=cv2.flip(frame,1)
    mask=frame[coordinates['y1']:coordinates['y2'],coordinates['y1']:coordinates['x2']]
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    
    mask=cv2.erode(mask,kernel,iterations=3)
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask=cv2.dilate(mask,kernel,iterations=3)
    _, mask= cv2.threshold(mask,150,255,cv2.THRESH_BINARY_INV)    
    small = cv2.resize(mask, (cnn_input_size, cnn_input_size), interpolation = cv2.INTER_NEAREST)
    #small=cv2.flip(small,1)
    
    #num=torch.from_numpy(small).reshape(28,28)
    img=torch.from_numpy(np.array(small)).reshape(28,28).unsqueeze(0).float()
    #print(num)
    #plt.imshow(img.squeeze(0))

    #pred = model(img.to(device)).cpu()
    #pred_label = torch.argmax(pred)
    
    #ix = random.randint(0, len(dataset['val'])-1)
    #img, label = dataset['val'][ix]
    pred = model(img.unsqueeze(0).to(device)).cpu()
    pred_label = torch.argmax(pred)
    #print('Ground Truth: {}, Prediction: {}'.format(label, pred_label))
    #plt.imshow(img.squeeze(0))

    
    #model and prediction
    ######num=torch.from_numpy(small).view((-1,28*28)).type(torch.FloatTensor).cpu()
    ######pred = model(num.unsqueeze(0)).cpu()
	
    ##pred=model(num).cpu()
    ##_,predicted= torch.max(pred.data,1)
    ##text="prediction "+ str(max(predicted.data.tolist()))
    text=" "+ str(pred_label)
    #rectangle
    a=cv2.rectangle(frame,(coordinates['x1'],coordinates['y1']),(coordinates['x2'],coordinates['y2']),(0,255,0),3)
    
    #text to be written
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,(coordinates['x1']+10,coordinates['y2']-10), font, 2,(0,255,255),2,cv2.LINE_AA)
    
    
    cv2.imshow("frame",frame)
    cv2.imshow("mask",small)
    
    
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


