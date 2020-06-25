import syft as sy
from syft.workers.node_client import NodeClient
import torch
#import pickle
#import time
import torchvision
from torchvision import datasets, transforms
#import tqdm

# Setup config
# Init hook, connect with grid nodes, etc...
# ==============================
hook = sy.TorchHook(torch)

# Connect directly to grid nodes
nodes = ["http://bob:3000/",
         "http://alice:3001/"]

compute_nodes = []
for node in nodes:
    compute_nodes.append( NodeClient(hook, node) )

# 1 - Load Dataset
# The code below will load and preprocess an N amount of MNIST data samples.
# ==============================
N_SAMPLES = 10000
MNIST_PATH = './dataset'

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                              ])

trainset = datasets.MNIST(MNIST_PATH, download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_SAMPLES, shuffle=False)
dataiter = iter(trainloader)
images_train_mnist, labels_train_mnist = dataiter.next()

# 2 - Split dataset
# We will split our dataset to send to nodes.
# ==============================
datasets_mnist = torch.split(images_train_mnist, int(len(images_train_mnist) / len(compute_nodes)), dim=0 ) #tuple of chunks (dataset / number of nodes)
labels_mnist = torch.split(labels_train_mnist, int(len(labels_train_mnist) / len(compute_nodes)), dim=0 )  #tuple of chunks (labels / number of nodes)


#3 - Tagging tensors
#The code below will add a tag (of your choice) to the data that will be sent to grid nodes. This tag is important as the gateway will need it to retrieve this data later.
# ==============================
tag_img = []
tag_label = []

for i in range(len(compute_nodes)):
    tag_img.append(datasets_mnist[i].tag("#X", "#mnist", "#dataset").describe("The input datapoints to the MNIST dataset."))
    tag_label.append(labels_mnist[i].tag("#Y", "#mnist", "#dataset").describe("The input labels to the MNIST dataset."))

# 4 - Sending our tensors to grid nodes
# ==============================
shared_x1 = tag_img[0].send(compute_nodes[0]) # First chunk of dataset to Bob
shared_x2 = tag_img[1].send(compute_nodes[1]) # Second chunk of dataset to Alice

shared_y1 = tag_label[0].send(compute_nodes[0]) # First chunk of labels to Bob
shared_y2 = tag_label[1].send(compute_nodes[1]) # Second chunk of labels to Alice

print("X tensor pointers: ", shared_x1, shared_x2)
print("Y tensor pointers: ", shared_y1, shared_y2)


# Disconnect nodes
# ==============================
for i in range(len(compute_nodes)):
    compute_nodes[i].close()


# PART-II
# ==============================
from syft.grid.public_grid import PublicGridNetwork
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

hook = sy.TorchHook(th)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

if(th.cuda.is_available()):
    th.set_default_tensor_type(th.cuda.FloatTensor)
    
model = Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
my_grid = PublicGridNetwork(hook, "http://gateway:5000")

data = my_grid.search("#X", "#mnist", "#dataset")
target = my_grid.search("#Y", "#mnist", "#dataset")
data = list(data.values())
target = list(target.values())

def epoch_total_size(data):
    total = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            total += data[i][j].shape[0]
            
    return total

N_EPOCS = 3
SAVE_MODEL = True
SAVE_MODEL_PATH = './models'

def train(epoch):
    model.train()
    epoch_total = epoch_total_size(data)
    current_epoch_size = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            current_epoch_size += len(data[i][j])
            worker = data[i][j].location
            model.send(worker)
            optimizer.zero_grad()
            pred = model(data[i][j])
            loss = criterion(pred, target[i][j])
            loss.backward()
            optimizer.step()
            model.get()
            loss = loss.get()
            print('Train Epoch: {} | With {} data |: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, worker.id, current_epoch_size, epoch_total,
                            100. *  current_epoch_size / epoch_total, loss.item()))
                    
for epoch in range(N_EPOCS):
    train(epoch)