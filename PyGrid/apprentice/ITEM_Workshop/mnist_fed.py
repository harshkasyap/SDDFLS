import os
from os import path
import logging
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.utils import save_image

import syft as sy
from syft.workers.node_client import NodeClient
from syft.frameworks.torch.fl import utils

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker
from syft.frameworks.torch.fl import utils

from skimage.util import random_noise

logger = logging.getLogger(__name__)
LOG_INTERVAL = 25

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Arguments():
    def __init__(self):
        self.batch_size = 32
        self.test_batch_size = 1000
        self.epochs = 10
        self.federate_after_n_batches = 50
        self.lr = 0.01
        self.cuda = False
        self.seed = 1
        self.save_model = False
        self.tb = SummaryWriter(comment="Mnist Federated training")

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

def train_on_batches(worker, batches, model_in, device, lr, tb):
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug("Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(worker.id, batch_idx, len(batches), 100.0 * batch_idx / len(batches), loss.item(),))

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    
    model.get()  # <-- NEW: get the model back
    return model, loss

def get_next_batches(fdataloader: sy.FederatedDataLoader, federate_after_n_batches: int):
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(federate_after_n_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches

def train(
    model, device, federated_train_loader, lr, federate_after_n_batches, tb, epoch
):
    model.train()
    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, federate_after_n_batches)
    counter = 0
    local_epoch = 0

    while True:
        logger.debug(f"Starting training round, batches [{counter}, {counter + federate_after_n_batches}]")
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, lr, tb
                )
                #tb.add_scalar(worker.id + ' Training Loss', loss_values[worker], local_epoch)
                tb.add_scalars("Training Loss/" + worker.id, {str(epoch): loss_values[worker]}, local_epoch)
            else:
                data_for_all_workers = False
        local_epoch += 1
        counter += federate_after_n_batches
        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, federate_after_n_batches)
    return model

def test(model, device, test_loader, tb, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    tb.add_scalar('Testing Loss', test_loss, epoch)

    logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    tb.add_scalar('Accuracy', accuracy, epoch)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

def main():
    hook = sy.TorchHook(torch)

    # Connect directly to grid nodes
    virtual_nodes = ["http://bob:3000/",
                     "http://alice:3001/",
                     "http://bill:3002/",
                     "http://james:3003/",
                     "http://william:3004/",
                     "http://eve:3005/"]

    virtual_workers = ['bob', 'alice', 'bill', 'james', 'william', 'eve']

    workers = []
    if True:
        for virtual_worker in virtual_workers:
            workers.append( sy.VirtualWorker(hook, id=virtual_worker) )
    else:
        for virtual_node in virtual_nodes:
            workers.append( NodeClient(hook, virtual_node) )

    args = Arguments()

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


    """    
    Prepare perturbed Data adding Gaussian noise
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./dataset",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ), 
        batch_size=1,
        shuffle=True
    )

    for index, data in enumerate(trainloader):
        img, target = data[0], data[1]
        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
        gauss_img = gauss_img.view(1, 28, 28)
        dirPath = os.path.join('./dataset/MNIST/Gaussian/', str(data[1].item()))
        if (not path.exists(dirPath)):
            os.mkdir(dirPath)
        save_image(gauss_img, dirPath + '/' + str(index) + '.png')
    """


    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST(
            "./dataset",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), 
                 transforms.Normalize((0.1307,), (0.3081,)),
                 AddGaussianNoise(1., 2.)]
            ),
        ).federate(tuple(workers)),
        batch_size=args.batch_size,
        shuffle=True,
        iter_per_worker=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./dataset",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches, args.tb, epoch)
        test(model, device, test_loader, args.tb, epoch)
    
    args.tb.close()

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()