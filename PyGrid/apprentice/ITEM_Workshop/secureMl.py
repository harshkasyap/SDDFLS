import syft as sy
import torch as th
import torch.nn.functional as F
from syft.workers.node_client import NodeClient

hook = sy.TorchHook(th)

sy.hook.local_worker.is_client_worker = False # We need to set up this flag to save plans states during model's build.

# Your model needs to extends syft plans
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__(id="convnet") # ID used to identify this model 
        self.fc1 = th.tensor([2.0, 4.0])
        self.bias = th.tensor([1000.0])

    def forward(self, x):
        return self.fc1.matmul(x) + self.bias

# Create an instance of it.
model = Net()

# Build model's plan sending any valid input (input's shape needs to match with model's dimensions)
model.build(th.tensor([1.0, 2]))

# We'll use the grid platform as a cloud service.
cloud_grid_service = sy.PublicGridNetwork(hook, "http://172.28.0.2:5000")

'''
This method will split your model weights into pieces,
distributing them through the grid network,
and storing a pointer plan that manages all remote references.
'''
cloud_grid_service.serve_model(model,id=model.id,allow_remote_inference=True, mpc=True) # If mpc flag is False, It will host a unencrypted model.

# Private user data that needs to be protected
user_input_data = th.tensor([5.0, 3])

'''
This method will search the desired encrypted model,
split your private data and send their slices
to the same nodes that stores the mpc model weights,
perform a distributed computing between mpc weights and mpc input data,
receive the mpc results and aggregate it, returning the inference's result.
'''
result = cloud_grid_service.run_remote_inference("convnet", user_input_data, mpc=True)# If mpc flag is False, It will send your real data to the platform.
print("Inference's result: ", result) # ( [2.0, 4.0] * [5.0, 3.0] ) + [1000] = [1022]