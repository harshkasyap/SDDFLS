import syft as sy
import torch as th
from syft.workers.node_client import NodeClient

hook = sy.TorchHook(th)

# Bob's House
# ==============================
bob_home_assistant = NodeClient(hook, "http://bob:3000/")

# smart light bulb
lb_energy_data_chunk = th.tensor([ 5.0, 8.5, 9.7, 2.5]).tag("#energy-consumption", "#light-bulb").describe("light bulb energy consumption(Kwatts)")

# smart coffe pot
coffe_energy_data_chunk = th.tensor([2.5, 3.7, 1.2, 1.0]).tag("#energy-consumption", "#coffe-pot").describe("coffe pot energy consumption(Kwatts)")

# smart fridge
fridge_energy_data_chunk = th.tensor([8.0, 4.9, 7, 10.9]).tag("#energy-consumption", "#fridge").describe("Fridge energy consumption(Kwatts)")


# Sending to home assistant
lb_energy_data_chunk.send(bob_home_assistant, garbage_collect_data=False)
coffe_energy_data_chunk.send(bob_home_assistant, garbage_collect_data=False)
fridge_energy_data_chunk.send(bob_home_assistant, garbage_collect_data=False)

# Alice's House
# ==============================
alice_home_assistant = NodeClient(hook, "http://alice:3001/")

# smart light bulb
lb_energy_data_chunk = th.tensor([ 3.0, 2.5, 6.7, 4.5]).tag("#energy-consumption", "#light-bulb").describe("light bulb energy consumption(Kwatts)")

# smart coffe pot
coffe_energy_data_chunk = th.tensor([0.5, 1.7, 5.2, 1.0]).tag("#energy-consumption", "#coffe-pot").describe("coffe pot energy consumption(Kwatts)")

# smartfridge
fridge_energy_data_chunk = th.tensor([3.0, 4.9, 8, 5.9]).tag("#energy-consumption", "#fridge").describe("Fridge energy consumption(Kwatts)")


# Sending to home assistant
lb_energy_data_chunk.send(alice_home_assistant , garbage_collect_data=False)
coffe_energy_data_chunk.send(alice_home_assistant , garbage_collect_data=False)
fridge_energy_data_chunk.send(alice_home_assistant, garbage_collect_data=False)


# Bill's House
# ==============================
bill_home_assistant = NodeClient(hook, "http://bill:3002/")

# smart light bulb
lb_energy_data_chunk = th.tensor([ 8.0, 7.5, 9.7, 2.5]).tag("#energy-consumption", "#light-bulb").describe("light bulb energy consumption(Kwatts)")

# smartfridge
fridge_energy_data_chunk = th.tensor([3.7, 4.3, 8, 5.9]).tag("#energy-consumption", "#fridge").describe("Fridge energy consumption(Kwatts)")

# Sending to home assistant
lb_energy_data_chunk.send(bill_home_assistant, garbage_collect_data=False)
fridge_energy_data_chunk.send(bill_home_assistant, garbage_collect_data=False)


# Gateway (Owner)
# ==============================
my_smart_grid = sy.PublicGridNetwork(hook, "http://gateway:5000")

# Energy spent with fridge
# ==============================
results = my_smart_grid.search("#energy-consumption")
#print("Results: ", results)


# Get energy expenditure average
# ==============================
from functools import reduce

def sum_energy_spent_by_home(home_id, query_dict):
    # It will aggregate home appliances' spent remotelly.
    # Example: If we have light-bulb and fridge at the same house, we need to aggregate it on the user's device.
    return reduce(lambda x,y: x + y, query_dict[home_id])

def smart_city_aggregation(x,y):
    print("Sending X(", x.location, ") to Y(", y.location, ").")
    x.location.connect_nodes(y.location)
    return x.move(y.location) + y

energy_spent_by_houses = [ sum_energy_spent_by_home(home_id, results) for home_id in results.keys() ]
total_spend = reduce(lambda x, y: smart_city_aggregation(x,y), energy_spent_by_houses)

p_average = total_spend / 3 # Total spent divided by number of houses
average = p_average.get()
print(average)
