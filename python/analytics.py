import os
import sys
import dask
import h5py
import json
import yaml
import time
from dask_ml.decomposition import PCA
import dask.dataframe
import dask.array as da
from deisa.dask import Deisa, get_connection_info
from distributed import Variable, Queue, get_client


gys_io_config = sys.argv[1]
with open(gys_io_config, 'r') as config_file:
    nb_iter = yaml.safe_load(config_file)["Application"]["n_iterations"]

with open("scheduler.json", 'r') as scheduler_file:
    dask_addr = json.load(scheduler_file)["address"]

print(f"[Deisa] Start connection of Deisa to {dask_addr}\n")
deisa = Deisa(get_connection_info=lambda: get_connection_info(dask_addr))
print("[Deisa] Connected\n")

global end_flag
end_flag = False

def sum_moments(density, velocity, temperature, timestep):
    sum_density = density[0].sum().compute() 
    sum_velocity = velocity[0].sum().compute() 
    sum_temperature = temperature[0].sum().compute() 

    print(f"[Deisa] Iteration {timestep}")
    print(f"[Deisa] sum density {sum_density}\n")
    print(f"[Deisa] sum velocity {sum_velocity}\n")
    print(f"[Deisa] sum temperature {sum_temperature}\n")

    if timestep == nb_iter-1:
        with h5py.File("deisa_fluid_moments.h5", 'w') as f:
            f.create_dataset("density", data=density[0])
            f.create_dataset("mean_velocity", data=velocity[0])
            f.create_dataset("temperature", data=temperature[0])

        global end_flag
        end_flag = True


deisa.register_sliding_window_callbacks(
        sum_moments,
        ("density", 1),
        ("velocity", 1),
        ("temperature", 1),
        when='AND'
        )

while not end_flag:
    time.sleep(1)

deisa.set("analytics_over", True)
deisa.close()

