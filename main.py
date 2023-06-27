import csv
import time

import pandas as pd
import nvidia_smi
import workload
import sys
from tensorflow.python.client import device_lib
from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.csv_handler import CSVHandler


nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
import tensorflow as tf
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#     selected_workload = sys.argv[1]
#     csv_handler = CSVHandler('result.csv')
#     workload_algorithm = workload.Workload(measure_power_secs=5*60)
#     with EnergyContext(handler=csv_handler,  start_tag=selected_workload) as ctx:
#         workload_algorithm.compute_workload_consumption(selected_workload)
#
#     csv_handler.save_data()
#
#     end_timestamp = time.time()
#     df = pd.read_csv('result.csv', sep=";")
#     samples_number = int((end_timestamp - float(df['timestamp'][0]))//(5*60))
#     consumption_dict = [{'total_consumption': float(df['nvidia_gpu_0'][0])/3600000} for _ in range(samples_number)]
#     keys = consumption_dict[0].keys()
#     with open(f"{selected_workload}_consumption.csv", 'w') as f:
#         dict_writer = csv.DictWriter(f, keys)
#         dict_writer.writeheader()
#         dict_writer.writerows(consumption_dict)






