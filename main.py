import csv
import time

import pandas as pd

import workload
import sys
from tensorflow.python.client import device_lib
from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.csv_handler import CSVHandler



import tensorflow as tf
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    selected_workload = sys.argv[1]
    track_mode = sys.argv[2]
    if track_mode == 'codecarbon':
        workload_algorithm = workload.Workload()
        workload_algorithm.compute_workload_consumption(selected_workload)
    else:
        csv_handler = CSVHandler('result.csv')
        workload_algorithm = workload.Workload()
        with EnergyContext(handler=csv_handler,  start_tag=selected_workload) as ctx:
            workload_algorithm.compute_workload_consumption(selected_workload, cc=True)

        csv_handler.save_data()

        end_timestamp = time.time()
        df = pd.read_csv('result.csv', sep=";")
        samples_number = int((end_timestamp - float(df['timestamp'][0]))//(5*60))
        consumption_dict = [{'total_consumption': float(df['nvidia_gpu_0'][0])/3600000} for _ in range(samples_number)]
        keys = consumption_dict[0].keys()
        with open(f"{selected_workload}_consumption.csv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(consumption_dict)






