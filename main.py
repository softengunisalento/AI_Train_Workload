import workload
import sys
from tensorflow.python.client import device_lib
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
print(device_lib.list_local_devices())
if __name__ == '__main__':
    selected_workload = sys.argv[1]
    csv_handler = CSVHandler('result.csv')
    workload_algorithm = workload.Workload(measure_power_secs=5*60)
    with EnergyContext(handler=csv_handler, domains=[RaplPackageDomain(1), NvidiaGPUDomain(0)], start_tag='foo') as ctx:
        workload_algorithm.compute_workload_consumption(selected_workload)

    csv_handler.save_data()





