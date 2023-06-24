import workload
import sys

if __name__ == '__main__':
    selected_workload = sys.argv[1]

    workload_algorithm = workload.Workload(measure_power_secs=5*60)
    workload_algorithm.compute_workload_consumption(selected_workload)
