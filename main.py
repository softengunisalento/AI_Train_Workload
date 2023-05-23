import workload
import sys

if __name__ == '__main__':
    selected_workload = 'isolation_forrest'#sys.argv[1]
    workload_algorithm = workload.Workload()
    workload_algorithm.compute_workload_consumption(selected_workload)
