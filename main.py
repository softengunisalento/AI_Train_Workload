import workload
import sys

if __name__ == '__main__':
    f = open("demofile3.txt", "w")
    f.write("Woops! I have deleted the content!")
    f.close()
    # selected_workload = sys.argv[1]
    # workload_algorithm = workload.Workload(measure_power_secs=15)
    # workload_algorithm.compute_workload_consumption(selected_workload)
