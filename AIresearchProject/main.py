from simulator.workloadSimulator import workloadSimulator

sim = workloadSimulator("data/borg_traces_data.csv", workloadData="average_usage")

for _ in range(10):
    print(sim.nextLoad())

    #-----------testing below, remove after----------------

from simulator.datasetLoader import datasetLoader
from simulator.workloadSimulator import workloadSimulator

loader = datasetLoader("data/borg_traces_data.csv", workloadData="average_usage")

print("Chosen workload column:", loader.workloadCol)
print("First 10 normalized loads:")
for _ in range(10):
    print(loader.nextLoad())

sim = workloadSimulator("data/borg_traces_data.csv", workloadData="average_usage")

print("\nFirst 10 simulated workload values:")
for _ in range(10):
    print(sim.nextLoad())


from simulator.datasetLoader import datasetLoader

loader = datasetLoader("data/borg_traces_data.csv", workloadData="average_usage")

series = loader.data[loader.workloadCol].astype(float)

print("Min:", series.min())
print("Max:", series.max())
print("Mean:", series.mean())
print("First 20 raw values:")
print(series.head(20).tolist())