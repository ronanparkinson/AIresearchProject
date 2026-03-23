from simulator.datasetLoader import datasetLoader

class workloadSimulator:
    def __init__(self, filepath, workloadData="average_usage"):
        self.loader = datasetLoader(filepath, workloadData=workloadData)

    def nextLoad(self):
        load = self.loader.nextLoad()

        cpuUsage = min(load, 1.0)
        memUsage = min(load * 0.8, 1.0)

        return {"cpu": cpuUsage, "memory": memUsage}

    def reset(self):
        self.loader.reset()
