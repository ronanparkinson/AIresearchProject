from evaluation.evaluationAgents import evaluationAgents
from evaluation.ppoEvaluation import ppoEvaluation

def results(title, results):
    print(f"\n{title}")
    print("-" * len(title))

    for key, value in results.items():
        print(f"{key}: {value}")

def avgResults(resultsList):
    keys = resultsList[0].keys()
    avgs = {}

    for k in keys:
        avgs[k] = sum(result[k] for result in resultsList) / len(resultsList)

    return avgs

if __name__ == "__main__":
    rewardVersion = ["v1", "v2", "v3"]
    numRuns = 5

    for m in rewardVersion:
        print(f"Reward Version: {m}")

        rbRuns = []
        ppoRuns = []

        for i in range(numRuns):
            print(f"Run {i+1}/{numRuns}")

            rbRuns.append(evaluationAgents(rewardVersion=m))
            ppoEvaluate = ppoEvaluation(rewardVersion=m)
            ppoRuns.append(ppoEvaluate.run())

        avgRBResults = avgResults(rbRuns)
        avgPPOResults = avgResults(ppoRuns)

        results("Rule based results: ", avgRBResults)
        results("ppo results: ", avgPPOResults)
