import csv
import json
import os.path

import matplotlib.pyplot as plt

from evaluation.evaluationAgents import evaluationAgents
from evaluation.ppoEvaluation import ppoEvaluation

results = "evaluation/officalResults"
graphs = os.path.join(results, "graphs")

def acceptFiles():
    os.makedirs(results, exist_ok=True)
    os.makedirs(graphs, exist_ok=True)

def avgResults(resList):
    keys = resList[0].keys()
    avg = {}

    for k in keys:
        avg[k] = sum(result[k] for result in resList) / len(resList)

    return avg

def saveCSV(rows, path):
    names = [
        "reward version",
        "agent",
        "reward",
        "average queue",
        "average cpu",
        "average instances",
        "scaling actions"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)

def saveRJson(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def plotbarMetric(summeryRows, metricName, output):
    rewardVersion = sorted(list(set(r["reward version"] for r in summeryRows)))

    ruleValues = []
    ppoValues = []

    for v in rewardVersion:
        ruleRow = next(
            r for r in summeryRows
            if r["reward version"] == v and r["agent"] == "Rule-Based"
        )
        ppoRow = next(
            r for r in summeryRows
            if r["reward version"] == v and r["agent"] == "PPO"
        )

        ruleValues.append(ruleRow[metricName])
        ppoValues.append(ppoRow[metricName])

    x = list(range(len(rewardVersion)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], ruleValues, width=width, label="Rule_Based")
    plt.bar([i + width / 2 for i in x], ppoValues, width=width, label="PPO")
    plt.xticks(x, ruleValues)
    plt.xlabel("Reward Version")
    plt.ylabel(metricName.title())
    plt.title(f"{metricName.title()} comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def plotHistory(ruleHistory, ppoHistory, metricName, rewardVersion, output):
    plt.figure(figsize=(9, 5))
    plt.plot(ruleHistory[metricName], label="Rule-Based")
    plt.plot(ppoHistory[metricName], label="PPO")
    plt.xlabel("Step")
    plt.ylabel(metricName.title())
    plt.title(f"{metricName.title()} Over Time ({rewardVersion})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

if __name__ == "__main__":
    acceptFiles()

    rewardVersions = ["v1", "v2", "v3"]
    numRuns = 5
    steps = 200

    summeryRows = []
    rawResults = {}

    for v in rewardVersions:
        print(f"\n Generating results for {v}")

        rbRuns = []
        ppoRuns = []

        rbPast = []
        ppoPast = []

        modelPath = f"training/PPOScaler{v}"

        for runIdx in range(numRuns):
            print(f"Run {runIdx + 1}/{numRuns}")

            rbSummery, rbPasts = evaluationAgents(
                steps=steps,
                rewardVersion=v
            )
            rbRuns.append(rbSummery)
            rbPast.append(rbPasts)

            ppoEval = ppoEvaluation(
                modelPath=modelPath,
                rewardVersion=v
            )
            ppoSummary, ppoHistory = ppoEval.run(steps=steps)
            ppoRuns.append(ppoSummary)
            ppoPast.append(ppoHistory)

        avgRB = avgResults(rbRuns)
        avgPPO = avgResults(ppoRuns)

        summeryRows.append({
            "reward version": v,
            "agent": "Rule-Based",
            **avgRB
        })

        summeryRows.append({
            "reward version": v,
            "agent": "PPO",
            **avgPPO
        })

        rawResults[v] = {
            "rule based runs": rbRuns,
            "ppo runs": ppoRuns,
            "rule based pasts": rbPast,
            "ppo histories": ppoPast
        }

        plotHistory(
            rbPast[0],
            ppoPast[0],
            "QueuePast",
            v,
            os.path.join(graphs, f"queueOverTime{v}.png")
        )

        plotHistory(
            rbPast[0],
            ppoPast[0],
            "instancePast",
            v,
            os.path.join(graphs, f"instancesOverTime{v}.png")
        )

        ###redo below this line###

    saveCSV(summeryRows, os.path.join(results, "summary_results.csv"))
    saveRJson(rawResults, os.path.join(results, "raw_results.json"))

    print(summeryRows[0].keys())
    print(summeryRows[1].keys())

    plotbarMetric(summeryRows, "reward", os.path.join(graphs, "reward_comparison.png"))
    plotbarMetric(summeryRows, "average queue", os.path.join(graphs, "queue_comparison.png"))
    plotbarMetric(summeryRows, "average instances", os.path.join(graphs, "instances_comparison.png"))
    plotbarMetric(summeryRows, "scaling actions", os.path.join(graphs, "scaling_actions_comparison.png"))

    print("\nOfficial results complete.")
    print(f"CSV: {os.path.join(results, 'summary_results.csv')}")
    print(f"JSON: {os.path.join(results, 'raw_results.json')}")
    print(f"Graphs: {graphs}")












