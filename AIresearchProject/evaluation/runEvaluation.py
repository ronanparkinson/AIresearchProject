from evaluation.evaluationAgents import evaluationAgents
from evaluation.ppoEvaluation import ppoEvaluation

if __name__ == "__main__":

    ruleBasedResults = evaluationAgents()
    ppoEvaluater = ppoEvaluation()
    ppoResults = ppoEvaluater.run()

    print("Rule based results")
    print(ruleBasedResults)

    print("\nPPO results")
    print(ppoResults)