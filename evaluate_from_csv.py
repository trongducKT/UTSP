import pandas as pd
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.solvers.exact_solver import DPSolver
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
import asyncio
import numpy as np
from typing import List
import json
import time
import traceback
# from infer import infer
import numpy as np
from dataset_generator_v2 import MetricTSPV2Generator
import traceback
from loadmodel import run_UTSP


def is_valid_path(path: List[int]) -> bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path) >= 3) and (path[0] == path[-1])


def caculate_cost(edges, path):
    if isinstance(path, list):
        assert is_valid_path(path), ValueError("Provided path is invalid")
        assert len(path) == problem.n_nodes + 1, ValueError(
            "An invalid number of cities are contained within the provided path"
        )

        distance = 0
        for i, source in enumerate(path[:-1]):
            destination = path[i + 1]
            distance += edges[source][destination]
        return distance


def is_approximately_equal(value1, value2, tolerance_percentage=0.00001):
    # Handle infinite values explicitly
    if np.isinf(value1) or np.isinf(value2):
        return value1 == value2

    # Calculate the absolute tolerance from the percentage
    tolerance = tolerance_percentage / 100 * value1
    return np.isclose(value1, value2, atol=tolerance)


def score_worse_than_reference(score, reference, objective_function):
    if objective_function == "min":
        if score > reference:
            return True
        else:
            return False
    else:
        if score < reference:
            return True
        else:
            return False


def scaled_rewards(
    score, benchmark: float, best_score, objective_function: str = "min"
):
    def score_gap(score, best_score, reference):
        if is_approximately_equal(score, reference):
            return 0.2  # matched the benchmark so assign a floor score
        elif score_worse_than_reference(score, reference, objective_function):
            return 0  # scored worse than the required benchmark
        else:
            # proportionally scale rewards based on the relative normalized scores
            assert not is_approximately_equal(
                best_score, reference
            ) and not score_worse_than_reference(
                best_score, reference, objective_function
            ), ValueError(
                f"Best score is worse than reference: best-{best_score}, ref-{reference}"
            )
            return (
                1 - abs(best_score - score) / abs(best_score - reference)
            ) * 0.8 + 0.2

    # we find scores that correspond to finite path costs
    # bt.logging.info(f"Miners were scored: {scores}")
    # print(f"Miners were scored: {scores}")
    # bt.logging.info(f"With the valid scores of: {filtered_scores}")
    # print(f"With the valid scores of: {filtered_scores}")
    if (score is None) and (score == np.inf):
        return 0
    worst_score = 2 * best_score
    # if best_score == benchmark:
    #     return [int(score!=None) for score in scores]
    if benchmark == np.inf:
        return score_gap(score, best_score, worst_score)
    elif not score_worse_than_reference(worst_score, benchmark, "min"):
        return score_gap(score, best_score, worst_score)
    else:
        return score_gap(score, best_score, benchmark)


# read file csv
df = pd.read_csv("./data_v7_14_09.csv")

# get column edges
selected_id = df["selected_id"].to_list()[:20]
dataset_ref = df["dataset_ref"].to_list()[:20]
best_scores = df["min_distance"].to_list()[:20]
n_nodes = df["n_nodes"].to_list()[:20]
# from edges to GraphProblem

print("________________________")
print("Testing MetricTSPGenerator V2")
loaded_datasets = {}
try:
    with np.load("dataset/Asia_MSB.npz") as f:
        loaded_datasets["Asia_MSB"] = np.array(f["data"])
except:
    pass
try:
    with np.load("dataset/World_TSP.npz") as f:
        loaded_datasets["World_TSP"] = np.array(f["data"])
except:
    pass

problems = []
for ids, ref, n_node in zip(selected_id, dataset_ref, n_nodes):
    ids = json.loads(ids)
    test_problem = GraphV2Problem(
        problem_type="Metric TSP",
        n_nodes=n_node,
        selected_ids=ids,
        cost_function="Geom",
        dataset_ref=ref,
    )
    MetricTSPV2Generator.recreate_edges(test_problem, loaded_datasets)
    problems.append(test_problem)


# sample_problem = problems[0]
# problems = [
#     MetricTSPV2Generator.recreate_edges(problem, loaded_datasets)
#     for problem in problems
# ]
# problems = []
# for i in range(10):
#     test_problem = MetricTSPV2Generator.generate_one_sample(
#         size=5000, load_datasets=loaded_datasets
#     )
#     MetricTSPV2Generator.recreate_edges(test_problem, loaded_datasets)
#     problems.append(test_problem)


benmark_solver = NearestNeighbourSolverVali()
# solver = HPNSolver()
# Optimized Dictionary of solvers with configurations to reduce time for larger graphs
# More aggressively optimized dictionary of solvers
# solvers = {
#     "small": ACOSolver(
#         ant_count=5, generations=150, alpha=1.0, beta=3.0, rho=0.7, q=100, strategy=0
#     ),
#     "medium": ACOSolver(
#         ant_count=15, generations=250, alpha=1.0, beta=3.5, rho=0.7, q=200, strategy=0
#     ),
#     "large": ACOSolver(
#         ant_count=30, generations=400, alpha=1.0, beta=4.0, rho=0.6, q=500, strategy=0
#     ),
#     "very_large": ACOSolver(
#         ant_count=70, generations=600, alpha=1.0, beta=4.5, rho=0.5, q=800, strategy=0
#     ),
#     "extreme_large": ACOSolver(
#         ant_count=80, generations=800, alpha=1.0, beta=4.5, rho=0.5, q=1000, strategy=0
#     ),
#     "extreme_extreme_large": ACOSolver(
#         ant_count=30, generations=400, alpha=1.0, beta=5.0, rho=0.8, q=500, strategy=2
#     ),
# }


# aco = ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
solver = DPSolver()
small = []
medium = []
large = []
very_large = []

count = 0
optimizes = []
rewards = []
for problem, best_score in zip(problems, best_scores):
    print("-------------------------len nodes: ", len(problem.edges))
    n_node = len(problem.edges)
    # if len(problem.nodes) == 0:
    #     continue

    # if 10 <= n_node <= 14:
    #     solver = solvers["small"]
    # if 15 <= n_node <= 25:
    #     solver = solvers["medium"]
    # if 26 <= n_node <= 99:
    #     solver = solvers["large"]
    # if 100 <= n_node <= 250:
    #     solver = solvers["very_large"]
    # if 250 <= n_node <= 1000:
    #     solver = solvers["extreme_large"]
    # if 1000 < n_node:
    #     solver = solvers["extreme_extreme_large"]
    try:
        start = time.time()
        # route = asyncio.run(solver.solve_problem(problem))
        # route = infer(problem.nodes)

        # data = np.array(problem.edges)
        # aco = ACO(num_city=data.shape[0], dis_mat=data.copy())
        # route, Best = aco.run()
        # model = GA(
        #     num_city=data.shape[0], num_total=25, iteration=500, dis_mat=data.copy()
        # )
        # route, path_len = model.run()

        # model = PSO(num_city=data.shape[0], dis_mat=data.copy())
        # route, Best = model.run()
        # route = infer_tabu_rule1(problem.edges)
        route, a, b, c = run_UTSP(problem.edges, 64, 2, 3.5, 20)
        # model = TS(num_city=data.shape[0], dis_mat=data.copy())
        # route, Best_length = model.run()

        # model = SA(num_city=data.shape[0], dis_mat=data.copy())
        # route, path_len = model.run()

        # model = DP(num_city=data.shape[0], dis_mat=data.copy())
        # route, Best = model.run()

        print("Time: ", time.time() - start)

        score = caculate_cost(problem.edges, route)
        # get reward from route
        benchmark_path = asyncio.run(benmark_solver.solve_problem(problem))
        benchmark_score = caculate_cost(problem.edges, benchmark_path)
        print(score, benchmark_score, best_score)

        reward = scaled_rewards(score, benchmark_score, best_score)
        rewards.append(reward)
        print("reward:", reward)
        optimize = (best_score - score) / best_score

        optimizes.append(optimize)
        if 10 <= n_node <= 14:
            small.append(optimize)
        if 15 <= n_node <= 25:
            medium.append(optimize)
        if 26 <= n_node <= 99:
            large.append(optimize)
        if 100 <= n_node:
            very_large.append(optimize)
        print("Optimize distance from base solver: ", optimize)
        print("Problem: ", count)
        count += 1
    except:
        # print traceback
        traceback.print_exc()
        continue

print("Mean optimize: ", np.mean(optimizes))
print("number of small: ", len(small))
print("Mean small: ", np.mean(small))
print("number of medium: ", len(medium))
print("Mean medium: ", np.mean(medium))
print("number of large: ", len(large))
print("Mean large: ", np.mean(large))
print("number of very large: ", len(very_large))
print("Mean very large: ", np.mean(very_large))
print("Mean reward: ", np.mean(rewards))
