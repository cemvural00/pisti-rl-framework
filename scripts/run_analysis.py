"""Post-training analysis pipeline.

Waits for the training queue to finish, then runs the tournament,
behavioral stats, and all plots. Results land in results/ and plots/.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_for_queue(log="runs/queue.log", sentinel="queue2 finished", hours=8):
    t0 = time.time()
    while time.time() - t0 < hours * 3600:
        if os.path.exists(log) and sentinel in open(log).read():
            return
        time.sleep(120)
    raise TimeoutError("queue did not finish in time")


def main():
    wait_for_queue()
    print("queue finished; starting analysis", time.strftime("%H:%M:%S"))

    from training.evaluate import run_tournament
    from analysis.behavior import compare_agents
    from analysis import plots

    agents = [
        "random",
        "greedy",
        "hunter",
        "expectimax:16,6",
        "ppo:runs/ppo_main/final_model",
        "ppo:runs/ppo_s1/final_model",
        "ppo:runs/ppo_s2/final_model",
        "ppo-nomem:runs/ppo_nomem/final_model",
    ]
    tournament = run_tournament(
        agents, n_deals=250, seed=123, out="results/tournament.json"
    )

    behavior = compare_agents(
        ["greedy", "hunter", "expectimax:16,6", "ppo:runs/ppo_main/final_model"],
        n_games=400,
        seed=7,
    )
    with open("results/behavior.json", "w") as f:
        json.dump(behavior, f, indent=2)

    plots.training_curves(
        ["runs/ppo_main", "runs/ppo_s1", "runs/ppo_s2", "runs/ppo_nomem"],
        "plots",
    )
    plots.tournament_heatmap(tournament, "plots")
    plots.ratings_bar(tournament, "plots")
    luck = plots.luck_vs_skill(tournament, "plots")
    with open("results/luck_vs_skill.json", "w") as f:
        json.dump(luck, f, indent=2)

    # Exploitability plot if the lane has produced results
    import glob

    points, baselines = [], {}
    for p in glob.glob("results/exploit/*.json"):
        d = json.load(open(p))
        if "target_steps" in d:
            points.append(
                {
                    "steps": d["target_steps"],
                    "exploitability_pts": d["exploitability_pts"],
                    "ci95": d["ci95"],
                }
            )
        else:
            baselines[d["target"]] = d["exploitability_pts"]
    if points:
        plots.exploitability_curve(points, "plots", baselines=baselines)

    print("analysis complete", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
