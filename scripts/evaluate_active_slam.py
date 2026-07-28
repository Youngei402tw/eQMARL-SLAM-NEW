"""Evaluate random or frontier active-SLAM policies on fixed unseen maps."""

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

import eqmarl
from eqmarl.environments.active_slam import MultiAgentSLAMEnv
from eqmarl.policies import FrontierJointPolicy, LearnedJointPolicy, RandomJointPolicy
from eqmarl.tools import NumpyJSONEncoder
from experiment_runner import load_experiment


def evaluate(policy_name: str, episodes: int, seed_start: int, config=None, weights=None):
    parameter_count = 0
    if policy_name == "actor":
        if not config or not weights:
            raise ValueError("actor evaluation requires --config and --weights")
        with open(config) as config_file:
            definition = yaml.load(config_file, Loader=eqmarl.yaml.ConfigLoader)
        experiment = load_experiment(definition)
        env = experiment["algorithm"].env
        model = experiment["algorithm"].model_actor
        model.load_weights(weights)
        parameter_count = int(model.count_params())
        policy = LearnedJointPolicy(model)
    else:
        env = MultiAgentSLAMEnv(seed=seed_start)
        policy = RandomJointPolicy(seed_start) if policy_name == "random" else FrontierJointPolicy()
    results = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed_start + episode)
        terminated = truncated = False
        reward = 0.0
        inference_seconds = 0.0
        inference_calls = 0
        while not (terminated or truncated):
            start = time.perf_counter()
            action = policy.action(env, observation)
            inference_seconds += time.perf_counter() - start
            inference_calls += 1
            observation, rewards, terminated, truncated, _ = env.step(action)
            reward += float(np.mean(rewards))
        results.append(
            {
                **env.episode_metrics(),
                "reward": reward,
                "inference_ms": 1000.0 * inference_seconds / inference_calls,
            }
        )
    summary = {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results])),
            "ci95": float(
                1.96 * np.std([result[key] for result in results], ddof=1) / np.sqrt(episodes)
            ) if episodes > 1 else 0.0,
        }
        for key in results[0]
    }
    return {
        "policy": policy_name,
        "episodes": episodes,
        "seed_start": seed_start,
        "trainable_parameters": parameter_count,
        "summary": summary,
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("random", "frontier", "actor"), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--config")
    parser.add_argument("--weights")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate(
        args.policy, args.episodes, args.seed_start, config=args.config, weights=args.weights
    )
    encoded = json.dumps(result, cls=NumpyJSONEncoder, indent=2)
    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
