from collections import deque
import numpy as np
import torch
from game import Game, Key, ALL_KEYS, NUM_KEYS
from network import ActorCritic
import sys
import os
import signal
import ppo
import csv

CHECKPOINT_PATH = "checkpoints/model.pt"
LOG_PATH = "logs/training.csv"

def init_log():
    os.makedirs("logs", exist_ok=True)
    # only write header if file doesn't exist (resume safe)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "episode", "mean_reward",
                             "policy_loss", "value_loss", "entropy", "entropy_key", "entropy_aim"])

def log_metrics(state, metrics):
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            state["step"],
            state["episode"],
            round(state["mean_reward"], 4),
            round(metrics["policy_loss"], 4),
            round(metrics["value_loss"],  4),
            round(metrics["entropy"],     4),
            round(metrics["entropy_key"],     4),
            round(metrics["entropy_aim"],     4),
        ])

def save_checkpoint(net, opt, state):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model":       net.state_dict(),
        "optimizer":   opt.state_dict(),
        "step":        state["step"],
        "episode":     state["episode"],
        "mean_reward": state["mean_reward"],
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved at step {state['step']}")

def load_checkpoint(net, opt):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint — starting fresh")
        return {"step": 0, "episode": 0, "mean_reward": 0.0}

    ckpt = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']}, episode {ckpt['episode']}")
    return {
        "step":        ckpt["step"],
        "episode":     ckpt["episode"],
        "mean_reward": ckpt["mean_reward"],
    }

def train():
    init_log()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Game()
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    state = load_checkpoint(net, opt)

    def handle_exit(sig, frame):
        print("Exit requested, saving...")
        save_checkpoint(net, opt, state)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    frame_stack = deque(maxlen=4)

    def reset():
        frame = game.reset()
        frame_stack.clear()
        for _ in range(4):
            frame_stack.append(frame)

        return frame_stack

    def get_obs(frame_stack):
        obs = np.array(frame_stack, dtype=np.uint8)
        return torch.from_numpy(obs).unsqueeze(0).to(device)

    frame_stack = reset()
    obs = get_obs(frame_stack)

    while True:
        rollout = ppo.collect_rollout(game, net, obs, reset, get_obs, frame_stack)
        metrics = ppo.update(rollout, net, opt)
        obs = rollout["last_obs"]
        state["step"] += 1
        state["episode"] += rollout["episodes_finished"]
        state["mean_reward"] = metrics["mean_reward"]

        print(
            f"step {state['step']:>6} | "
            f"ep {state['episode']:>5} | "
            f"reward {state['mean_reward']:>8.2f} | "
            f"policy_loss {metrics['policy_loss']:>7.4f} | "
            f"value_loss {metrics['value_loss']:>7.4f} | "
            f"entropy {metrics['entropy']:>6.4f} | "
            f"entropy_key {metrics['entropy_key']:>6.4f} | "
            f"entropy_aim {metrics['entropy_aim']:>6.4f}"
        )
        log_metrics(state, metrics)   # same data, written to CSV

        if state["step"] % 1 == 0: # just do it every 1...
            save_checkpoint(net, opt, state)

if __name__ == "__main__":
    train()