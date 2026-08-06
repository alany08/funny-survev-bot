import torch
import numpy as np
from game import Game
import torch.nn.functional as F
from network import ActorCritic
import time

N_STEPS = 2048 # steps per rollout
N_EPOCHS = 4 # passes over buffer per update
BATCH_SIZE = 64 # minibatch size, used to be 64
GAMMA = 0.99 # discount factor
GAE_LAMBDA = 0.95 # GAE smoothing
CLIP_EPS = 0.2 # PPO clip range
VALUE_COEF = 0.5 # value loss weight
ENTROPY_COEF = 0.01 # entropy bonus weight
MAX_GRAD_NORM = 0.5 # gradient clipping
STEP_TIME = 0.2 # time between steps

def collect_rollout(game: Game, net: ActorCritic, obs: torch.Tensor, reset, get_obs, frame_stack) -> dict:
    obs_buf = []
    key_buf = []
    aim_buf = []
    log_prob_buf = []
    value_buf = []
    reward_buf = []
    done_buf = []

    episodes_finished = 0

    for _ in range(N_STEPS):
        # timing for consistency
        start_time = time.time()

        with torch.no_grad():
            key_action, aim_action, log_prob, value = net.get_action(obs)

        key_vec = key_action.squeeze(0).cpu().numpy().tolist() # [21] binary list
        aim_vec = tuple(aim_action.squeeze(0).cpu().tolist()) # (x, y) from 0...1

        frame, reward, done = game.step(key_vec, aim_vec)

        obs_buf.append(obs)
        key_buf.append(key_action)
        aim_buf.append(aim_action)
        log_prob_buf.append(log_prob)
        value_buf.append(value)
        reward_buf.append(reward)
        done_buf.append(done)

        frame_stack.append(frame)
        obs = get_obs(frame_stack)

        # timing consistency
        diff = time.time() - start_time
        if STEP_TIME - diff < -0.3:
            print(f"WARN: step took {-STEP_TIME + diff}s longer than STEP_TIME")
        elif STEP_TIME - diff < 0:
            pass
            # negligible
        else:
            time.sleep(STEP_TIME - diff)

        if done:
            episodes_finished += 1
            frame_stack = reset()
            obs = get_obs(frame_stack)

    with torch.no_grad():
        _, _, _, last_value = net.get_action(obs)

        return {
            "obs": torch.cat(obs_buf, dim=0),
            "key_actions": torch.cat(key_buf, dim=0),
            "aim_actions": torch.cat(aim_buf, dim=0),
            "log_probs": torch.cat(log_prob_buf, dim=0),
            "values": torch.cat(value_buf, dim=0),
            "rewards": reward_buf,
            "dones": done_buf, 
            "last_value": last_value,
            "last_obs": obs,
            "episodes_finished": episodes_finished,
        }

def compute_gae(rewards: list, dones: list, values: torch.Tensor, last_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation
    """

    N = len(rewards)
    advantages = torch.zeros(N)
    gae = 0

    values_flat = values.squeeze(-1).cpu()
    last_value_flat = last_value.squeeze().cpu().item()

    for t in reversed(range(N)):
        if dones[t]:
            next_value = 0
            gae = 0

        else:
            next_value = values_flat[t + 1].item() if t + 1 < N else last_value_flat

        delta = rewards[t] + GAMMA * next_value - values_flat[t].item()
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + values_flat
    return advantages, returns

def update(rollout: dict, net: ActorCritic, opt: torch.optim.Optimizer) -> dict:
    """
    PPO update, runs N_EPOCHS over the rollout buffer in random minibatches
    """
    device = next(net.parameters()).device

    obs = rollout["obs"]
    key_actions = rollout["key_actions"]
    aim_actions = rollout["aim_actions"]
    old_log_probs = rollout["log_probs"]
    values = rollout["values"]
    last_value = rollout["last_value"]

    advantages, returns = compute_gae(rollout["rewards"], rollout["dones"], values, last_value)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = obs.to(device)
    key_actions = key_actions.to(device)
    aim_actions = aim_actions.to(device)
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    N = len(rollout["rewards"])

    # data tracking
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_entropy_key = 0.0
    total_entropy_aim = 0.0
    n_updates = 0

    for _ in range(N_EPOCHS):
        # random minibatch indices
        indices = torch.randperm(N)

        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]

            # --- re-evaluate stored actions under current policy ---
            log_probs, entropy, values_pred, entropy_key, entropy_aim = net.evaluate_actions(
                obs[batch_idx],
                key_actions[batch_idx],
                aim_actions[batch_idx],
            )

            # --- PPO clipped surrogate loss ---
            ratio        = torch.exp(log_probs - old_log_probs[batch_idx])
            adv_batch    = advantages[batch_idx]

            surr1        = ratio * adv_batch
            surr2        = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_batch
            policy_loss  = -torch.min(surr1, surr2).mean()

            # --- value loss ---
            value_loss   = F.mse_loss(
                values_pred.squeeze(-1),
                returns[batch_idx]
            )

            # --- entropy bonus ---
            entropy_loss = -entropy.mean()

            # --- total loss ---
            loss = (
                policy_loss
              + VALUE_COEF  * value_loss
              + ENTROPY_COEF * entropy_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.mean().item()
            total_entropy_key += entropy_key.mean().item()
            total_entropy_aim += entropy_aim.mean().item()
            n_updates         += 1

    mean_reward = float(np.mean(rollout["rewards"]))

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss":  total_value_loss  / n_updates,
        "entropy":     total_entropy     / n_updates,
        "mean_reward": mean_reward,
        "entropy_key": total_entropy_key / n_updates,
        "entropy_aim": total_entropy_aim / n_updates,
    }