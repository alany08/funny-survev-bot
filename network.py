import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta

from game import NUM_KEYS

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4,   32,  kernel_size=8, stride=4),  # → [32,  159, 158]
            nn.ReLU(),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2),  # → [64,   78,  78]
            nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=4, stride=2),  # → [128,  38,  38]
            nn.ReLU(),
            nn.Conv2d(128, 64,  kernel_size=3, stride=1),  # → [64,   36,  36]
            nn.ReLU(),
        )

        flat_size = self._get_flat_size()

        self.mlp = nn.Sequential(
            nn.Linear(flat_size, 1024),  # bigger MLP to use spare VRAM
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 360, 640, device=next(self.parameters()).device)
            return self.conv(dummy).flatten(1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x

class ActorCritic(nn.Module):
    """
        PPO Actor-Critic with:
            - Shared CNN backbone
            - key_head:     Bernoulli(NUM_KEYS=21) — one bit per key/action
            - aim_head:     Beta(α, β)                         — (x, y) normalized to (0,1)
            - value_head: scalar V(s)                        — critic estimate
        
        Aim is gated by Key.MOVE bit in key_vector:
            MOVE=1 → apply aim coords        (converted to pixels in SurvevEnv)
            MOVE=0 → mouse doesn't move
        """
    def __init__(self):
        super().__init__()

        self.backbone = CNNBackbone()

        # heads
        self.key_head = nn.Linear(256, NUM_KEYS)
        self.aim_alpha = nn.Linear(256, 2)
        self.aim_beta = nn.Linear(256, 2)

        self.value_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.backbone.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)

        for head in [self.key_head, self.aim_alpha, self.aim_beta]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _aim_dist(self, features: torch.Tensor) -> Beta:
        alpha = F.softplus(self.aim_alpha(features)) + 1.0
        beta = F.softplus(self.aim_beta(features)) + 1.0
        return Beta(alpha, beta)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        key_dist = Bernoulli(logits=self.key_head(features))
        aim_dist = self._aim_dist(features)
        value = self.value_head(features)

        return key_dist, aim_dist, value

    def get_action(self, obs: torch.Tensor):
        key_dist, aim_dist, value = self.forward(obs)

        key_action = key_dist.sample()
        aim_action = aim_dist.sample()

        log_prob = (
            key_dist.log_prob(key_action).sum(-1)
            +    aim_dist.log_prob(aim_action).sum(-1)    
        )

        return key_action, aim_action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, key_action: torch.Tensor, aim_action: torch.Tensor):
        key_dist, aim_dist, value = self.forward(obs)

        log_prob = (
            key_dist.log_prob(key_action).sum(-1)
            + aim_dist.log_prob(aim_action).sum(-1)    
        )

        entropy_key = key_dist.entropy().sum(-1)
        entropy_aim = aim_dist.entropy().sum(-1) 

        entropy = entropy_key + entropy_aim

        return log_prob, entropy, value, entropy_key, entropy_aim