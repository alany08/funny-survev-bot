from collections import deque
import numpy as np
import torch
from game import Game, Key, ALL_KEYS, NUM_KEYS
from network import ActorCritic
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

CHECKPOINT_PATH = "checkpoints/model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = Game()
net = ActorCritic().to(device)

ckpt = torch.load(CHECKPOINT_PATH)
net.load_state_dict(ckpt["model"])

print(f"Model @ step {ckpt['step']}, episode {ckpt['episode']}")


game.join_game()

input("Hit enter to capture frame")

frame = game.screenshot()
frame_stack = deque(maxlen=4)
for _ in range(4):
    frame_stack.append(frame)

obs = np.array(frame_stack, dtype=np.uint8)
obs = torch.from_numpy(obs).unsqueeze(0).to(device)

# ----------------------
# gradient visualization for CNN embedding layer output
# ----------------------
frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.float32) / 255.0
cam = GradCAM(model = net.backbone, target_layers = [net.backbone.conv[6]])
grayscale_cam = cam(input_tensor = obs)
vis = show_cam_on_image(frame_rgb, grayscale_cam[0])
plt.imsave(f"gradcam_vis_step{ckpt['step']}_ep{ckpt['episode']}.png", vis)