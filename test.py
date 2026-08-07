from collections import deque
import numpy as np
from PIL import Image
import torch
from game import Game, Key, ALL_KEYS, NUM_KEYS
from network import ActorCritic
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms

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

obs_np = np.array(frame_stack, dtype=np.uint8)
obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)

# ----------------------
# gradient visualization for CNN embedding layer output
# ----------------------
frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.float32) / 255.0
cam = GradCAM(model = net.backbone, target_layers = [net.backbone.conv[6]])
grayscale_cam = cam(input_tensor = obs)
vis = show_cam_on_image(frame_rgb, grayscale_cam[0])
plt.imsave(f"gradcam_vis_step{ckpt['step']}_ep{ckpt['episode']}.png", vis)


"""
resnet18 benchmark

frame_pil = Image.fromarray(frame).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
obs = transform(frame_pil).unsqueeze(0)
frame_rgb = np.array(
    frame_pil.resize((224, 224))
).astype(np.float32) / 255.0

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cam = GradCAM(model = resnet18, target_layers = [resnet18.layer4[-1].conv2])
grayscale_cam = cam(input_tensor = obs)
vis = show_cam_on_image(frame_rgb, grayscale_cam[0])
plt.imsave(f"gradcam_vis_resnet18.png", vis)
"""