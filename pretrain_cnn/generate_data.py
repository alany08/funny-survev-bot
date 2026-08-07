from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from game import Game
from collections import deque
import time
import pickle
from copy import deepcopy

session = Game()
session.join_game()

session.driver.execute_script("""
window.keys = "wasdfrxmv1234567890"
window.key_status = Array(20).fill(0)
window.mousepos = [0, 0]
document.body.onkeydown = (e) => {
    idx = window.keys.indexOf(e.key);
    if (idx !== -1) window.key_status[idx] = 1;
}
document.body.onkeyup = (e) => {
    idx = window.keys.indexOf(e.key);
    if (idx !== -1) window.key_status[idx] = 0;
}
document.body.onmousedown = (e) => {
    window.key_status[19] = 1;
}
document.body.onmouseup = (e) => {
    window.key_status[19] = 0;
}
document.body.onmousemove = (e) => {
    ratiox = e.clientX / window.innerWidth;
    ratioy = e.clientY / window.innerHeight;
    window.mousepos =  [ratiox, ratioy]
}""")

def log_state(game):
    return game.driver.execute_script("return [window.key_status, window.mousepos]")

collected_data = []
try:
    collected_data = pickle.load("labeled_data.pkl")
except:
    print("No past save, starting fresh")

ss_queue = deque(maxlen = 4)

for _ in range(4):
    ss_queue.append(session.screenshot())

while True:
    if session.is_dead():
        session.reset()

    time_start = time.time()

    ss_queue.append(session.screenshot())
    state = log_state(session)
    print("Logged state:", state)

    collected_data.append([deepcopy(ss_queue), state])

    if len(collected_data) % 50 == 0:
        with open("labeled_data.pkl", "wb") as f:
            pickle.dump(collected_data, f)

    time_end = time.time()

    if time_end - time_start < 0.2:
        time.sleep(0.2 - (time_end - time_start))
