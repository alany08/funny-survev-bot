from enum import Enum

#web interface
from selenium import webdriver
import selenium
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import read_version_from_cmd
from webdriver_manager.core.os_manager import PATTERN
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import time

#image processing
from io import BytesIO
from PIL import Image
import base64

#math
import numpy as np
import cv2

class Key(Enum):
    W   = "w"
    A   = "a"
    S   = "s"
    D   = "d"
    F   = "f"
    R   = "r"
    X   = "x"
    M   = "m"
    V   = "v"
    K1  = "1"
    K2  = "2"
    K3  = "3"
    K4  = "4"
    K5  = "5"
    K6  = "6"
    K7  = "7"
    K8  = "8"
    K9  = "9"
    K0  = "0"
    LMB = "lmb"
    MOVE = "move"

ALL_KEYS    = list(Key)
NUM_KEYS    = len(ALL_KEYS)  # 21 → Bernoulli head size
# Gaussian for move x, and move y. only applies of MOVE key is set to 1 in Bernoulli

class Game:
    WIDTH = 640*2
    HEIGHT = 360*2

    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-infobars")
        options.add_argument("--headless=new")   # new headless mode, better GPU support
        options.add_argument("--use-gl=egl")
        options.add_argument("--enable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--remote-debugging-address=0.0.0.0")

        browser_version = read_version_from_cmd(
            "/usr/sbin/chromium --version",
            PATTERN["chromium"],
        )
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager(driver_version = browser_version).install()),
            options=options
        )
        driver.get("https://survev.io/")

        self.driver = driver
        self.actions = ActionChains(driver)

        # other misc setup
        self.set_desired_window_size()

        # state tracking
        self._state = {
            "health": 100.0,
            "kills": 0
        }

        self._last_input = [0] * len(ALL_KEYS)

    def set_desired_window_size(self):
        outer_w = self.driver.execute_script("return window.outerWidth")
        outer_h = self.driver.execute_script("return window.outerHeight")
        inner_w = self.driver.execute_script("return window.innerWidth")
        inner_h = self.driver.execute_script("return window.innerHeight")

        delta_w = outer_w - inner_w
        delta_h = outer_h - inner_h

        self.driver.set_window_size(self.WIDTH + delta_w, self.HEIGHT + delta_h)

    # input stuff
    def join_game(self):
        element = self.driver.find_element(By.ID, "btn-start-mode-0")
        WebDriverWait(self.driver, 10, ignored_exceptions = [selenium.common.exceptions.ElementNotInteractableException]).until(lambda _: element.click() or True)
        time.sleep(3)

    def do_input(self, input_str: str):
        # value is tuple for mouse coords move/click, str for key
        # down is true if it's keydown, else keyup
        raw_string = input_str.replace("+", "").replace("-", "")

        if raw_string in "wasdfrx12347890mv":
            if "+" in input_str:
                self.actions.key_down(raw_string)
            elif "-" in input_str:
                self.actions.key_up(raw_string)
            else:
                raise ValueError("No +- specified for keypress")

        elif raw_string == "lmb":
            if "+" in input_str:
                self.actions.click_and_hold()
            elif "-" in input_str:
                self.actions.release()
            else:
                raise ValueError("No +- specified for mouse click")

        elif "move" in input_str:
            # "move 123, 123"
            coord_x = int(input_str.replace("move", "").split(",")[0])
            coord_y = int(input_str.replace("move", "").split(",")[1])

            body = self.driver.find_element(By.CSS_SELECTOR, "body")
            self.actions.move_to_element_with_offset(body, coord_x - int(self.WIDTH/2), coord_y - int(self.HEIGHT/2))

    def execute_all_input(self):
        self.actions.perform()

    def do_input_binary(self, vector: list[int], coordinate: tuple[int]):
        if len(ALL_KEYS) != len(vector):
            raise ValueError("Length of input vector does not equal enumerated key options")
        for i in range(len(vector)):
            if vector[i] != self._last_input[i]:
                if vector[i] == 1 and ALL_KEYS[i].value != "move":
                    self.do_input("+" + ALL_KEYS[i].value)
                elif vector[i] == 0 and ALL_KEYS[i].value != "move":
                    self.do_input("-" + ALL_KEYS[i].value)
                elif vector[i] == 1 and ALL_KEYS[i].value == "move":
                    self.do_input(f"move {int(coordinate[0] * self.WIDTH)},{int(coordinate[1] * self.HEIGHT)}")

        self.execute_all_input()

        self._last_input = vector.copy()

    # game cycle handling
    def screenshot(self):
        # png = self.driver.get_screenshot_as_png()

        screenshot_data = self.driver.execute_cdp_cmd("Page.captureScreenshot", {
            "format": "png",  # Or "jpeg" for smaller size
            "quality": 100     # Compression adjustment if using jpeg
        })

        png = base64.b64decode(screenshot_data["data"])

        # with open("debug_frame_raw.png", "wb") as f:
        #     f.write(png)

        stream = BytesIO(png)
        frame = np.array(Image.open(stream))    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (int(self.WIDTH/2), int(self.HEIGHT/2)))

        return frame

    def is_dead(self) -> bool:
        dead = self.driver.execute_script('return document.getElementById("ui-stats-header").textContent != ""')
        return dead

    def reset(self):
        self._state = {
            "health": 100.0,
            "kills": 0
        }

        try:
            element = self.driver.find_element(By.CSS_SELECTOR, ".ui-stats-restart")
            WebDriverWait(
                self.driver,
                10,
                ignored_exceptions = [
                    selenium.common.exceptions.ElementNotInteractableException,
                ]
            ).until(lambda _: element.click() or True)
        except selenium.common.exceptions.NoSuchElementException:
            pass

        if self.driver.current_url != "https://survev.io/":
            print("WARN: URL mismatch at", self.driver.current_url)
            self.driver.get("https://survev.io/")

        for key in (ALL_KEYS):
            if key.value != "move":
                self.do_input("-" + key.value)
                self.execute_all_input()

        self.join_game()
        return self.screenshot()

    def get_health(self) -> float:
        health = self.driver.execute_script('return document.getElementById("ui-health-actual").style.width')
        health = float(health.replace("%", ""))
        return health

    def get_kills(self) -> int:
        kills = self.driver.execute_script('return document.querySelector(".ui-player-kills").textContent')
        return int(kills)

    def get_rank(self) -> int:
        if self.is_dead():
            rank = self.driver.execute_script('return document.querySelector(".ui-stats-header-value").textContent')
            rank = int(rank.replace("#", ""))

            return rank

    def get_reward(self) -> float:
        kills = self.get_kills()
        health = self.get_health()

        d_kills = kills - self._state["kills"]
        d_health = health - self._state["health"]

        dead = self.is_dead()
        if dead:
            rank = self.get_rank()
        else:
            rank = -1

        self._state = {
            "health": health,
            "kills": kills
        }

        reward = 0
        reward += d_kills * 10
        reward += d_health * 0.1
        reward += 0.01

        if dead and rank != 1:
            kill_reduction = min(self._state["kills"] * 2.0, 15.0)
            reward -= (20 - kill_reduction)
        elif dead and rank == 1:
            reward += 100

        return reward

    def step(self, action_vec: list[int], aim_vec: tuple[int]):
        self.do_input_binary(action_vec, aim_vec)

        next_frame = self.screenshot()
        reward = self.get_reward()
        dead = self.is_dead()

        return next_frame, reward, dead

if __name__ == "__main__":
    game = Game()
    game.join_game()

    image = game.screenshot()
    cv2.imshow("ss", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    while True:
        time.sleep(0.1)

        print(game.get_health())
        print(game.get_kills())

        if game.is_dead():
            print("Dead:", game.get_rank())
            game.reset()