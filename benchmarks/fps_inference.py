"""Run 1000 inferences on a single image and print FPS."""

import time
from pathlib import Path

from mayaku.inference import Predictor
from mayaku.utils.image import read_image

IMAGE_PATH = "input.jpg"
WEIGHTS    = Path("ema-model_iter_0090000.pth")
N = 1000

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x", weights=WEIGHTS)
image = read_image(IMAGE_PATH)

# Warmup
for _ in range(10):
    print('w')
    predictor(image)

start = time.time()
for _ in range(N):
    print('x')
    predictor(image)
elapsed = time.time() - start

print(f"Iterations : {N}")
print(f"Elapsed    : {elapsed:.2f} s")
print(f"FPS        : {N / elapsed:.2f}")
