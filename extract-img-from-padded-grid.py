import argparse
import os

from PIL.Image import open

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("file", type=str)
parser.add_argument("-x", type=int, nargs="+")
parser.add_argument("-y", type=int, nargs="+")
parser.add_argument("-s", "--size", type=int, default=256)
parser.add_argument("-p", "--padding", type=int, default=2)
parser.add_argument("-h", "--height", type=int, default=9)
parser.add_argument("-w", "--width", type=int, default=16)
file, x, y, s, p, h, w = vars(parser.parse_args()).values()

assert len(x) == len(y), "Lenght of x and y coordinates must be equal!"

path, ext = os.path.splitext(file)

for x, y in zip(x, y):
    gx = 1 + x * (s + 2)
    gy = 1 + y * (s + 2)
    open(file).crop((gx, gy, gx + s + 2, gy + s + 2)).save(f"{path}_{x}_{y}{ext}")
