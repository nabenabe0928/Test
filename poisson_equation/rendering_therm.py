import os
from PIL import Image, ImageDraw
import glob

element = ["Au","Cu","Fe","Hg"][0]
path = os.getcwd() + "/thermal_dist_{}/".format(element)
files = [path + "thermal_figure_{}_{}.png".format(element, i) for i in range(len(os.listdir(path)))]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save("new_magnetic_thermal_dist.gif" ,save_all = True, append_images = images, duration = 500, optimize = False, loop = 0)