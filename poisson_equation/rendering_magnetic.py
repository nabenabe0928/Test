import os
from PIL import Image, ImageDraw
import glob

path = os.getcwd() + "/magnetic/"
files = [path + "magnetic_figure{}.png".format(i) for i in range(len(os.listdir(path)))]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save("new_magnetic.gif" ,save_all = True, append_images = images, duration = 500, optimize = False, loop = 0)

print("######")

path = os.getcwd() + "/density/"
files = [path + "density_figure{}.png".format(i) for i in range(len(os.listdir(path)))]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save("new_density.gif" ,save_all = True, append_images = images, duration = 500, optimize = False, loop = 0)
