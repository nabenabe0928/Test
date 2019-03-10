import os
from PIL import Image, ImageDraw
import glob

elements = ["Au", "Cu", "Fe", "Hg"]

for element in elements:
	path = os.getcwd() + "/dynamic_{}_img/".format(element)
	files = [path + "dynamic_figure{}.png".format(i) for i in range(len(os.listdir(path)))]
	images = list(map(lambda file: Image.open(file), files))
	print(element)
	images.pop(0).save("new_dynamic_{}.gif".format(element) ,save_all = True, append_images = images, duration = 5, optimize = False, loop = 0)
