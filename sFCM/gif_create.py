import imageio
import os

folder = "PCiDS/sFCM/gifs"
ims = []
for (s, w, files) in os.walk(folder):
    for f in sorted(list(files), key = lambda x : int(x.split('.')[0])):
        ims.append(os.path.join(folder, f))
        print(f)

images = [imageio.imread(f) for f in ims]

imageio.mimsave(os.path.join(folder, 'out.gif'), images, duration = 1)