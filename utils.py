import os
import shutil
import glob
from PIL import Image


def save_gif(img_dir, gif_dir, file_prefix, duration):
    # img_dirにある画像をgif_dirにgifとして保存する
    num_gif_files = len(next(os.walk(gif_dir))[2])
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(fp=f"{gif_dir}/{file_prefix}_{str(num_gif_files).zfill(2)}.gif", format="GIF",
                append_images=imgs, save_all=True, duration=duration, loop=0)
    shutil.copyfile(f"{gif_dir}/{file_prefix}_{str(num_gif_files).zfill(2)}.gif", f"{gif_dir}/{file_prefix}_for_simulate.gif")
    print(f"GIF saved: {gif_dir}/{file_prefix}_{str(num_gif_files).zfill(2)}.gif")
    shutil.rmtree(img_dir)
    os.makedirs(img_dir)
