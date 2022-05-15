import numpy as np
from pathlib import Path
from PIL import Image
import logging
import matplotlib.pyplot as plt
import torch
import math
import glob, re
from torchvision import transforms


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def feature_visualization_yolov5(features, prefix_name, save_dir, n=64, orignal_img=None):
    #https://github.com/ultralytics/yolov5/issues/1067#issuecomment-869606540
    """
    features:       Features to be visualized
    module_type:    Module type
    module_idx:     Module layer index within model
    n:              Maximum number of feature maps to plot
    """
    plt.figure(tight_layout=True)
    feat_size = features.shape[-1]
    blocks = torch.chunk(features, features.shape[1], dim=1)  # block by channel dimension
    n_1 = min(n, len(blocks)) + 1#add 1 is orignal image
    subplot_size = int(math.sqrt(n_1))
    subplot_size = subplot_size+1 if subplot_size*subplot_size <n_1 else subplot_size
    for i in range(n_1 - 1):
        feature = transforms.ToPILImage()(blocks[i].squeeze())
        ax = plt.subplot(subplot_size, subplot_size, i + 1)
        ax.axis('off')
        plt.imshow(feature)  # cmap='gray'
        ax.title.set_text(f'channel_id {i}')
    #original image
    feature = transforms.ToPILImage()(orignal_img)
    ax = plt.subplot(subplot_size, subplot_size, i + 2)
    ax.axis('off')
    plt.imshow(feature)  # cmap='gray'

    f = f"{prefix_name}_size_i{orignal_img.shape[-1]}-f{feat_size}_features.png"
    print(f'Saving {save_dir / f}...')
    plt.title(prefix_name)
    plt.savefig(save_dir / f, dpi=300)
    plt.close()
    return f

def make_gif(imgs, duration_secs, outname):
    head, *tail = [Image.fromarray((x * 255).astype(np.uint8)) for x in imgs]
    ms_per_frame = 1000 * duration_secs / instances
    head.save(outname, format='GIF', append_images=tail, save_all=True, duration=ms_per_frame, loop=0)

def make_mp4(imgs, duration_secs, outname):
    import shutil
    import subprocess as sp

    FFMPEG_BIN = shutil.which("ffmpeg")
    assert FFMPEG_BIN is not None, 'ffmpeg not found, install with "conda install -c conda-forge ffmpeg"'
    assert len(imgs[0].shape) == 3, 'Invalid shape of frame data'

    resolution = imgs[0].shape[0:2]
    fps = int(len(imgs) / duration_secs)

    command = [ FFMPEG_BIN,
                '-y', # overwrite output file
                '-f', 'rawvideo',
                '-vcodec' ,'rawvideo',
                '-s', f'{resolution[0]}x{resolution[1]}', # size of one frame
                '-pix_fmt', 'rgb24',
                '-r', f'{fps}',
                '-i', '-', # imput from pipe
                '-an', # no audio
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '17',
                str(Path(outname).with_suffix('.mp4')) ]

    #frame_data = np.concatenate([(x * 255).astype(np.uint8).reshape(-1) for x in imgs])
    frame_data = np.concatenate([(x).astype(np.uint8).reshape(-1) for x in imgs])
    with sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE) as p:
        ret = p.communicate(frame_data.tobytes())
        if p.returncode != 0:
            print(ret[1].decode("utf-8"))
            raise sp.CalledProcessError(p.returncode, command)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger