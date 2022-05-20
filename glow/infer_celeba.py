"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root> <z_dir> <dissec>
"""

import os, datetime
import cv2
import random
import torch
import torchvision.utils

import vision
import numpy as np
from docopt import docopt
from glow.builder import build
from glow.config import JsonConfig
from tqdm import tqdm
from pathlib import Path
import flowdissect_util
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms
import pickle
import logging
from termcolor import colored
from tabulate import tabulate
from functools import reduce
import torch_fidelity
from GlowWrapper4PPL import GenerativeModelWrap
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

is_debug = False
BOOTSTRAP = '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">'


def model_summary(model_list):
    output_shapes = model_list.flow.output_shapes
    if not isinstance(model_list, list):
        model_list = [model_list]

    from operator import mul
    param_statistics = dict()
    for i in range(0, len(output_shapes)):
        param_statistics[f'flow.layers.{i}'] = [0, output_shapes[i]]

    for model in model_list:
        data = []
        trainable_param_num = 0
        all_param_num = 0
        for key, value in model.named_parameters():
            _num = reduce(mul, list(value.size()), 1)
            data.append([key, list(value.size()), _num, value.requires_grad, value.dtype, value.device, value.is_leaf,
                         str(value.grad_fn)])
            all_param_num += _num

            for i in range(len(output_shapes)):
                if f'flow.layers.{i}.' in key:
                    param_statistics[f'flow.layers.{i}'][0] += _num

            if value.requires_grad:
                trainable_param_num += _num
        table = tabulate(data,
                         headers=["name", "shape", "numbers", "requires_grad", "dtype", "device", "is_leaf", "grad_fn"])
        logging.warning(" Arg Parameters: #param={}, #param(trainable) = {}".format(all_param_num, trainable_param_num))
        logging.info(colored(
            "Model Summary",
            "cyan",
        ))
        logging.info(
            "\n\n" +
            table
        )

        data_param = []
        param_list = []
        for key, value in param_statistics.items():
            data_param.append([key, value[0], value[1]])
            param_list.append(value[0])
        from glow.misc.wandb_layerwise_param import plot_param
        plot_param(np.array(param_list))

        table_param = tabulate(data_param,
                               headers=["name", "param_num", "output_shape"])
        logging.info(
            "\n\n" +
            table_param
        )
        logging.info(model)
        return all_param_num, trainable_param_num


def postprocess(x, n_bits=8):
    x = torch.clamp(x, 0, 1)
    # x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


def select_index(name, l, r, _input=None, description=None):
    index = None
    while index is None:
        print("Pls Input: Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            if _input is None:
                line = int(input().strip())
            else:
                line = int(_input)
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]).cuda(), eps_std=0.3, reverse=True)
    if False:
        img = x[0].permute(1, 2, 0).detach().cpu().numpy()
        img = img[:, :, ::-1]
        img = cv2.resize(img, (256, 256))

        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1)
    else:
        img = x[0].detach().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = torch.from_numpy(img)
        img = transforms.Resize(256)(img)  # .permute(1, 2, 0)
    return img


def run_z_interrupt(graph, interrupt_z, level_id, eps_std=0.3, larger_size=256):
    with torch.no_grad():
        graph.eval()
        if is_debug: print('from new layer_id {}'.format(level_id))
        x = graph.reverse_flow(z=torch.tensor(interrupt_z).cuda(),
                               y_onehot=None,
                               eps_std=eps_std,
                               dissec=dict(interrupt_z=level_id,
                                           interrupt_z_value=torch.tensor(interrupt_z).cuda()))
        img = x[0].detach().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = torch.from_numpy(img)
        img = transforms.Resize(larger_size)(img)  # .permute(1, 2, 0)
        return img


def run_z_interrupt_2x(graph, interrupt_z, level_id, eps_std=0.3):
    with torch.no_grad():
        graph.eval()
        if is_debug: print('from new layer_id {}'.format(level_id))
        x = graph.reverse_flow(z=torch.tensor(interrupt_z).cuda(),
                               y_onehot=None,
                               eps_std=eps_std,
                               dissec=dict(interrupt_z=level_id,
                                           interrupt_z_value=torch.tensor(interrupt_z).cuda()))
        img = x.detach().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = torch.from_numpy(img)
        return img

def run_z_interrupt_2z(graph, interrupt_z, level_id, dummy_x):
    with torch.no_grad():
        graph.eval()
        if is_debug: print('from new layer_id {}'.format(level_id))
        interrupt_z = torch.tensor(interrupt_z).cuda()
        z, nll, misc_dict = graph.normal_flow(x=dummy_x,
                                                y_onehot=None,
                                                dissec=dict(interrupt_z=level_id,
                                                            output_feat_detach=False,
                                                            interrupt_z_value=interrupt_z))
        nll = nll.detach().cpu().numpy()
        return nll

def run_z_interrupt_2z_opt(graph, interrupt_z, level_id, dummy_x, lr=1e+2, num_iter = 20):
        graph.eval().cuda()
        if is_debug: print('from new layer_id {}'.format(level_id))
        interrupt_z = torch.clone(interrupt_z).cuda()
        interrupt_z.requires_grad = True
        optim = torch.optim.SGD([interrupt_z],
                                lr=lr)  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=optim%20sgd#torch.optim.SGD
        for iter in range(num_iter):
            z, nll, misc_dict = graph.normal_flow(x=dummy_x,
                                                    y_onehot=None,
                                                    dissec=dict(interrupt_z=level_id,
                                                                output_feat_detach=False,
                                                                interrupt_z_value=interrupt_z))
            loss = nll.mean() + 0
            graph.zero_grad()
            optim.zero_grad()
            loss.backward()
            print(torch.norm(interrupt_z), torch.norm(interrupt_z.grad), loss)
            optim.step()
            #loss.backward(gradient=grad, retain_graph=True)
        interrupt_z.requires_grad = False


        with torch.no_grad():
            graph.eval().cuda()
            interrupt_zzz = torch.clone(interrupt_z).cuda()
            if True:
                z, nll, misc_dict = graph.normal_flow(x=dummy_x,
                                                      y_onehot=None,
                                                      dissec=dict(interrupt_z=level_id,
                                                                  output_feat_detach=False,
                                                                  interrupt_z_value=interrupt_zzz))
                nll = nll.detach().cpu().numpy()
                return nll
            else:
                x = graph.reverse_flow(z=torch.clone(interrupt_z).cuda(),
                                       y_onehot=None,
                                       eps_std=eps_std,
                                       dissec=dict(interrupt_z=level_id,output_feat_detach=False,
                                                   interrupt_z_value=interrupt_zzz))
                img = x.detach().cpu().numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                img = torch.from_numpy(img)
                return img




def save_images(images, names, log_root):
    save_dir = flowdissect_util.increment_path(Path(log_root) / 'infer')  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "{}.png".format(name)), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    dissec = args['<dissec>']

    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False

    hparams = JsonConfig("hparams/celeba_inf.json")
    log_root = hparams.Dir.log_root
    batch_size = hparams.Train.batch_size
    dataset = vision.Datasets["celeba"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # build
    graph = build(hparams, False)["graph"]
    if is_debug: print(graph);model_summary(graph)
    graph.eval()
    dataset = dataset(dataset_root, transform=transform)

    assert dissec in ['midz', 'midz_load', 'midz_allimg_load_depreciated', 'midz_allimg_vis_depreciated', 'midz_vis',
                      'mask_midz_vis',
                      'clever_midz_vis',
                      "clever_double_midz_vis",
                      "cal_fid",
                      "cal_nll",
                      'vis_feat', 'endz', 'rf',
                      'pca', 'pca_vis']

    import wandb
    wandb.init(project="taoflow-glow", entity='vincenthu')
    #wandb.run.summary["img_to_train"] = img_to_train

    if 'pca' in dissec:
        from flowdissect_pca import compute_pca

        if True:
            print('Not cached')
            level_id = 100  # 8, 16, 36, 72, 100
            n_components = 256
            output_shape = graph.flow.output_shapes[level_id]
            dump_path = Path(os.path.join(hparams.Dir.log_root, f'pca_dict_l{level_id}_c{n_components}.npz'))

        if dissec == 'pca':
            for level_id in [100, 72, 36, 16, 8]:
                print('Not cached')
                # layer_id = 100
                n_components = 256
                output_shape = graph.flow.output_shapes[level_id]
                dump_path = Path(os.path.join(hparams.Dir.log_root, f'pca_dict_l{level_id}_c{n_components}.npz'))
                t_start = datetime.datetime.now()
                X_comp, X_global_mean, X_stdev = compute_pca(
                    graph=graph,
                    layer_id=level_id,
                    estimator='ipca',
                    n_components=n_components,
                    batch_size=12,
                    n=1_000_000,
                    seed=88,
                    dump_path=dump_path,
                    ds=dataset,
                    is_debug=is_debug,
                )
                print('Total time:', datetime.datetime.now() - t_start)
            exit(0)
        elif dissec == 'pca_vis':
            print('loading from precached file.')
            data = np.load(dump_path, allow_pickle=False)  # does not contain object arrays
            X_comp = data['act_comp']
            X_global_mean = data['act_mean']
            X_stdev = data['act_stdev']
            X_var_ratio = data['var_ratio']
            # X_stdev_random = data['random_stdevs']
            n_comp = X_comp.shape[0]
            data.close()


        else:
            raise

        print('total var_ratio = {}'.format(np.sum(X_var_ratio)))
        X_comp = torch.from_numpy(X_comp).float()  # -1, 1, C, H, W
        X_global_mean = torch.from_numpy(X_global_mean).float()  # 1, C, H, W
        X_stdev = torch.from_numpy(X_stdev).float()

        with torch.no_grad():
            save_dir = flowdissect_util.increment_path(
                Path(hparams.Dir.log_root) / f'pca_vis_L{level_id}_')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir

            graph.eval()
            ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                   drop_last=True)
            for img_id, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):
                if img_id > 30 and is_debug: break
                z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                         dissec=dict(output_feat=True, output_feat_detach=True))

                layer_feat = z_list[level_id]['z'].reshape(batch_size, -1)
                layer_feat_meaned = layer_feat  # - X_global_mean

                normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-8)

                html = ''
                html += BOOTSTRAP

                for comp_id in range(16):
                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f' Component {comp_id}, X_stdev {X_stdev[comp_id]}, var_ratio {X_var_ratio[comp_id]} </div><br>'
                    magnitude = list(np.linspace(-3.0, 3.0, num=7))
                    image_interps = []

                    img_interrupted = run_z_interrupt(graph, interrupt_z=layer_feat.reshape(-1, output_shape[1],
                                                                                            output_shape[2],
                                                                                            output_shape[3]),
                                                      level_id=level_id,
                                                      eps_std=0.3, larger_size=256)
                    image_interps.append(img_interrupted)

                    for _mag_id, _mag in enumerate(magnitude):
                        # project to specific component
                        dotp = torch.sum(layer_feat_meaned * normalize(X_comp[comp_id]), dim=-1, keepdim=True)
                        zeroing_offset_lat = dotp * normalize(X_comp[comp_id])

                        # pick a random image, and generate z from image x
                        delta = _mag * X_stdev[comp_id] * X_comp[comp_id]
                        if False:
                            layer_feat_interrupted = layer_feat_meaned - zeroing_offset_lat + delta
                        else:
                            layer_feat_interrupted = layer_feat_meaned + delta
                        if False:
                            print('torch.norm(layer_feat)', torch.norm(layer_feat), _mag_id)
                            print('torch.norm(layer_feat_meaned)', torch.norm(layer_feat_meaned), _mag_id)
                            print('torch.norm(zeroing_offset_lat)', torch.norm(zeroing_offset_lat), _mag_id)
                            print('torch.norm(delta)', torch.norm(delta), _mag_id)
                            print('*' * 20)
                        layer_feat_interrupted = layer_feat_interrupted.reshape(-1, output_shape[1], output_shape[2],
                                                                                output_shape[3])
                        img_interrupted = run_z_interrupt(graph, interrupt_z=layer_feat_interrupted,
                                                          level_id=level_id,
                                                          eps_std=0.3, larger_size=256)
                        image_interps.append(img_interrupted)

                    grid = make_grid(image_interps[:], nrow=len(image_interps)).permute(1, 2, 0)
                    save_name = f'L{level_id}_img{img_id}_c{comp_id}.png'
                    html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                    html += f"<h2>{save_name}</h2><br><br>"
                    html += f"<hr/>"
                    (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                    print(f'Saving {save_dir / save_name}...')
                    with open(os.path.join(save_dir, f"00000_l{level_id}_img{img_id}.html"), "w") as outputfile:
                        outputfile.write(html)

    if dissec == 'midz':
        dataset = torch.utils.data.DataLoader(dataset,
                                              batch_size=hparams.Train.batch_size,
                                              num_workers=1,
                                              shuffle=False,
                                              drop_last=True)
        dissec_dict = dict(output_feat=True, output_feat_detach=True)
        ds_iter = iter(dataset)
        x = next(ds_iter)['x']
        z, nll, y_logits, z_list = graph.forward(x.cuda(), dissec=dissec_dict)
        print('a')
        exit(0)
    elif dissec == 'vis_feat':
        vis_feat = True
        vis_image_num = 10
        recompute = True
        class_num = 40
        feat_dict_path = os.path.join(hparams.Dir.log_root, f'midz_dict.pickle')
        if vis_feat:
            name = 'vis_feat'
            save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / name)  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            print(f'vis_feat dir: {save_dir}')
            ds_torch = torch.utils.data.DataLoader(dataset, batch_size=hparams.Train.batch_size, shuffle=False,
                                                   drop_last=True)
            for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):
                if i > 30 and is_debug: break
                z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                         dissec=dict(output_feat=True, output_feat_detach=True))
                html = ''
                html += BOOTSTRAP
                for vis_img_id in range(vis_image_num):
                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f' {vis_img_id} </div><br>'
                    for l_id in [8, 16, 24, 36, 72, 100]:
                        fname = flowdissect_util.feature_visualization_yolov5(
                            features=z_list[l_id]['z'][vis_img_id].unsqueeze(0),
                            prefix_name=f'batch{i}_img{vis_img_id}_l{l_id}', n=24, save_dir=save_dir,
                            orignal_img=postprocess(batch_data['x'][vis_img_id]))
                        html += f"<img src='{fname}'   class='img-fluid'   /><br>"
                        html += f"<h2>{fname}</h2><br><br>"
                        html += f"<hr/>"
                    with open(os.path.join(save_dir, f"00000_batch{i}.html"), "w") as outputfile:
                        outputfile.write(html)


    elif dissec == 'midz_load':
        vis_feat = False
        vis_image_num = 1
        recompute = True
        class_num = 40
        feat_dict_path = os.path.join(hparams.Dir.log_root, f'midz_dict_v2.pickle')
        if vis_feat:
            save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / 'vis_feat')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            print(f'vis_feat dir: {save_dir}')

        if (not os.path.exists(feat_dict_path)) or recompute:
            attr_dict = graph.generate_attr_deltaz_midz(dataset, feat_dict_path)
        else:
            print('loading pickle...')
            with open(feat_dict_path, 'rb') as f:
                attr_dict = pickle.load(f)

        print(attr_dict.keys())

    elif dissec == 'midz_allimg_load':
        vis_feat = False
        vis_image_num = 1
        recompute = True
        class_num = 40
        levels = [8, 36, 72, 100]
        attr_ids = [20, 8, 9, 15, 22, 31, 35, 33]
        feat_dict_path = os.path.join(hparams.Dir.log_root, f'midz_allimg_dict.pickle')
        if vis_feat:
            name = 'vis_feat'
            save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / name)  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            print(f'vis_feat dir: {save_dir}')

        if (not os.path.exists(feat_dict_path)) or recompute:
            attr_dict = graph.generate_midz_allimg(dataset, feat_dict_path, levels=levels, attr_ids=attr_ids)
        else:
            print('loading pickle...')
            with open(feat_dict_path, 'rb') as f:
                attr_dict = pickle.load(f)
        print(attr_dict.keys())

    elif dissec == 'midz_allimg_vis':
        print('loading pickle...')
        feat_dict_path = os.path.join(log_root, f'midz_dict_v2.pickle_iter16800')
        levels = 101
        class_num = 40
        attrs = 3  # pos, neg, delta
        if True:
            attr_ids = [20, 8, 9, 15, 22, 31, 35, 33]
            level_ids_exploring = [100, 72, 36, 8]
        else:
            attr_ids = [20, 8]
            level_ids_exploring = [100, 72, 36, 8]

        # attr_ids = [20, 8]
        # level_ids_exploring = [100]
        interplate_n = 4
        l2_magnitude = 0.3
        delta_scale = 4

        normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)
        norm2_sum = lambda v: torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)

        with open(feat_dict_path, 'rb') as f:
            attr_dict = pickle.load(f)

        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=hparams.Train.batch_size, shuffle=True,
                                               drop_last=True)
        for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
            if i > 30 and is_debug: break
            if i > 100: break
            z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                     dissec=dict(output_feat=True, output_feat_detach=True))
            save_dir = flowdissect_util.increment_path(
                Path(hparams.Dir.log_root) / 'new_midz_allimg_vis')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            for img_id in range(z.shape[0]):
                html = ""
                html += BOOTSTRAP

                for cls_id in attr_ids:
                    attr_name = dataset.attrs[cls_id]
                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f' {attr_name} </div><br>'
                    for level_id in level_ids_exploring:
                        _, channel_num, w_size, _ = graph.flow.output_shapes[level_id]
                        exp_num = 4
                        exp_delta = w_size // exp_num
                        for spatial_pos_x in range(0, w_size, exp_delta):
                            for spatial_pos_y in range(0, w_size, exp_delta):
                                boxes = (256 // w_size) * torch.tensor(
                                    [spatial_pos_y, spatial_pos_x, min(spatial_pos_y + exp_delta, w_size),
                                     min(spatial_pos_x + exp_delta, w_size)]).reshape(1, 4)

                                image_interps = []
                                common_feat_delta_init = attr_dict[f'level{level_id}_cls{cls_id}_delta']
                                common_feat_delta = np.zeros_like(common_feat_delta_init)
                                common_feat_delta[:, spatial_pos_x: spatial_pos_x + exp_delta,
                                spatial_pos_y:spatial_pos_y + exp_delta] = common_feat_delta_init[:,
                                                                           spatial_pos_x: spatial_pos_x + exp_delta,
                                                                           spatial_pos_y:spatial_pos_y + exp_delta]
                                current_feat = z_list[level_id]['z'][img_id]
                                assert len(current_feat.shape) == 3
                                current_feat_updated = current_feat
                                img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                                  level_id=level_id,
                                                                  eps_std=0.3, larger_size=256)
                                img_interrupted = torchvision.utils.draw_bounding_boxes(img_interrupted, boxes,
                                                                                        colors="#FF00FF")
                                image_interps.append(img_interrupted)

                                for iii in range(1, interplate_n):
                                    d = common_feat_delta * delta_scale * float(iii) / float(interplate_n + 1)
                                    # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                                    for _signal in [-1, 1]:
                                        current_feat_updated = current_feat + d * (_signal)
                                        if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                                        current_feat_updated = current_feat_updated.unsqueeze(0)
                                        img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                                          level_id=level_id,
                                                                          eps_std=0.3, larger_size=256)
                                        img_interrupted = torchvision.utils.draw_bounding_boxes(img_interrupted, boxes,
                                                                                                colors="#FF00FF")
                                        image_interps.append(img_interrupted)
                                grid = make_grid(image_interps[:], nrow=len(image_interps)).permute(1, 2, 0)
                                save_name = f'img{img_id}_{attr_name}{cls_id}_highlevelfeat{level_id}_position{spatial_pos_x}_{spatial_pos_y}_size{exp_delta}.png'
                                html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                                html += f"<h2>{save_name}</h2><br><br>"
                                (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                                print(f'Saving {save_dir / save_name}...')

                        exp_num = 3
                        exp_delta = channel_num // exp_num
                        for channel_pos in [0, 1]:
                            image_interps = []
                            common_feat_delta_init = attr_dict[f'level{level_id}_cls{cls_id}_delta']
                            common_feat_delta = np.zeros_like(common_feat_delta_init)
                            common_feat_delta[channel_pos:channel_pos + exp_delta] = common_feat_delta_init[
                                                                                     channel_pos:channel_pos + exp_delta]

                            current_feat = z_list[level_id]['z'][img_id]
                            assert len(current_feat.shape) == 3
                            current_feat_updated = current_feat
                            img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                              level_id=level_id,
                                                              eps_std=0.3)
                            image_interps.append(img_interrupted)

                            for iii in range(1, interplate_n):
                                d = common_feat_delta * delta_scale * float(iii) / float(interplate_n + 1)
                                # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                                for _signal in [-1, 1]:
                                    current_feat_updated = current_feat + d * (_signal)
                                    if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                                    current_feat_updated = current_feat_updated.unsqueeze(0)
                                    img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                                      level_id=level_id,
                                                                      eps_std=0.3)
                                    image_interps.append(img_interrupted)
                            grid = make_grid(image_interps[:], nrow=len(image_interps)).permute(1, 2, 0)
                            save_name = f'img{img_id}_{attr_name}{cls_id}_highlevelfeat{level_id}_cid{channel_pos}_{exp_delta}.png'
                            html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                            html += f"<h2>{save_name}</h2><br><br>"
                            (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(
                                os.path.join(save_dir, save_name))
                            print(f'Saving {save_dir / save_name}...')

                with open(os.path.join(save_dir, f"00000_img{img_id}_ds{delta_scale}.html"), "w") as outputfile:
                    outputfile.write(html)
                if img_id > 10 and True: exit(0)
                print('aaaaaa')

    elif dissec == 'midz_vis':
        print('loading pickle...')
        feat_dict_path = os.path.join(log_root, f'midz_dict_v2.pickle_iter16800')
        levels = 101
        class_num = 40
        attrs = 3  # pos, neg, delta
        attr_ids = [20, 8, 9, 15, 22, 31, 35, 33]
        level_ids_exploring = [8, 16, 32, 64, 96, 100]
        # attr_ids = [20, 8]
        # level_ids_exploring = [100]
        interplate_n = 4
        l2_magnitude = 0.3
        delta_scale = 4

        normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)
        norm2_sum = lambda v: torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)

        with open(feat_dict_path, 'rb') as f:
            attr_dict = pickle.load(f)
        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=hparams.Train.batch_size, shuffle=True,
                                               drop_last=True)
        for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
            if i > 30 and is_debug: break
            z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                     dissec=dict(output_feat=True, output_feat_detach=True))
            save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / 'midz_vis_v2_')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            for img_id in range(z.shape[0]):
                html = ""
                html += BOOTSTRAP

                for cls_id in attr_ids:
                    attr_name = dataset.attrs[cls_id]
                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f' {attr_name} </div><br>'
                    for level_id in level_ids_exploring:
                        image_interps = []
                        common_feat_delta = attr_dict[f'level{level_id}_cls{cls_id}_delta']
                        common_feat_pos_mean = attr_dict[f'level{level_id}_cls{cls_id}_pos']
                        common_feat_neg_mean = attr_dict[f'level{level_id}_cls{cls_id}_neg']
                        current_feat = z_list[level_id]['z'][img_id]
                        assert len(current_feat.shape) == 3
                        current_feat_updated = current_feat
                        img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                          level_id=level_id,
                                                          eps_std=0.3)
                        image_interps.append(img_interrupted)

                        # for iii in range(1, interplate_n):
                        for delta_step in list(np.linspace(-4, 4, num=9)):
                            d = common_feat_delta * delta_scale * delta_step / float(interplate_n + 1)
                            # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                            current_feat_updated = current_feat + d
                            if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                            current_feat_updated = current_feat_updated.unsqueeze(0)
                            img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                              level_id=level_id,
                                                              eps_std=0.3)
                            image_interps.append(img_interrupted)

                        grid = make_grid(image_interps[:], nrow=len(image_interps), padding=15, pad_value=255).permute(
                            1, 2, 0)
                        save_name = f'img{img_id}_{attr_name}{cls_id}_level{level_id}.png'
                        html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                        html += f"<h2>{save_name}</h2><br><br>"
                        (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                        print(f'Saving {save_dir / save_name}...')
                        with open(os.path.join(save_dir, f"00000_img{img_id}_ds{delta_scale}.html"), "w") as outputfile:
                            outputfile.write(html)
                        if False:
                            fig = plt.figure()
                            ax = plt.subplot(len(image_interps), 1, 1)
                            ax.axis('off')
                            # transforms.ToPILImage()(img)
                            plt.imshow(grid)
                            # ax.title.set_text(f'{attr_name}', fontsize=10)

                            ax.set_title(img_name, fontsize=3)
                            # ax.set_xlabel(name)

                            f = f"{img_name}.png"
                            plt.tight_layout()
                            plt.savefig(save_dir / f, dpi=300)
                            plt.close()

                if img_id > 10 and True: exit(0)
                print('aaaaaa')

    elif dissec == 'mask_midz_vis':
        print('loading pickle...')
        feat_dict_path = os.path.join(log_root, f'midz_dict_v2.pickle_iter16800')
        class_num = 40
        attrs = 3  # pos, neg, delta
        attr_ids = [i for i in range(class_num)]  # ['Male', 'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache', 'Smiling', 'Wearing_Hat', 'Wavy_Hair']
        level_ids_exploring = [8, 16, 32, 48, 64, 80, 96, 100]
        attr_names = [dataset.attrs[id] for id in attr_ids]
        print(dataset.attrs)
        mask_name_list = ['hair', 'skin', 'mouth', 'nose', 'hat', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g']]
        # ['class0_placeholder', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        # attr_ids = [20, 8]
        # level_ids_exploring = [100]
        target_couple = [
                         ('Black_Hair', 'hair'),
                         ('Wavy_Hair', 'hair'),
                         ('Straight_Hair', 'hair'),
                         #('Wearing_Hat', 'hat'),
                         ('Smiling', 'skin'),
                         ('Pointy_Nose', 'skin'),
                         ('Pale_Skin', 'skin'),

                         ('No_Beard', 'skin'),
                         ('Mustache', 'skin'),
                         ('Mouth_Slightly_Open', 'skin'),
                         ('Big_Lips', 'skin'),


                         ('Male', 'skin'),
                         ('Young', 'skin'),
                         #('Eyeglasses', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'])
                         ]
        target_couple_full = [(attr, 'full_img') for attr, _ in target_couple]
        target_couple.extend(target_couple_full)
        interplate_n = 4
        l2_magnitude = 0.3
        eps_std = 0.3
        delta_scale = 1
        batch_size = hparams.Train.batch_size

        output_shapes = [graph.flow.output_shapes[level_id] for level_id in level_ids_exploring]

        normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)
        norm2_sum = lambda v: torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)

        with open(feat_dict_path, 'rb') as f:
            attr_dict = pickle.load(f)
        from vision.datasets.celebamask30k_1024 import CelebAMaskHQ256

        dataset = CelebAMaskHQ256(n_bits=8, img_size=64, mode=True)
        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
            if i > 30 and is_debug: break
            z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                     dissec=dict(output_feat=True, output_feat_detach=True))
            save_dir = flowdissect_util.increment_path(
                Path(hparams.Dir.log_root) / 'mask_midz_vis_v2_')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            for img_id in range(z.shape[0]):
                html = ""
                html += BOOTSTRAP

                for attr_name, mask_name in target_couple:
                    attr_id = attr_ids[attr_names.index(attr_name)]

                    target_mask = torch.zeros_like(batch_data['seg'])
                    if isinstance(mask_name, str):
                        if mask_name=='full_img':
                            target_mask = 1
                        else:
                            mask_id = dataset.mask_name_list.index(mask_name)
                            target_mask[batch_data['seg'] == mask_id] = 1
                    elif isinstance(mask_name, list):
                        for _mask_name in mask_name:
                            mask_id = dataset.mask_name_list.index(_mask_name)
                            target_mask[batch_data['seg'] == mask_id] = 1
                        mask_name = '-'.join(mask_name)
                    else:
                        raise

                    ###########

                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f' interp {attr_name} on area {mask_name} </div><br>'
                    for level_id in level_ids_exploring:

                        _target_mask_resized = (transforms.Resize(graph.flow.output_shapes[level_id][-1],
                                                                  interpolation=transforms.InterpolationMode.NEAREST)(
                            target_mask[img_id:img_id + 1])).squeeze(1)  # remove channel number, is 1
                        _target_mask_resized = _target_mask_resized.numpy()

                        image_interps = []
                        common_feat_delta = attr_dict[f'level{level_id}_cls{attr_id}_delta']
                        # common_feat_pos_mean = attr_dict[f'level{level_id}_cls{attr_id}_pos']
                        # common_feat_neg_mean = attr_dict[f'level{level_id}_cls{attr_id}_neg']
                        current_feat = z_list[level_id]['z'][img_id]
                        assert len(current_feat.shape) == 3
                        current_feat_updated = current_feat
                        img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                          level_id=level_id,
                                                          eps_std=eps_std, larger_size=256)
                        if True:
                            vis_im = img_interrupted.permute(1, 2, 0).contiguous().numpy()
                            vis_parsing_anno_color = \
                                (transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)(
                                    target_mask))[
                                    img_id]
                            vis_parsing_anno_color = vis_parsing_anno_color.permute(1, 2, 0).numpy()
                            vis_parsing_anno_color = np.repeat(vis_parsing_anno_color, 3, axis=2) * 255
                            # vis_parsing_anno_color[:,:] = np.array([255,0,0]).reshape(1,3)
                            img_interrupted = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4,
                                                              vis_parsing_anno_color, 0.6, 0)
                            img_interrupted = cv2.cvtColor(img_interrupted, cv2.COLOR_BGR2RGB)
                            img_interrupted = torch.from_numpy(img_interrupted).permute(2, 0, 1).contiguous()

                        image_interps.append(img_interrupted)

                        for delta_step in list(np.linspace(-interplate_n, interplate_n, num=2*interplate_n+1)):
                            d = common_feat_delta * _target_mask_resized * delta_scale * delta_step
                            # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                            current_feat_updated = current_feat + d
                            if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                            current_feat_updated = current_feat_updated.unsqueeze(0)
                            img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                              level_id=level_id,
                                                              eps_std=eps_std)
                            image_interps.append(img_interrupted)

                        grid = make_grid(image_interps[:], nrow=len(image_interps), padding=15, pad_value=255).permute(
                            1, 2, 0)
                        save_name = f'img{img_id}_interp_{attr_name}{attr_id}_onarea_{mask_name}_level{level_id}.png'
                        html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                        html += f"<h2>{save_name}</h2><br><br>"
                        (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                        print(f'Saving {save_dir / save_name}...')
                        with open(os.path.join(save_dir, f"00000_img{img_id}_ds{delta_scale}.html"), "w") as outputfile:
                            outputfile.write(html)

                if img_id > 10 and True: exit(0)
                print('aaaaaa')

    elif dissec == 'clever_midz_vis':
        if True:
            hparams = JsonConfig("hparams/clever_das6.json")
            log_root = hparams.Dir.log_root
            batch_size = hparams.Train.batch_size
            # build
            graph = build(hparams, False)["graph"]
            if is_debug: print(graph);model_summary(graph)
            graph.eval()

        levels = 64
        class_num = 40
        interplate_n = 4
        l2_magnitude = 0.3
        delta_scale = 1
        level_ids_exploring = [100, 72, 36, 8]
        output_shapes = [graph.flow.output_shapes[level_id] for level_id in level_ids_exploring]

        normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)
        norm2_sum = lambda v: torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)

        if True:
            if True:
                # dataset_root = "/home/thu/dataset/CLEVR_v1.0/images/train"
                dataset_root = "/home/thu/dataset/CLEVR_v1.0/images/val"
                from vision.datasets.clever import CleverDataset

                # set transform of dataset
                transform = transforms.Compose([
                    transforms.CenterCrop(hparams.Data.center_crop),
                    transforms.Resize(hparams.Data.resize),
                    transforms.ToTensor()])
                # build graph and dataset
                dataset = CleverDataset(dataset_root, transform=transform)
                ds_torch = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
                                                       drop_last=True)
                # elif True:
                dataset_root = "/home/thu/lab/stylegan3_main/chaiyujin-glow/clevr-dataset-gen-main/clever_single1k/images"
                from vision.datasets.clever import CleverDataset

                # set transform of dataset
                transform = transforms.Compose([
                    transforms.CenterCrop(hparams.Data.center_crop),
                    transforms.Resize(hparams.Data.resize),
                    transforms.ToTensor()])
                # build graph and dataset
                dataset = CleverDataset(dataset_root, transform=transform)
                ds_torch_delta = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
                                                             drop_last=True)

        # for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
        for i, (batch_data, batch_data_delta) in tqdm(enumerate(zip(ds_torch, ds_torch_delta)),
                                                      total=min(len(ds_torch), len(ds_torch_delta)),
                                                      desc='explore images'):
            if i > 30 and is_debug: break
            z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                     dissec=dict(output_feat=True, output_feat_detach=True))
            save_dir = flowdissect_util.increment_path(
                Path(hparams.Dir.log_root) / 'clever_midz_vis_v1_')  # increment run
            save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            html = ""
            html += BOOTSTRAP
            ###########
            html += f"<div class='alert alert-primary' role='alert'>" \
                    f' interp  </div><br>'
            for img_id in range(0, z.shape[0], 2):
                img_id2 = img_id + 1
                for level_id in level_ids_exploring:
                    image_interps = []
                    current_feat = z_list[level_id]['z'][img_id]
                    current_feat_2 = z_list[level_id]['z'][img_id2]
                    assert len(current_feat.shape) == 3
                    img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                      level_id=level_id,
                                                      eps_std=0.3)
                    image_interps.append(img_interrupted)

                    img_interrupted = run_z_interrupt(graph,
                                                      interrupt_z=current_feat_2.unsqueeze(0),
                                                      level_id=level_id,
                                                      eps_std=0.3)
                    image_interps.append(img_interrupted)

                    for delta_step in list(np.linspace(-4, 4, num=9)):
                        # d = common_feat_delta * _target_mask_resized * delta_scale * delta_step
                        # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                        # d = delta_scale * delta_step * current_feat_2
                        d = delta_scale * delta_step * current_feat_2.mean([1, 2], keepdim=True)
                        current_feat_updated = current_feat + d
                        if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                        current_feat_updated = current_feat_updated.unsqueeze(0)
                        img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                          level_id=level_id,
                                                          eps_std=0.3)
                        image_interps.append(img_interrupted)

                    grid = make_grid(image_interps[:], nrow=len(image_interps), padding=15, pad_value=255).permute(
                        1, 2, 0)
                    save_name = f'img{img_id}_interp_level{level_id}.png'
                    html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                    html += f"<h2>{save_name}</h2><br><br>"
                    (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                    print(f'Saving {save_dir / save_name}...')
                    with open(os.path.join(save_dir, f"00000_batch{i}_ds{delta_scale}.html"), "w") as outputfile:
                        outputfile.write(html)

                if img_id > 1000 and True: exit(0)
                print('aaaaaa')

    elif dissec == 'clever_double_midz_vis':
        if True:
            hparams = JsonConfig("hparams/clever_das6.json")
            log_root = hparams.Dir.log_root
            batch_size = hparams.Train.batch_size
            # build
            graph = build(hparams, is_training=False)["graph"]
            if is_debug: print(graph);model_summary(graph)
            graph.eval()

        levels = 64
        class_num = 40
        interplate_n = 4
        l2_magnitude = 0.3
        delta_scale = 1
        level_ids_exploring = [100, 72, 36, 8]
        output_shapes = [graph.flow.output_shapes[level_id] for level_id in level_ids_exploring]

        normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)
        norm2_sum = lambda v: torch.sqrt(torch.sum(v ** 2, dim=[0, 1, 2], keepdim=True) + 1e-8)

        if False:
            with torch.no_grad():
                save_dir = flowdissect_util.increment_path(
                    Path(hparams.Dir.log_root) / 'clever_samples')  # increment run
                save_dir.mkdir(parents=True, exist_ok=True)  # make dir
                print(save_dir)
                img = graph(z=None, y_onehot=None, eps_std=0.5, reverse=True)
                img = (torch.clamp(img, min=0, max=1.0) * 255).type(torch.uint8)
                grid = make_grid(img, nrow=int(math.sqrt(img.shape[0])), padding=8, pad_value=255)
                pil_img = (transforms.ToPILImage())(grid)
                pil_img.save(os.path.join(save_dir, '1.png'))
                exit(0)

        if True:
            if True:
                # dataset_root = "/home/thu/dataset/CLEVR_v1.0/images/train"
                dataset_root = "/home/thu/dataset/CLEVR_v1.0/images/val"
                from vision.datasets.clever import CleverDataset

                # set transform of dataset
                transform = transforms.Compose([
                    transforms.CenterCrop(hparams.Data.center_crop),
                    transforms.Resize(hparams.Data.resize),
                    transforms.ToTensor()])
                # build graph and dataset
                dataset = CleverDataset(dataset_root, transform=transform)
                ds_torch = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                                       drop_last=True)
                # elif True:
                dataset_root = "/home/thu/lab/stylegan3_main/chaiyujin-glow/clevr-dataset-gen-main/clever_single1k/images"
                from vision.datasets.clever import CleverDataset

                # set transform of dataset
                transform = transforms.Compose([
                    transforms.CenterCrop(hparams.Data.center_crop),
                    transforms.Resize(hparams.Data.resize),
                    transforms.ToTensor()])
                # build graph and dataset
                dataset = CleverDataset(dataset_root, transform=transform)
                ds_torch_delta = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                                             drop_last=True)

        save_dir = flowdissect_util.increment_path(
            Path(hparams.Dir.log_root) / 'clever_double_midz_vis_v1_')  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
        for i, (batch_data, batch_data_delta) in tqdm(enumerate(zip(ds_torch, ds_torch_delta)),
                                                      total=min(len(ds_torch), len(ds_torch_delta)),
                                                      desc='explore images'):
            if i > 30 and is_debug: break
            z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                     dissec=dict(output_feat=True, output_feat_detach=True))

            delta_z, delta_nll, delta_y_logits, delta_z_list = graph.forward(batch_data_delta['x'].cuda(),
                                                                             dissec=dict(output_feat=True,
                                                                                         output_feat_detach=True))

            html = ""
            html += BOOTSTRAP
            ###########
            html += f"<div class='alert alert-primary' role='alert'>" \
                    f' interp  </div><br>'

            for img_id in range(0, z.shape[0], 1):
                for level_id in level_ids_exploring:
                    image_interps = []
                    current_feat = z_list[level_id]['z'][img_id]
                    current_feat_2 = delta_z_list[level_id]['z'][img_id]
                    assert len(current_feat.shape) == 3
                    img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat.unsqueeze(0),
                                                      level_id=level_id,
                                                      eps_std=0.3)
                    image_interps.append(img_interrupted)

                    img_interrupted = run_z_interrupt(graph,
                                                      interrupt_z=current_feat_2.unsqueeze(0),
                                                      level_id=level_id,
                                                      eps_std=0.3)
                    image_interps.append(img_interrupted)

                    for delta_step in list(np.linspace(-4, 4, num=9)):
                        # d = common_feat_delta * _target_mask_resized * delta_scale * delta_step
                        # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                        # d = delta_scale * delta_step * current_feat_2
                        d = delta_scale * delta_step * current_feat_2.mean([1, 2], keepdim=True)
                        current_feat_updated = current_feat + d
                        if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                        current_feat_updated = current_feat_updated.unsqueeze(0)
                        img_interrupted = run_z_interrupt(graph, interrupt_z=current_feat_updated,
                                                          level_id=level_id,
                                                          eps_std=0.3)
                        image_interps.append(img_interrupted)

                    grid = make_grid(image_interps[:], nrow=len(image_interps), padding=15, pad_value=255).permute(
                        1, 2, 0)
                    save_name = f'img{img_id}_interp_level{level_id}.png'
                    html += f"<img src='{save_name}'   class='img-fluid'   /><br>"
                    html += f"<h2>{save_name}</h2><br><br>"
                    (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                    print(f'Saving {save_dir / save_name}...')
                    with open(os.path.join(save_dir, f"00000_batch{i}_ds{delta_scale}.html"), "w") as outputfile:
                        outputfile.write(html)

                if img_id > 10 and True: exit(0)
                print('aaaaaa')


    elif dissec == 'cal_nll':
        print('loading pickle...')
        feat_dict_path = os.path.join(log_root, f'midz_dict_v2.pickle_iter16800')
        class_num = 40
        attrs = 3  # pos, neg, delta
        attr_ids = [20, 8, 9, 15, 22, 31, 35,
                    33]  # ['Male', 'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache', 'Smiling', 'Wearing_Hat', 'Wavy_Hair']
        level_ids_exploring = [8, 16, 32, 64, 96, 100]
        attr_names = [dataset.attrs[id] for id in attr_ids]
        mask_name_list = ['hair', 'skin', 'mouth', 'nose', 'hat', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g']]
        # ['class0_placeholder', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        # attr_ids = [20, 8]
        # level_ids_exploring = [100]
        target_couple = [('Male', 'hair'), ('Male', 'skin'), ('Black_Hair', 'hair'),
                         ('Wearing_Hat', 'hair'), ('Wearing_Hat', 'hat'),
                         ('Wearing_Hat', 'hair'), ('Smiling', 'skin'),
                         ('Smiling', 'mouth'),
                         ('Eyeglasses', 'skin'),
                         ('Eyeglasses', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'])]

        output_shapes = [graph.flow.output_shapes[level_id] for level_id in level_ids_exploring]



        with open(feat_dict_path, 'rb') as f:
            attr_dict = pickle.load(f)
        from vision.datasets.celebamask30k_1024 import CelebAMaskHQ256

        dataset = CelebAMaskHQ256(n_bits=8, img_size=64, mode=True)
        batch_size = hparams.Train.batch_size
        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)

        interplate_n = 4
        l2_magnitude = 0.3
        delta_scale = 1
        ref_num = 5000
        assert len(ds_torch)>=ref_num//batch_size
        print(f'running batch number = {ref_num//batch_size}, ref_image_num = {ref_num}')


        #save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / 'cal_nll_v1_')  # increment run
        #save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        nll_refs = []
        nll_interps = dict()

        graph.eval()

        try:

            for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
                if i > ref_num and True: break
                with torch.no_grad():
                    graph.eval()
                    z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                         dissec=dict(output_feat=True, output_feat_detach=True))
                nll_refs.append(nll.detach().cpu().numpy())
                dummy_x = batch_data['x'].cuda()
                for attr_name, mask_name in target_couple:
                    attr_id = attr_ids[attr_names.index(attr_name)]
                    target_mask = torch.zeros_like(batch_data['seg'])
                    if isinstance(mask_name, str):
                        mask_id = dataset.mask_name_list.index(mask_name)
                        target_mask[batch_data['seg'] == mask_id] = 1
                    elif isinstance(mask_name, list):
                        for _mask_name in mask_name:
                            mask_id = dataset.mask_name_list.index(_mask_name)
                            target_mask[batch_data['seg'] == mask_id] = 1
                        mask_name = '-'.join(mask_name)
                    else:raise
                    ##############
                    for level_id in level_ids_exploring:
                        dict_key_name = f'{attr_name}_{mask_name}_L{level_id}'
                        if dict_key_name not in nll_interps:
                            nll_interps[dict_key_name] = dict(level=level_id, attr_name=attr_name, mask_name=mask_name)
                            nll_interps[dict_key_name]['result'] = list()
                        output_shape = graph.flow.output_shapes[level_id]
                        batch_masked_feat = torch.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]))
                        batch_untact_feat = torch.zeros_like(batch_masked_feat)
                        for img_id in range(batch_size):
                            _target_mask_resized = (transforms.Resize(graph.flow.output_shapes[level_id][-1],
                                                                      interpolation=transforms.InterpolationMode.NEAREST)(
                                target_mask[img_id:img_id + 1])).squeeze(1)  # remove channel number, is 1
                            common_feat_delta = torch.from_numpy(attr_dict[f'level{level_id}_cls{attr_id}_delta'])* _target_mask_resized
                            batch_masked_feat[img_id] = common_feat_delta
                            batch_untact_feat[img_id] = z_list[level_id]['z'][img_id]


                        #nll = run_z_interrupt_2z_opt(graph, interrupt_z=batch_untact_feat,level_id=level_id,dummy_x=dummy_x)

                        for delta_step in list(np.linspace(-interplate_n, interplate_n, num=2*interplate_n+1)):
                            d = batch_masked_feat * delta_scale * delta_step
                            # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                            current_feat_updated = batch_untact_feat + d
                            if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                            nll_interrupted = run_z_interrupt_2z(graph, interrupt_z=current_feat_updated,
                                                               level_id=level_id,
                                                               dummy_x=dummy_x)
                            nll_interps[dict_key_name]['result'].append(nll_interrupted)
        except KeyboardInterrupt:
            print('KeyboardInterrupt exception is caught, continue')

        #NLL for ref
        wandb_table = wandb.Table(
            columns=['name', 'level', 'attr_name', 'mask_name', 'nll', 'num_ref', 'num_gen'])
        nll_refs = np.concatenate(nll_refs)
        num_ref = nll_refs.shape[0]
        nll_refs = np.mean(nll_refs)
        for _key, _value in nll_interps.items():
            tmp = np.concatenate(_value['result'])
            num_gen = tmp.shape[0]
            nll_interps[_key] = np.mean(tmp)
            wandb_table.add_data(_key, _value['level'], _value['attr_name'], _value['mask_name'], nll_interps[_key], num_ref,
                             num_gen)
            print('*' * 38)
        _dict = dict()
        _dict['fid_hyperparam'] = wandb_table
        wandb.log(_dict)

        print(nll_refs)
        print(nll_interps)

            #NLL for generated
        #print(f'Saving {save_dir / save_name}...')
        # calcuate NLL


    elif dissec == 'cal_fid':
        print('loading pickle...')
        model_summary(graph)
        feat_dict_path = os.path.join(log_root, f'midz_dict_v2.pickle_iter16800')
        class_num = 40
        attrs = 3  # pos, neg, delta
        attr_ids = [20, 8, 9, 15, 22, 31, 35,
                    33]  # ['Male', 'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache', 'Smiling', 'Wearing_Hat', 'Wavy_Hair']
        level_ids_exploring = [100]#[8, 16, 32, 64, 96, 100]
        attr_names = [dataset.attrs[id] for id in attr_ids]
        mask_name_list = ['hair', 'skin', 'mouth', 'nose', 'hat', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g']]
        # ['class0_placeholder', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        # attr_ids = [20, 8]
        # level_ids_exploring = [100]
        if False:
            target_couple = [('Male', 'hair'), ('Male', 'skin'), ('Black_Hair', 'hair'),
                             ('Wearing_Hat', 'hair'), ('Wearing_Hat', 'hat'),
                             ('Smiling', 'skin'),
                             ('Smiling', 'mouth'),
                             ('Eyeglasses', 'skin'),
                             ('Eyeglasses', ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'])]
        else:
            target_couple = [('Male', 'hair'),
                             #('Male', 'skin'), ('Black_Hair', 'hair'),
                            # ('Wearing_Hat', 'hair'),('Smiling', 'skin'),
                            # ('Eyeglasses', 'skin')
                             ]

        output_shapes = [graph.flow.output_shapes[level_id] for level_id in level_ids_exploring]

        with open(feat_dict_path, 'rb') as f:
            attr_dict = pickle.load(f)
        from vision.datasets.celebamask30k_1024 import CelebAMaskHQ256


        dataset = CelebAMaskHQ256(n_bits=8, img_size=64, mode=True)
        batch_size = 200#hparams.Train.batch_size
        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        interplate_n = 2
        l2_magnitude = 0.3
        delta_scale = 1
        ref_num = len(ds_torch)*batch_size#3000
        ref_num = 5000
        is_analysis_logdet = True
        assert len(ds_torch)>=ref_num//batch_size
        print(f'running batch number = {ref_num//batch_size}, ref_image_num = {ref_num}')

        if is_analysis_logdet:
            columns = ['batch_id', 'level_id', 'logdet']
            wandb_table_logdet = wandb.Table(columns=columns)



        save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / 'cal_fid_v1_')  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        print(save_dir)

        img_ref_dir = Path(os.path.join(save_dir, 'refs'))
        img_ref_dir.mkdir(parents=True, exist_ok=True)  # make dir

        nll_refs = []
        nll_interps = dict()
        interp_id=0
        ref_id = 0

        with torch.no_grad():
            try:
                graph.eval()
                for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='explore images'):
                    if i*batch_size > ref_num and True: break
                    if i>10:break

                    for _ in range(batch_size):
                        (transforms.ToPILImage()(batch_data['x'][_])).save(
                            os.path.join(img_ref_dir, f'{ref_id}.png'));
                        ref_id += 1

                    z, nll, y_logits, z_list = graph.forward(batch_data['x'].cuda(),
                                                             dissec=dict(output_feat=True, output_feat_detach=True))
                    if is_analysis_logdet:
                        print(torch.mean(y_logits['logpx']))
                        print(torch.mean(y_logits['logpz']))
                        print(torch.mean(y_logits['logdet']))
                        for _de_levelid in range(len(graph.flow.output_shapes)):
                            logdet = torch.mean(z_list[_de_levelid]['logdet']/y_logits['pixels']).item()
                            wandb_table_logdet.add_data(*[i, _de_levelid, logdet])
                        wandb.log({f'logdet_batch_{i}':wandb_table_logdet})
                        print('is_analysis_logdet done')
                    nll_refs.append(nll.detach().cpu().numpy())
                    dummy_x = batch_data['x'].cuda()
                    for attr_name, mask_name in target_couple:
                        attr_id = attr_ids[attr_names.index(attr_name)]
                        target_mask = torch.zeros_like(batch_data['seg'])
                        if isinstance(mask_name, str):
                            mask_id = dataset.mask_name_list.index(mask_name)
                            target_mask[batch_data['seg'] == mask_id] = 1
                        elif isinstance(mask_name, list):
                            for _mask_name in mask_name:
                                mask_id = dataset.mask_name_list.index(_mask_name)
                                target_mask[batch_data['seg'] == mask_id] = 1
                            mask_name = '-'.join(mask_name)
                        else:raise
                        ##############
                        for level_id in level_ids_exploring:
                            dict_key_name = f'{attr_name}_{mask_name}_L{level_id}'
                            new_dir = Path(os.path.join(save_dir, dict_key_name))
                            if dict_key_name not in nll_interps:
                                new_dir.mkdir(parents=True, exist_ok=True)  # make dir
                                nll_interps[dict_key_name] = dict(dir_name=new_dir, level=level_id, attr_name=attr_name, mask_name=mask_name)

                            output_shape = graph.flow.output_shapes[level_id]
                            batch_masked_feat = torch.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]))
                            batch_untact_feat = torch.zeros_like(batch_masked_feat)
                            for img_id in range(batch_size):
                                _target_mask_resized = (transforms.Resize(graph.flow.output_shapes[level_id][-1],
                                                                          interpolation=transforms.InterpolationMode.NEAREST)(
                                    target_mask[img_id:img_id + 1])).squeeze(1)  # remove channel number, is 1
                                common_feat_delta = torch.from_numpy(attr_dict[f'level{level_id}_cls{attr_id}_delta'])* _target_mask_resized
                                batch_masked_feat[img_id] = common_feat_delta
                                batch_untact_feat[img_id] = z_list[level_id]['z'][img_id]


                            #img_interrupted = run_z_interrupt_2x(graph, interrupt_z=batch_untact_feat,level_id=level_id, eps_std=0.3)

                            for delta_step in list(np.linspace(-interplate_n, interplate_n, num=2*interplate_n+1)):
                                d = batch_masked_feat * delta_scale * delta_step
                                # d = norm2_sum(current_feat)*l2_magnitude*normalize(common_feat_delta)* float(iii) / float(interplate_n + 1)
                                current_feat_updated = batch_untact_feat + d
                                if is_debug: print(norm2_sum(current_feat), norm2_sum(current_feat_updated))
                                img_interrupted = run_z_interrupt_2x(graph, interrupt_z=current_feat_updated,
                                                                     level_id=level_id, eps_std=0.3)
                                for _ in range(batch_size):
                                    (transforms.ToPILImage()(img_interrupted[_])).save(os.path.join(nll_interps[dict_key_name]['dir_name'], f'{interp_id}.png')); interp_id+=1

                    if is_analysis_logdet:exit(0)

            except KeyboardInterrupt:
                print('KeyboardInterrupt exception is caught, continue')



            # create a wandb.Table() with corresponding columns
            wandb_table = wandb.Table(columns=['name', 'level', 'attr_name','mask_name','fid', 'ics', 'kid','ppl', 'num_ref', 'num_gen'])
            # calculate FID, IS, PPL, KID per folder-pair
            for _key, _value in nll_interps.items():
                cur_dir = _value['dir_name']
                print('https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metrics.py')
                graph_wrapper = GenerativeModelWrap(graph, level_id=100)
                metrics_dict_ppl = torch_fidelity.calculate_metrics(input1=graph_wrapper,
                                                                    input2=graph_wrapper,
                                                                    cuda=True, isc=False, fid=False, kid=False,
                                                                    ppl=True,
                                                                    verbose=True)
                print(metrics_dict_ppl)
                ppl = metrics_dict_ppl['perceptual_path_length_mean']


                print(f'calculate statistics from {_key}， interp_dir {cur_dir}, ref_dir {img_ref_dir}')
                metrics_dict = torch_fidelity.calculate_metrics(input1=os.path.join(os.getcwd(),cur_dir), input2=os.path.join(os.getcwd(),img_ref_dir), cuda=True, isc=True, fid=True, kid=True, ppl=False, verbose=True)
                print(metrics_dict)
                ics = metrics_dict['inception_score_mean']
                fid = metrics_dict['frechet_inception_distance']
                kid = metrics_dict['kernel_inception_distance_mean']
                #ppl = metrics_dict['perceptual_path_length_mean']


                num_ref = len(os.listdir(os.path.join(os.getcwd(),img_ref_dir)))
                num_gen = len(os.listdir(os.path.join(os.getcwd(),cur_dir)))

                wandb_table.add_data(_key, _value['level'], _value['attr_name'], _value['mask_name'], fid, ics, kid, -1,  num_ref, num_gen)
                print('*' * 38)
            _dict = dict()
            _dict[f'fid_r{interplate_n}'] = wandb_table
            wandb.log(_dict)




    elif dissec == 'rf':

        ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        img_shape = [batch_size, 3, 64, 64]
        percentile_suppress_under = 0.2
        from numeric_rf import NumericRF

        save_dir = flowdissect_util.increment_path(Path(log_root) / 'rf_vis')  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        print(f'rf dir: {save_dir}')

        if True:
            # with torch.no_grad():
            graph.eval()
            graph.to('cpu')
            rf = NumericRF(graph, input_shape=img_shape, percentile_suppress_under=percentile_suppress_under)

            # level_ids = [64, 100]  # [8, 16, 32, 64, 96, 100]#[i for i in range(len(graph.flow.output_shapes))]
            level_ids = [100, 72, 36, 8]
            ds_torch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                   drop_last=True)
            for img_id, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):

                html = ''
                html += BOOTSTRAP
                xs = batch_data['x']
                for level_id in level_ids:  # level_ids[4::8]:
                    html += f"<div class='alert alert-primary' role='alert'>" \
                            f'  Level_id: {level_id} </div><br>'
                    output_shape = graph.flow.output_shapes[level_id]
                    output_size, feat_dim = output_shape[-1], output_shape[1]
                    exp_per_size = 3  # 3x3
                    if True:
                        for i in range(0, output_size, output_size // exp_per_size):
                            for j in range(0, output_size, output_size // exp_per_size):
                                _name = f'rf_l{level_id}_i{i}_j{j}_p{percentile_suppress_under}.png'
                                save_name = os.path.join(save_dir, _name)
                                rf.heatmap(pos=(i, j), input_img=xs.clone(), level_id=level_id)
                                rf.plot(fname=save_name, add_text=True, use_out=None)
                                html += f"<img src='{_name}' class='img-fluid' /><br>"
                                html += f"<p>{_name}</p><br><br>"
                                with open(os.path.join(save_dir, f"img{img_id}.html"), "w") as outputfile:
                                    outputfile.write(html)

                    for c_id in range(0, feat_dim, feat_dim // exp_per_size):
                        _name = f'rf_l{level_id}_c{c_id}_p{percentile_suppress_under}.png'
                        save_name = os.path.join(save_dir, _name)
                        rf.heatmap(pos=None, input_img=xs.clone(), level_id=level_id, c_id=c_id)
                        rf.plot(fname=save_name, add_text=True, use_out=None)
                        html += f"<img src='{_name}' class='img-fluid' /><br>"
                        html += f"<p>{_name}</p><br><br>"
                        with open(os.path.join(save_dir, f"img{img_id}.html"), "w") as outputfile:
                            outputfile.write(html)

            graph.to('gpu')




    elif dissec == 'endz':
        # get Z
        if not generate_z:
            # try to load
            try:
                delta_Z = []
                for i in range(hparams.Glow.y_classes):
                    z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                    delta_Z.append(z)
            except FileNotFoundError:
                # need to generate
                generate_z = True
                print("Failed to load {} Z".format(hparams.Glow.y_classes))
                quit()
        if generate_z:
            delta_Z = graph.generate_attr_deltaz(dataset)
            for i, z in enumerate(delta_Z):
                np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
            print("Finish generating")

        plot_id = 0
        interplate_n = 5
        if True:
            image_ids = [20, 21, 22, 23, 24, 25, 26, 27, 28]
            attr_ids = [20, 8, 9, 15, 22, 31, 35, 33]
        else:
            image_ids = [20, 21, 22, 23]
            attr_ids = [20, 8]
        _scale = 0.7
        save_dir = flowdissect_util.increment_path(Path(hparams.Dir.log_root) / 'endz_vis')  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        plt.figure(figsize=(0.6 * interplate_n * _scale, (int(len(image_ids) * len(attr_ids) * _scale * 0.6))))
        for image_id in image_ids:
            html = ''
            html += BOOTSTRAP

            for attr_id in attr_ids:
                # interact with user
                base_index = select_index("base image", 0, len(dataset), _input=image_id, description=None)
                attr_index = select_index("attritube", 0, len(delta_Z), description=None, _input=attr_id)  #
                # attr_index = select_index("attritube", 0, len(delta_Z), description = dataset.attrs, _input=attr_id)
                attr_name = dataset.attrs[attr_index]
                z_delta = delta_Z[attr_index]
                graph.eval()
                z_base = graph.generate_z(dataset[base_index]["x"])
                # begin to generate new image
                images = []
                names = []
                images.append(run_z(graph, z_base))
                names.append("reconstruct_origin")

                for i in tqdm(range(0, interplate_n + 1)):
                    d = z_delta * float(i) / float(interplate_n)
                    images.append(run_z(graph, z_base - d))
                    names.append("{}, {}".format(attr_name, i))

                    if i > 0:
                        images.append(run_z(graph, z_base + d))
                        names.append("{}, {}".format(attr_name, - i))

                grid = make_grid(images[:], nrow=len(images)).permute(1, 2, 0)
                ax = plt.subplot(len(image_ids) * len(attr_ids), 1, plot_id + 1)
                ax.axis('off')
                # transforms.ToPILImage()(img)
                plt.imshow(grid)
                # ax.title.set_text(f'{attr_name}', fontsize=10)
                ax.set_title(f'i{attr_id}_{attr_name}', fontsize=2)
                # ax.set_xlabel(name)
                plot_id += 1
                save_name = f'img{image_id}_{attr_name}{attr_id}.png'
                (transforms.ToPILImage()(grid.permute(2, 0, 1))).save(os.path.join(save_dir, save_name))
                html += f"<img src='{save_name}' class='img-fluid' /><br>"
                html += f"<p>{save_name}</p><br><br>"
                if is_debug:
                    print(plot_id, len(image_ids) * len(attr_ids) * len(images))
                    print(names)

            with open(os.path.join(save_dir, f"img{image_id}_interp{interplate_n}.html"), "w") as outputfile:
                outputfile.write(html)

        f = f"infer.png"
        print(f'Saving {save_dir / f}...')
        # plt.title('infer.png')
        plt.show()
        plt.tight_layout()
        plt.savefig(save_dir / f, dpi=300)
