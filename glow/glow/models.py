import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils
import pickle
import math



def f(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        modules.Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]), nn.ReLU(inplace=False),
        modules.Conv2dZeros(hidden_channels, out_channels))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False):
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling, \
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowStep1D(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False):
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling, \
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.output_instance_name = []
        self.K = K
        self.L = L
        H, W, C = image_shape
        assert C == 1 or C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                  "C == 1 or C == 3")
        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(modules.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            self.output_instance_name.append('SqueezeLayer')
            print(self.output_shapes[-1], 'SqueezeLayer', len(self.output_shapes))

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed))
                self.output_shapes.append(
                    [-1, C, H, W])
                self.output_instance_name.append('FlowStep')
                print(self.output_shapes[-1], 'FlowStep', len(self.output_shapes))
            # 3. Split2d
            if i < L - 1:
                self.layers.append(modules.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                self.output_instance_name.append('Split2d')
                C = C // 2
                print(self.output_shapes[-1], 'Split2d', len(self.output_shapes))

        if False:
            self.layer_name2id = dict()
            pre_size = self.output_shapes[0][-1]
            layer_id = 0
            self.layer_name2id[f'32_{layer_id}'] = dict(id=0, name=self.output_instance_name[0],
                                                        shape=self.output_shapes[0])
            for i, (shape, instance_name) in enumerate(zip(self.output_shapes[1:], self.output_instance_name[1:])):
                real_i = i + 1
                if shape[-1] == pre_size:
                    layer_id += 1
                else:
                    layer_id = 0  # reset to 0
                    pre_size = shape[-1]

                if shape[-1] == 32:
                    self.layer_name2id[f'32_{layer_id}'] = dict(id=real_i, name=instance_name, shape=shape)
                elif shape[-1] == 16:
                    self.layer_name2id[f'16_{layer_id}'] = dict(id=real_i, name=instance_name, shape=shape)
                elif shape[-1] == 8:
                    self.layer_name2id[f'8_{layer_id}'] = dict(id=real_i, name=instance_name, shape=shape)
                else:
                    raise
                print(shape, instance_name)

            print(self.layer_name2id)
        else:
            self.layer_name2id = dict()

    def forward(self, input, logdet=0., reverse=False, eps_std=None, dissec=None):
        if not reverse:
            return self.encode(input, logdet, dissec=dissec)
        else:
            return self.decode(input, eps_std, dissec=dissec)

    def encode(self, z, logdet=0.0, dissec=None):
        if 'interrupt_z' not in dissec or dissec['interrupt_z'] is None:
            z_list = []
            for layer, shape in zip(self.layers, self.output_shapes):
                if isinstance(layer, modules.Split2d):
                    instance_name = 'Split2d'
                elif isinstance(layer, modules.SqueezeLayer):
                    instance_name = 'SqueezeLayer'
                elif isinstance(layer, FlowStep):
                    instance_name = 'FlowStep'
                else:
                    raise
                z, logdet = layer(z, logdet, reverse=False)
                z_list.append(dict(z=z.detach().cpu() if dissec['output_feat_detach'] else z,
                                   logdet=logdet.detach().cpu(),
                                   name=instance_name))

            if dissec['output_feat']:
                return z, logdet, z_list
            else:
                return z, logdet
        else:
            begin_id = dissec['interrupt_z']  # self.layer_name2id[dissec['layer_name']]['id']  # from x to z
            # print('decoding(z -> x) from new layer_id {}'.format(begin_id))
            z = dissec['interrupt_z_value']  # override by interrupted midz

            z_list = []
            for layer, shape in zip(self.layers[begin_id+1: ], self.output_shapes):
                if isinstance(layer, modules.Split2d):
                    instance_name = 'Split2d'
                elif isinstance(layer, modules.SqueezeLayer):
                    instance_name = 'SqueezeLayer'
                elif isinstance(layer, FlowStep):
                    instance_name = 'FlowStep'
                else:
                    raise
                z, logdet = layer(z, logdet, reverse=False)
                z_list.append(dict(z=z.detach().cpu() if dissec['output_feat_detach'] else z,
                                   logdet = logdet.detach().cpu(),
                                   name=instance_name))

            if 'output_feat' in dissec and dissec['output_feat']:
                return z, logdet, z_list
            else:
                return z, logdet


    def decode(self, z, eps_std=None, dissec=None):
        if dissec['interrupt_z'] is None:
            for layer in reversed(self.layers):
                if isinstance(layer, modules.Split2d):
                    z, logdet = layer(z, logdet=0, reverse=True, eps_std=eps_std)
                else:
                    z, logdet = layer(z, logdet=0, reverse=True)
            return z
        else:
            begin_id = dissec['interrupt_z']  # self.layer_name2id[dissec['layer_name']]['id']  # from x to z
            # print('decoding(z -> x) from new layer_id {}'.format(begin_id))
            z = dissec['interrupt_z_value']  # override by interrupted midz
            for layer in reversed(self.layers[:begin_id + 1]):
                if isinstance(layer, modules.Split2d):
                    z, logdet = layer(z, logdet=0, reverse=True, eps_std=eps_std)
                else:
                    z, logdet = layer(z, logdet=0, reverse=True)
            return z


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective, chw

class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, hparams):
        super().__init__()
        self.flow = FlowNet(image_shape=hparams.Glow.image_shape,
                            hidden_channels=hparams.Glow.hidden_channels,
                            K=hparams.Glow.K,
                            L=hparams.Glow.L,
                            actnorm_scale=hparams.Glow.actnorm_scale,
                            flow_permutation=hparams.Glow.flow_permutation,
                            flow_coupling=hparams.Glow.flow_coupling,
                            LU_decomposed=hparams.Glow.LU_decomposed)

        self.layer_name2id = self.flow.layer_name2id
        self.hparams = hparams
        self.n_bits = hparams.Data.n_bits
        self.y_classes = hparams.Glow.y_classes
        # for prior
        if hparams.Glow.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        if hparams.Glow.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = modules.LinearZeros(
                hparams.Glow.y_classes, 2 * C)
            self.project_class = modules.LinearZeros(
                C, hparams.Glow.y_classes)
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.register_parameter(
            "prior_h",
            nn.Parameter(torch.zeros([hparams.Train.batch_size // num_device,
                                      self.flow.output_shapes[-1][1] * 2,
                                      self.flow.output_shapes[-1][2],
                                      self.flow.output_shapes[-1][3]])))
        self.fuck_prior = True

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        try:
            assert torch.sum(h) == 0.0#why? error?
        except:
            import ipdb;ipdb.set_trace()
        if self.hparams.Glow.learn_top:
            h = self.learn_top(h)
        if self.hparams.Glow.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return thops.split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None,
                eps_std=None, reverse=False, dissec=None):
        if dissec is None:  # set default
            dissec = dict(output_feat=False, output_feat_detach=False, layer_name=None, interrupt_z=None)
        if not reverse:
            return self.normal_flow(x, y_onehot, dissec=dissec)
        else:
            return self.reverse_flow(z, y_onehot, eps_std, dissec=dissec)

    def normal_flow(self, x, y_onehot, dissec):
        if False:
            pixels = thops.pixels(x)#this line is fucking buggy
            z = x + torch.normal(mean=torch.zeros_like(x),
                                 std=torch.ones_like(x) * (1. / (2**self.n_bits)))  # random noise
            logdet = torch.zeros_like(x[:, 0, 0, 0])
            logdet += float(-np.log((2**self.n_bits)) * pixels)
        else:
            z, logdet, pixels  = uniform_binning_correction(x, n_bits=self.n_bits)

        # encode
        if 'output_feat' in dissec and dissec['output_feat']:
            z, objective, z_list = self.flow(z, logdet=logdet, reverse=False, dissec=dissec)
        else:
            z, objective = self.flow(z, logdet=logdet, reverse=False, dissec=dissec)
        # prior
        if self.fuck_prior:
            mean = torch.zeros((x.shape[0], self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3] ),requires_grad=False).cuda()
            logs = torch.zeros((x.shape[0], self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2], self.flow.output_shapes[-1][3] ),requires_grad=False).cuda()
        else:
            mean, logs = self.prior(y_onehot)

        #import ipdb;ipdb.set_trace()

        final_z_logp = modules.GaussianDiag.logp(mean, logs, z)
        objective_copy = torch.clone(objective)
        objective += final_z_logp

        if self.hparams.Glow.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.) * pixels)

        dict_result = dict(y_logits=y_logits,
                           pixels=pixels,
                           pz=torch.exp(final_z_logp).detach().cpu(),
                           logpz=final_z_logp.detach().cpu()/pixels,
                           logpx=objective.detach().cpu()/pixels,
                           logdet = objective_copy.detach().cpu()/pixels)

        if 'output_feat' in dissec and dissec['output_feat']:
            return z, nll, dict_result, z_list
        else:
            return z, nll, dict_result

    def reverse_flow(self, z, y_onehot, eps_std, dissec):
        with torch.no_grad():
            if self.fuck_prior:
                mean = torch.zeros((self.prior_h.size(0), self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                    self.flow.output_shapes[-1][3]), requires_grad=False).cuda()
                logs = torch.zeros((self.prior_h.size(0), self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                    self.flow.output_shapes[-1][3]), requires_grad=False).cuda()
            else:
                mean, logs = self.prior(y_onehot)
            if z is None:
                z = modules.GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std, reverse=True, dissec=dissec)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    def generate_z(self, img):
        with torch.no_grad():
            self.eval()
            B = self.hparams.Train.batch_size
            x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
            z, _, _ = self(x)
            self.train()
            return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    def generate_attr_deltaz_midz(self, dataset, feat_dict_path):
        assert "y_onehot" in dataset[0]
        self.eval()
        level_zs = dict()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            levels = len(self.flow.output_shapes)
            for level_id in range(levels):
                level_zs[level_id] = dict(
                    attrs_pos_z=[[0, 0] for _ in range(self.y_classes)],
                    attrs_neg_z=[[0, 0] for _ in range(self.y_classes)],
                )

            ds_torch = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False,
                                                   drop_last=True)
            for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):

                xs = batch_data['x'].cuda()
                batch_size = xs.shape[0]
                _, _, _, z_list = self.forward(xs,
                                                         dissec=dict(output_feat=True, output_feat_detach=True))
                for level_id in range(levels):
                    zzz = z_list[level_id]['z'].detach().cpu().numpy()
                    for k in range(xs.shape[0]):
                        z = zzz[k]
                        y = batch_data['y_onehot'][k].numpy()
                        for ai in range(self.y_classes):
                            if y[ai] > 0:
                                level_zs[level_id]['attrs_pos_z'][ai][0] += z
                                level_zs[level_id]['attrs_pos_z'][ai][1] += 1
                            else:
                                level_zs[level_id]['attrs_neg_z'][ai][0] += z
                                level_zs[level_id]['attrs_neg_z'][ai][1] += 1
                if i % 2000 == 0:
                    # break
                    attr_dict = dict()
                    for level_id in range(levels):
                        for ai in range(self.y_classes):
                            z_pos = level_zs[level_id]['attrs_pos_z'][ai][0] / max(
                                float(level_zs[level_id]['attrs_pos_z'][ai][1]), 1e-5)
                            z_neg = level_zs[level_id]['attrs_neg_z'][ai][0] / max(
                                float(level_zs[level_id]['attrs_neg_z'][ai][1]), 1e-5)
                            z_delta = z_pos - z_neg
                            attr_dict[f'level{level_id}_cls{ai}_pos'] = z_pos
                            attr_dict[f'level{level_id}_cls{ai}_neg'] = z_neg
                            attr_dict[f'level{level_id}_cls{ai}_delta'] = z_delta
                    with open(feat_dict_path + f'_iter{i}', 'wb') as f:
                        print('start dumping z_dict. {}'.format(feat_dict_path + f'_iter{i}'))
                        pickle.dump(attr_dict, f)
                        print('finish dumping z_dict.')

        # break
        attr_dict = dict()
        for level_id in range(levels):
            for ai in range(self.y_classes):
                z_pos = level_zs[level_id]['attrs_pos_z'][ai][0] / max(
                    float(level_zs[level_id]['attrs_pos_z'][ai][1]), 1e-5)
                z_neg = level_zs[level_id]['attrs_neg_z'][ai][0] / max(
                    float(level_zs[level_id]['attrs_neg_z'][ai][1]), 1e-5)
                z_delta = z_pos - z_neg
                attr_dict[f'level{level_id}_cls{ai}_pos'] = z_pos
                attr_dict[f'level{level_id}_cls{ai}_neg'] = z_neg
                attr_dict[f'level{level_id}_cls{ai}_delta'] = z_delta
        with open(feat_dict_path, 'wb') as f:
            print('start dumping z_dict. {}'.format(feat_dict_path))
            pickle.dump(attr_dict, f)
            print('finish dumping z_dict.')

        self.train()
        return level_zs



    def generate_midz_allimg(self, dataset, feat_dict_path, levels = [8, 36, 72, 100], attr_ids = [20, 8, 9, 15, 22, 31, 35, 33]):
        assert "y_onehot" in dataset[0]
        self.eval()
        level_zs = dict()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            levels = levels
            for level_id in levels:
                level_zs[level_id] = dict(
                    attrs_pos_z=[list() for _ in range(self.y_classes)],
                    attrs_neg_z=[list() for _ in range(self.y_classes)],
                )

            ds_torch = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False,
                                                   drop_last=True)
            for i, batch_data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):

                xs = batch_data['x'].cuda()
                batch_size = xs.shape[0]
                _, _, _, z_list = self.forward(xs,
                                                         dissec=dict(output_feat=True, output_feat_detach=True))
                for level_id in levels:
                    zzz = z_list[level_id]['z'].detach().cpu().numpy()
                    for k in range(xs.shape[0]):
                        z = zzz[k]
                        y = batch_data['y_onehot'][k].numpy()
                        for ai in attr_ids:
                            if y[ai] > 0:
                                level_zs[level_id]['attrs_pos_z'][ai].append(z)
                            else:
                                level_zs[level_id]['attrs_neg_z'][ai].append(z)
                if i % 2000 == 0:
                    # break
                    attr_dict = dict()
                    for level_id in levels:
                        for ai in attr_ids:
                            attr_dict[f'level{level_id}_cls{ai}_pos'] = level_zs[level_id]['attrs_pos_z'][ai]
                            attr_dict[f'level{level_id}_cls{ai}_neg'] = level_zs[level_id]['attrs_neg_z'][ai]
                    with open(feat_dict_path + f'_iter{i}', 'wb') as f:
                        print('start dumping z_dict. {}'.format(feat_dict_path + f'_iter{i}'))
                        pickle.dump(attr_dict, f)
                        print('finish dumping z_dict.')

        # break
        attr_dict = dict()
        for level_id in levels:
            for ai in attr_ids:
                attr_dict[f'level{level_id}_cls{ai}_pos'] = level_zs[level_id]['attrs_pos_z'][ai]
                attr_dict[f'level{level_id}_cls{ai}_neg'] = level_zs[level_id]['attrs_neg_z'][ai]
        with open(feat_dict_path, 'wb') as f:
            print('start dumping z_dict. {}'.format(feat_dict_path + f'_iter{i}'))
            pickle.dump(attr_dict, f)
            print('finish dumping z_dict.')

        self.train()
        return level_zs

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())
