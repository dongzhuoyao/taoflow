import torch
import matplotlib.pyplot as plt
import numpy as np


def postprocess(x, n_bits=8):
    x = torch.clamp(x, 0, 1)
    # x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


def get_bounds(gradient, percentile_suppress_under):
    # coords = gradient.nonzero(as_tuple=True)  # get non-zero coords
    percentile_20 = gradient.min().item() + (gradient.max().item() - gradient.min().item()) * percentile_suppress_under
    coords = (gradient > percentile_20).nonzero(as_tuple=True)
    names = ['h', 'w']
    ret = {}

    for ind in [0, 1]:
        mini, maxi = coords[ind].min().item(), coords[ind].max().item()
        ret[names[ind]] = {'bounds': (mini, maxi), 'range': maxi - mini}
    return ret


def plot_input_output(image, gradient, output_tensor, out_pos, coords, ishape, c_id=None, fname=None, add_text=False,
                      use_out=None):
    fig = plt.figure(figsize=(13, 8))
    ax = [plt.subplot2grid(shape=(4, 1), loc=(0, 0), rowspan=3),
          plt.subplot2grid(shape=(4, 1), loc=(3, 0))]

    # Plot RF
    oshape = output_tensor.squeeze().shape
    oshape = list(oshape)
    oshape = oshape[1:]  # force to use first image

    if c_id is None:
        ax[0].set_title("Input(Image) shape %s" % (ishape[1:]))
    else:
        ax[0].set_title("Input(Image) shape {}, c_id {}".format(ishape[1:], c_id))

    if image is not None:
        if image.ndim == 4:
            image = image[0]

        if not isinstance(image, np.ndarray):
            print(image.shape)
            image = postprocess(image).permute(1, 2, 0).numpy()

        ax[0].imshow(image)
    else:
        ax[0].imshow(gradient, cmap='copper', interpolation='nearest')

    # Draw RF bounds
    h0, w0, h, w = coords
    ax[0].add_patch(plt.Rectangle((w0 - 0.5, h0 - 0.5), w + 1, h + 1, fill=False, edgecolor='cyan'))
    if add_text:
        ax[0].text(w0 + w + 2, h0, 'Receptive Field', size=12, color='cyan', weight='bold')

    if out_pos is not None:
        if use_out is not None:
            out = use_out
        else:
            out = np.random.rand(*oshape)
        # Plot channel mean of output
        ax[1].set_title("Output(Z) shape %s" % (list(oshape)))
        ax[1].imshow(out.mean(0), cmap='binary', interpolation='nearest')
        ax[1].add_patch(plt.Rectangle((out_pos[1] - 0.5, out_pos[0] - 0.5), 1, 1, color='red'))
        if add_text:
            ax[1].text(out_pos[1] + 1, out_pos[0], f'{list(out_pos)}', size=14, color='red', weight='bold')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, format='png')
        plt.close()
        print('saving {}'.format(fname))


class NumericRF:

    def __init__(self, model, input_shape, percentile_suppress_under):

        if not isinstance(input_shape, list):
            input_shape = list(input_shape)

        self.model = model.eval()

        if len(input_shape) == 3:
            input_shape = [1] + input_shape

        assert len(input_shape) == 4
        self.input_shape = input_shape
        self.percentile_suppress_under = percentile_suppress_under

    def _remove_bias(self):
        for conv in self.model:
            conv.bias.data.fill_(0)
            conv.bias.requires_grad = False

    def get_rf_coords(self):
        h0, w0 = [self._info[k]['bounds'][0] for k in ['h', 'w']]
        h, w = [self._info[k]['range'] for k in ['h', 'w']]
        return h0, w0, h, w

    def heatmap(self, pos=None, input_img=None, level_id=None, c_id=None):
        self.pos = pos
        self.c_id = c_id

        # Step 1: build computational graph
        if input_img is None:
            self.inp = torch.zeros(self.input_shape, requires_grad=True)
        else:
            # self.inp = input_img#.unsqueeze(0)
            # self.inp.to('cpu')
            self.inp = torch.clone(input_img.detach())
            self.inp.requires_grad = True
        # https://discuss.pytorch.org/t/grad-attribute-of-a-non-leaf-tensor-being-accessed/82313/2
        z, nll, y_logits, z_list = self.model(self.inp, None,
                                              dissec=dict(output_feat=True, output_feat_detach=False))

        if level_id is None:
            # Step 2: zero out gradient tensor
            grad = torch.zeros_like(self.out)
            # Step 3: this could be any non-zero value
            if self.pos is None:
                grad[:,self.c_id, :, :] = 1.0  # torch.Size([1, 96, 4, 4])
            else:
                grad[..., self.pos[0], self.pos[1]] = 1.0  # torch.Size([1, 96, 4, 4])
            # grad[:,0,:,:] = 1.0
            # Step 4: propagate tensor backward
            self.out.backward(gradient=grad, retain_graph=True)
        else:
            self.out = z_list[level_id]['z']
            grad = torch.zeros_like(self.out)
            if self.pos is None:
                grad[:,self.c_id, :, :] = 1.0  # torch.Size([1, 96, 4, 4])
            else:
                grad[..., self.pos[0], self.pos[1]] = 1.0  # torch.Size([1, 96, 4, 4])
            self.out.backward(gradient=grad, retain_graph=True)

        # Step 5: average signal over batch and channel + we only care about magnitute of signal
        self.grad_data = self.inp.grad.mean([0, 1]).abs().data  # mean on batch, channel dims

        self._info = get_bounds(self.grad_data, self.percentile_suppress_under)

        return self._info

    def info(self):
        return self._info

    def plot(self, fname=None, add_text=False, use_out=None):
        plot_input_output(image=self.inp.clone(),
                          gradient=self.grad_data,
                          output_tensor=self.out,
                          out_pos=self.pos,
                          c_id=self.c_id,
                          coords=self.get_rf_coords(),
                          ishape=self.input_shape,
                          fname=fname,
                          add_text=add_text,
                          use_out=use_out,
                          )
