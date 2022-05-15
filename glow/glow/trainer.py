import os
import torch
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, plot_prob
from .config import JsonConfig
from .models import Glow
from . import thops
import wandb



class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                 devices, data_device,
                 dataset, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)

        # set members
        # append date info
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")

        #v2, fix random noise pixelnumber error bug
        self.log_dir = os.path.join(hparams.Dir.log_root.replace('results/','results_v2/'), "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints
        # model relative
        self.graph = graph
        self.optim = optim
        self.weight_y = hparams.Train.weight_y
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm
        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device
        self.image_size = hparams.Data.resize
        print('**** image size= {} to calculate BPD'.format(self.image_size))
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      num_workers=hparams.Data.workers,
                                      shuffle=True,
                                      drop_last=True)
        self.n_epoches = (hparams.Train.num_batches + len(self.data_loader) - 1)
        self.n_epoches = self.n_epoches // len(self.data_loader)

        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_condition
        assert self.y_criterion in ["multi-classes", "single-class"]

        # notice the order, wandb init before tensorboard writer init!
        wandb.tensorboard.patch(
            root_logdir=self.log_dir)  # https://github.com/wandb/client/issues/1782#issuecomment-779191950
        wandb.init(name=self.log_dir, project="chaiyujin", entity='vincenthu')
        wandb.config.update(dict(hparams))
        wandb.run.summary["devices"] = len(self.devices)
        img_to_train = self.n_epoches * len(self.data_loader) * self.batch_size
        print('**** init img_to_train = {} ******'.format(img_to_train))
        wandb.run.summary["img_to_train"] =img_to_train

        if False:
            for name, param in graph.named_parameters():
                if 'actnorm' in name:
                    print(name)
                    param.requires_grad = True
                else:
                    #print('fuck', name)
                    param.requires_grad = False


        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        self.inference_gap = hparams.Train.inference_gap


    def train(self):
        # set to training state
        self.graph.train()
        self.global_step = self.loaded_step
        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch {}/{}".format(epoch, self.n_epoches))
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):
                # if i_batch>10: break
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                # get batch data
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                x = batch["x"]
                y = None
                y_onehot = None
                if self.y_condition:
                    if self.y_criterion == "multi-classes":
                        assert "y_onehot" in batch, "multi-classes ask for `y_onehot` (torch.FloatTensor onehot)"
                        y_onehot = batch["y_onehot"]
                    elif self.y_criterion == "single-class":
                        assert "y" in batch, "single-class ask for `y` (torch.LongTensor indexes)"
                        y = batch["y"]
                        y_onehot = thops.onehot(y, num_classes=self.y_classes)

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)
                # parallel
                if len(self.devices) > 1 and not hasattr(self.graph, "module"):
                    print("[Parallel] move to {}".format(self.devices))
                    self.graph = torch.nn.parallel.DataParallel(self.graph, self.devices, self.devices[0])
                # forward phase
                dissec_logdet = True
                if not dissec_logdet:
                    z, nll, y_logits = self.graph(x=x, y_onehot=y_onehot)
                else:
                    z, nll, y_logits, z_list = self.graph(x=x, y_onehot=y_onehot,  dissec=dict(output_feat=True, output_feat_detach=True))

                # loss
                loss_generative = Glow.loss_generative(nll)
                loss_classes = 0


                if self.y_condition:
                    loss_classes = (Glow.loss_multi_classes(y_logits, y_onehot)
                                    if self.y_criterion == "multi-classes" else
                                    Glow.loss_class(y_logits, y))
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative.item(), self.global_step)
                    wandb.log(dict(loss_generative=loss_generative.item(),
                                   lr_group0=self.optim.param_groups[0]['lr'],
                                   batch=self.global_step,
                                   imgs=self.global_step * self.batch_size))
                    if dissec_logdet:
                        #print(torch.mean(y_logits['logpx']))
                        #print(torch.mean(y_logits['logpz']))
                        #print(torch.mean(y_logits['logdet']))
                        layer_num =len(self.graph.flow.output_shapes)
                        logdet_dict = dict()
                        for _de_levelid in range(0, layer_num, 8):
                            logdet = torch.mean(z_list[_de_levelid]['logdet'] / y_logits['pixels']).item()
                            logdet_dict[f'logdet_l{_de_levelid}'] = logdet
                        wandb.log(logdet_dict)

                    if self.y_condition:
                        self.writer.add_scalar("loss/loss_classes", loss_classes, self.global_step)
                        wandb.log(dict(loss_classes=loss_classes))
                loss = loss_generative + loss_classes * self.weight_y

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                # operate grad


                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                        wandb.log(dict(grad_norm=grad_norm))
                # step
                self.optim.step()

                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                if self.global_step % self.plot_gaps == 0:
                    img = self.graph(z=z, y_onehot=y_onehot, reverse=True)
                    # img = torch.clamp(img, min=0, max=1.0)
                    if self.y_condition:
                        if self.y_criterion == "multi-classes":
                            y_pred = torch.sigmoid(y_logits)
                        elif self.y_criterion == "single-class":
                            y_pred = thops.onehot(torch.argmax(F.softmax(y_logits, dim=1), dim=1, keepdim=True),
                                                  self.y_classes)
                        y_true = y_onehot
                    for bi in range(min([len(img), 4])):
                        self.writer.add_image("0_reverse/{}".format(bi), torch.cat((img[bi], batch["x"][bi]), dim=1),
                                              self.global_step)
                        if self.y_condition:
                            self.writer.add_image("1_prob/{}".format(bi),
                                                  plot_prob([y_pred[bi], y_true[bi]], ["pred", "true"]),
                                                  self.global_step)

                # inference
                if hasattr(self, "inference_gap"):
                    if self.global_step % self.inference_gap == 0:
                        img = self.graph(z=None, y_onehot=y_onehot, eps_std=0.5, reverse=True)
                        # img = torch.clamp(img, min=0, max=1.0)
                        for bi in range(min([len(img), 4])):
                            self.writer.add_image("2_sample/{}".format(bi), img[bi], self.global_step)

                # global step
                self.global_step += 1

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
