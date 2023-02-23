import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pytorch_lightning as pl
from utils_slom import exists, default, SupConLoss, Siren, plot_islands_agreement
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from spikingjelly.clock_driven import neuron, layer, surrogate, functional

TOKEN_ATTEND_SELF_VALUE = -5e-4


class SpikingConvTokenizer(pl.LightningModule):
    def __init__(self, FLAGS, in_channels=3, embedding_dim=64, surrogate_fct=surrogate.ATan(), use_max_pool=True):
        super(SpikingConvTokenizer, self).__init__()
        self.FLAGS = FLAGS
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),)

        if self.FLAGS.spiking_neuron == "IFNeuron":
            self.sn1 = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                    surrogate_function=surrogate_fct,
                                                    detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "LIFNeuron":
            self.sn1 = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                     tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                     detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "PLIFNeuron":
            self.sn1 = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                      init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                      detach_reset=self.FLAGS.detach_reset)
        else:
            print("No such neurons!")
            exit(0)

        self.block2 = nn.Sequential(
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),)

        if self.FLAGS.spiking_neuron == "IFNeuron":
            self.sn2 = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                    surrogate_function=surrogate_fct,
                                                    detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "LIFNeuron":
            self.sn2 = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                     tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                     detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "PLIFNeuron":
            self.sn2 = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                      init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                      detach_reset=self.FLAGS.detach_reset)
        else:
            print("No such neurons!")
            exit(0)

        self.block3 = nn.Sequential(
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim),)

        if self.FLAGS.spiking_neuron == "IFNeuron":
            self.sn3 = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                    surrogate_function=surrogate_fct,
                                                    detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "LIFNeuron":
            self.sn3 = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                     tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                     detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "PLIFNeuron":
            self.sn3 = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                                      init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                                      detach_reset=self.FLAGS.detach_reset)
        else:
            print("No such neurons!")
            exit(0)

        if use_max_pool:
            self.pool = nn.MaxPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1))
        else:
            self.pool = nn.AvgPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1))

    def forward(self, x):
        out1 = self.block1(x)
        sp1 = self.sn1(out1)
        out2 = self.block2(sp1)
        sp2 = self.sn2(out2)
        out3 = self.block3(sp2)
        sp3 = self.sn3(out3)
        out = self.pool(sp3)
        return out


class SpikingColumnNet(pl.LightningModule):
    def __init__(self, FLAGS, dim, groups, mult=4, surrogate_fct=surrogate.ATan()):
        super().__init__()
        self.FLAGS = FLAGS
        total_dim = dim * groups
        num_patches = (self.FLAGS.conv_image_size // self.FLAGS.patch_size) ** 2

        if self.FLAGS.spiking_neuron == "IFNeuron":
            self.spiking_levels = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                     surrogate_function=surrogate_fct,
                                     detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "LIFNeuron":
            self.spiking_levels = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                      tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                      detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "PLIFNeuron":
            self.spiking_levels = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                       init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                       detach_reset=self.FLAGS.detach_reset)
        else:
            print("No such neurons!")
            exit(0)

        self.net1 = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups=groups),
            nn.LayerNorm(num_patches),)

        if self.FLAGS.spiking_neuron == "IFNeuron":
            self.col_sn1 = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                     surrogate_function=surrogate_fct,
                                     detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "LIFNeuron":
            self.col_sn1 = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                      tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                      detach_reset=self.FLAGS.detach_reset)
        elif self.FLAGS.spiking_neuron == "PLIFNeuron":
            self.col_sn1 = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                       init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                       detach_reset=self.FLAGS.detach_reset)
        else:
            print("No such neurons!")
            exit(0)

        self.net2 = nn.Sequential(
            nn.Conv1d(total_dim * mult, total_dim, 1, groups=groups),
            nn.LayerNorm(num_patches),)

        if self.FLAGS.column_potential:
            self.col_sn2 = neuron.IFNode(v_threshold=float('inf'), v_reset=0.0, surrogate_function=surrogate_fct, \
                                         detach_reset=self.FLAGS.detach_reset)
        else:
            if self.FLAGS.spiking_neuron == "IFNeuron":
                self.col_sn2 = neuron.IFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                         surrogate_function=surrogate_fct,
                                         detach_reset=self.FLAGS.detach_reset)
            elif self.FLAGS.spiking_neuron == "LIFNeuron":
                self.col_sn2 = neuron.LIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                          tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                          detach_reset=self.FLAGS.detach_reset)
            elif self.FLAGS.spiking_neuron == "PLIFNeuron":
                self.col_sn2 = neuron.PLIFNode(v_threshold=self.FLAGS.v_th, v_reset=self.FLAGS.v_rst, \
                                           init_tau=self.FLAGS.tau, surrogate_function=surrogate_fct,
                                           detach_reset=self.FLAGS.detach_reset)
            else:
                print("No such neurons!")
                exit(0)

        self.rearrange = Rearrange('b (l d) n -> b n l d', l=groups)

    def forward(self, levels):
        if self.FLAGS.column_potential:
            levels_out1 = self.net1(self.spiking_levels(levels))
            col_sp1 = self.col_sn1(levels_out1)
            levels_out2 = self.net2(col_sp1)
            self.col_sn2(levels_out2)
            col_sp2 = self.col_sn2.v
            levels_out = self.rearrange(col_sp2)
        else:
            levels_out1 = self.net1(self.spiking_levels(levels))
            col_sp1 = self.col_sn1(levels_out1)
            levels_out2 = self.net2(col_sp1)
            col_sp2 = self.col_sn2(levels_out2)
            levels_out = self.rearrange(col_sp2)
        return levels_out

class ConsensusAttention(pl.LightningModule):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out


class SpikingGlom(pl.LightningModule):
    def __init__(self, FLAGS, *, consensus_self=False, local_consensus_radius=0, surrogate_function=surrogate.ATan()):
        super(SpikingGlom, self).__init__()
        self.FLAGS = FLAGS

        self.num_patches_side = (self.FLAGS.conv_image_size // self.FLAGS.patch_size)
        
        self.num_patches = self.num_patches_side ** 2
        self.iters = default(self.FLAGS.iters, self.FLAGS.levels * 2)
        self.batch_acc = 0

        self.wl = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wBU = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wTD = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wA = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)

        self.image_to_tokens = nn.Sequential(
            SpikingConvTokenizer(self.FLAGS, in_channels=self.FLAGS.n_channels, embedding_dim=self.FLAGS.patch_dim // (self.FLAGS.patch_size ** 2), surrogate_fct=surrogate_function),
            Rearrange('b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1=self.FLAGS.patch_size, p2=self.FLAGS.patch_size),
        )

        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(FLAGS.patch_dim),
            nn.Dropout(p=self.FLAGS.dropout),
            Rearrange('b n d -> b (n d)'),
            nn.LayerNorm(self.num_patches * FLAGS.patch_dim),
            nn.Dropout(p=self.FLAGS.dropout),
            nn.Linear(self.num_patches * FLAGS.patch_dim, self.num_patches * FLAGS.patch_dim),
            nn.LayerNorm(self.num_patches * FLAGS.patch_dim),
            nn.GELU(),
            nn.LayerNorm(self.num_patches * FLAGS.patch_dim),
            nn.Dropout(p=self.FLAGS.dropout),
            nn.Linear(self.num_patches * FLAGS.patch_dim, self.FLAGS.contr_dim)
        )

        self.classification_head_from_contr = nn.Sequential(
            nn.Linear(self.FLAGS.contr_dim, self.FLAGS.contr_dim),
            nn.GELU(),
            nn.Linear(self.FLAGS.contr_dim, self.FLAGS.n_classes)
        )

        self.init_levels = nn.Parameter(torch.randn(self.FLAGS.levels, FLAGS.patch_dim))
        self.bottom_up = SpikingColumnNet(self.FLAGS, dim=FLAGS.patch_dim, groups=self.FLAGS.levels, surrogate_fct=surrogate_function)
        self.top_down = SpikingColumnNet(self.FLAGS, dim=FLAGS.patch_dim, groups=self.FLAGS.levels - 1, surrogate_fct=surrogate_function)
        self.attention = ConsensusAttention(self.num_patches_side, attend_self=consensus_self, local_consensus_radius=local_consensus_radius)

        

    def forward(self, img, levels=None):
        b, device = img.shape[0], img.device

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b=b, n=n)

        hiddens = [levels]

        num_contributions = torch.empty(self.FLAGS.levels, device=device).fill_(4)
        num_contributions[-1] = 3 

        for _ in range(self.iters):

            levels_with_input = torch.cat((bottom_level, levels.float()), dim=-2)

            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])

            top_down_out = self.top_down(torch.flip(levels_with_input[..., 2:, :], [2]))
            top_down_out = F.pad(torch.flip(top_down_out, [2]), (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels.float())

            levels_sum = torch.stack((
                levels * self.wl, \
                bottom_up_out * self.wBU, \
                top_down_out * self.wTD, \
                consensus * self.wA
            )).sum(dim=0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            self.log('Weights/wl', self.wl)
            self.log('Weights/wBU', self.wBU)
            self.log('Weights/wTD', self.wTD)
            self.log('Weights/wA', self.wA)

            levels = levels_mean
            hiddens.append(levels)

        all_levels = torch.stack(hiddens)

        top_level = all_levels[self.FLAGS.denoise_iter, :, :, -1]

        top_level = self.contrastive_head(top_level)
        top_level = F.normalize(top_level, dim=1)          

        return top_level, all_levels[-1, 0, :, :, :]

    def training_step(self, train_batch, batch_idx):
        functional.reset_net(self)
        image = train_batch[0]
        label = train_batch[1]
        self.training_batch_idx = batch_idx

        if not self.FLAGS.supervise:
            image = torch.cat([image[0], image[1]], dim=0)

        if self.FLAGS.supervise:
            with torch.no_grad():
                top_level, _ = self.forward(image)
                
        else:
            top_level, toplot = self.forward(image)
            if self.FLAGS.plot_islands:
                plot_islands_agreement(toplot, image[0, :, :, :], label[0])

        if self.FLAGS.supervise:
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.detach(), label, topk=(1,))[0]
            self.log('Training_accuracy', self.batch_acc, prog_bar=True, sync_dist=True)
            self.batch_acc = 0

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.FLAGS.num_gpus, self.FLAGS.batch_size // self.FLAGS.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Training_loss', loss, sync_dist=True)
        self.log('Training_LR', self.optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        functional.reset_net(self)
        image = val_batch[0]
        label = val_batch[1]
        self.val_batch_idx = batch_idx

        if not self.FLAGS.supervise:
            image = torch.cat([image[0], image[1]], dim=0)

        if self.FLAGS.supervise:
            with torch.no_grad():
                top_level, _ = self.forward(image)
                
        else:
            top_level, toplot = self.forward(image)
            if self.FLAGS.plot_islands:
                plot_islands_agreement(toplot, image[0, :, :, :], label[0])

        if self.FLAGS.supervise:
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.detach(), label, topk=(1,))[0]
            self.log('Validation_accuracy', self.batch_acc, prog_bar=True, sync_dist=True)
            self.batch_acc = 0

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.FLAGS.num_gpus, self.FLAGS.batch_size // self.FLAGS.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Validation_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        functional.reset_net(self)
        image = test_batch[0]
        label = test_batch[1]
        self.test_batch_idx = batch_idx

        if not self.FLAGS.supervise:
            image = torch.cat([image[0], image[1]], dim=0)

        if self.FLAGS.supervise:
            with torch.no_grad():
                top_level, _ = self.forward(image)
                
        else:
            top_level, toplot = self.forward(image)
            if self.FLAGS.plot_islands:
                plot_islands_agreement(toplot, image[0, :, :, :], label[0])

        if self.FLAGS.supervise:
            output = self.classification_head_from_contr(top_level)
            loss = F.cross_entropy(output, label)
            self.batch_acc = self.accuracy(output.detach(), label, topk=(1,))[0]
            self.log('Test_accuracy', self.batch_acc, prog_bar=True, sync_dist=True)

        else:
            f1, f2 = torch.split(top_level, [self.FLAGS.batch_size // self.FLAGS.num_gpus, self.FLAGS.batch_size // self.FLAGS.num_gpus], dim=0)
            output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            criterion = SupConLoss(temperature=self.FLAGS.temperature)
            loss = criterion(output, label)

        self.log('Test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.FLAGS.learning_rate,
            weight_decay=self.FLAGS.weight_decay,
        )

        return {'optimizer': self.optimizer}

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
