# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import os
import random
import string
import time
from pathlib import Path
from typing import List, Optional, Union, Dict

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import umap
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from solo.utils.misc import gather, omegaconf_select
from tqdm import tqdm


class AutoUMAP(Callback):
    def __init__(
        self,
        name: str,
        logdir: Union[str, Path] = Path("auto_umap"),
        frequency: int = 1,
        keep_previous: bool = False,
        color_palette: str = "hls",
        domain: bool = False
    ):
        """UMAP callback that automatically runs UMAP on the validation dataset and uploads the
        figure to wandb.

        Args:
            cfg (DictConfig): DictConfig containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to Path("auto_umap").
            frequency (int, optional): number of epochs between each UMAP. Defaults to 1.
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
            keep_previous (bool, optional): whether to keep previous plots or not.
                Defaults to False.
        """

        super().__init__()

        self.name = name
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.color_palette = color_palette
        self.keep_previous = keep_previous
        self.domain = domain

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.auto_umap = omegaconf_select(cfg, "auto_umap", default={})
        cfg.auto_umap.enabled = omegaconf_select(cfg, "auto_umap.enabled", default=False)
        cfg.auto_umap.dir = omegaconf_select(cfg, "auto_umap.dir", default="auto_umap")
        cfg.auto_umap.frequency = omegaconf_select(cfg, "auto_umap.frequency", default=1)

        return cfg

    @staticmethod
    def random_string(letter_count=4, digit_count=4):
        tmp_random = random.Random(time.time())
        rand_str = "".join((tmp_random.choice(string.ascii_lowercase) for x in range(letter_count)))
        rand_str += "".join((tmp_random.choice(string.digits) for x in range(digit_count)))
        rand_str = list(rand_str)
        tmp_random.shuffle(rand_str)
        return "".join(rand_str)

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if self.logdir.exists():
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = []
            version = "offline-" + self.random_string()
            while version in existing_versions:
                version = "offline-" + self.random_string()
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = self.logdir / version
            self.umap_placeholder = f"{self.name}-{version}" + "-ep={}.pdf"
        else:
            self.path = self.logdir
            self.umap_placeholder = f"{self.name}" + "-ep={}.pdf"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Performs initial setup on training start.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)

    def plot_old(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        device = module.device
        bb_feats = []
        style_proj_feats = []
        cont_proj_feats = []

        Y = []

        # set module to eval model and collect all feature representations
        module.eval()
        with torch.no_grad():
            for x, y in trainer.val_dataloaders[0]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                all_feats_dict = module(x)
                feat = all_feats_dict["feats"]
                style_proj = all_feats_dict["s"]
                cont_proj = all_feats_dict["z"]

                y = gather(y)
                bb_feats.append(feat.cpu())
                style_proj_feats.append(style_proj.cpu())
                cont_proj_feats.append(cont_proj.cpu())
                Y.append(y.cpu())

        module.train()
        if self.domain:
            umap_keys = ['backbone_features','content_projections']
            umap_data = [bb_feats, cont_proj_feats]
        else:
            umap_keys = ['backbone_features','style_projections','content_projections']
            umap_data = [bb_feats, style_proj_feats, cont_proj_feats]
        palettes  = ['hls', 'hls', 'hls', 'hls']
        sns.set_palette("dark")

        for idxx, data in enumerate(umap_data):
            if trainer.is_global_zero and len(data):
                data = torch.cat(data, dim=0).numpy()
                dY = torch.cat(Y, dim=0)
                num_classes = len(torch.unique(dY))
                dY = dY.numpy()

                data = umap.UMAP(n_components=2).fit_transform(data)

                # passing to dataframe
                df = pd.DataFrame()
                df["feat_1"] = data[:, 0]
                df["feat_2"] = data[:, 1]
                df["Y"] = dY
                plt.figure(figsize=(16, 9))
                ax = sns.scatterplot(
                    x="feat_1",
                    y="feat_2",
                    hue="Y",
                    palette=sns.color_palette(palettes[idxx], num_classes),
                    data=df,
                    legend="full",
                    alpha=0.3,
                )
                ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
                ax.tick_params(left=False, right=False, bottom=False, top=False)

                # manually improve quality of imagenet umaps
                if num_classes > 100:
                    anchor = (0.5, 1.8)
                else:
                    anchor = (0.5, 1.35)

                plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
                plt.tight_layout()

                if isinstance(trainer.logger, pl.loggers.WandbLogger):
                    # wandb.log(
                    #     {"validation_umap": wandb.Image(ax)},
                    #     commit=False,
                    # )
                    wandb.log(
                        {umap_keys[idxx]: wandb.Image(ax)},
                        commit=False,
                    )

                # save plot locally as well
                epoch = trainer.current_epoch  # type: ignore
                # plt.savefig(self.path / self.umap_placeholder.format(epoch))
                plt.close()
    
    def gather_for_umap(self, feat_list: Dict, key: str):
        
        feat = feat_list[key]
        feat = gather(feat)
        return feat.cpu()

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        device = module.device
        bb_feats = []
        style_proj_feats = []
        cont_proj_feats = []
        styleproj_flag = False

        module.eval()
        with torch.no_grad():
            
            if not self.domain:        
                Y = []
                for x, y in trainer.val_dataloaders[0]:
                   
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    all_feats_dict = module(x)
                    
                    feat = self.gather_for_umap(all_feats_dict, "feats")
                    cont_proj = self.gather_for_umap(all_feats_dict, "z")
                    if "s" in all_feats_dict.keys():
                        styleproj_flag = True
                        style_proj = self.gather_for_umap(all_feats_dict, "s")
                        style_proj_feats.append(style_proj)
                    y = gather(y).cpu()

                    bb_feats.append(feat)
                    cont_proj_feats.append(cont_proj)
                    Y.append(y)

            else: 
                Y = []
                YD = []
                for x, yc, yd in trainer.val_dataloaders[0]:
                    
                    x = x.to(device, non_blocking=True)
                    yc = yc.to(device, non_blocking=True)
                    yd = yd.to(device, non_blocking=True)
                    all_feats_dict = module(x)

                    feat = self.gather_for_umap(all_feats_dict, "feats")
                    cont_proj = self.gather_for_umap(all_feats_dict, "z")
                    if "s" in all_feats_dict.keys():
                        styleproj_flag = True
                        style_proj = self.gather_for_umap(all_feats_dict, "s")
                        style_proj_feats.append(style_proj)                    
                    yc = gather(yc).cpu()
                    yd = gather(yd).cpu()

                    bb_feats.append(feat)
                    cont_proj_feats.append(cont_proj)
                    Y.append(yc)
                    YD.append(yd)

        module.train()
        sns.set_palette("dark")
        if not styleproj_flag:
            umap_keys = ['backbone_features','content_projections']
            umap_data = [bb_feats, cont_proj_feats]
        else:
            umap_keys = ['backbone_features','style_projections','content_projections']
            umap_data = [bb_feats, style_proj_feats, cont_proj_feats]

        for idxx, data in enumerate(umap_data):
            
            if trainer.is_global_zero and len(data):
                
                    data = torch.cat(data, dim=0).numpy()
                    data = umap.UMAP(n_components=2).fit_transform(data)

                    dY = torch.cat(Y, dim=0)
                    num_classes = len(torch.unique(dY))
                    dY = dY.numpy()
                    
                    df = pd.DataFrame()
                    df["feat_1"] = data[:, 0]
                    df["feat_2"] = data[:, 1]
                    df["Y"] = dY
                    
                    plt.figure(figsize=(16, 9))
                    ax = sns.scatterplot(
                        x="feat_1",
                        y="feat_2",
                        hue="Y",
                        palette=sns.color_palette('hls', num_classes),
                        data=df,
                        legend="full",
                        alpha=0.3,
                    )
                    ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
                    ax.tick_params(left=False, right=False, bottom=False, top=False)
                    
                    if num_classes > 100:
                        anchor = (0.5, 1.8)
                    else:
                        anchor = (0.5, 1.35)
                    
                    plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
                    plt.tight_layout()
                    if isinstance(trainer.logger, pl.loggers.WandbLogger):
                        wandb.log(
                            {f'{umap_keys[idxx]}_class': wandb.Image(ax)},
                            commit=False,
                        )
                    plt.close()

                    if self.domain:  

                        dYD = torch.cat(YD, dim=0)
                        num_domains = len(torch.unique(dYD))
                        dYD = dYD.numpy()

                        df = pd.DataFrame()
                        df["feat_1"] = data[:, 0]
                        df["feat_2"] = data[:, 1]
                        df["Y"] = dYD
                        
                        plt.figure(figsize=(16, 9))
                        ax = sns.scatterplot(
                            x="feat_1",
                            y="feat_2",
                            hue="Y",
                            palette=sns.color_palette('hls', num_domains),
                            data=df,
                            legend="full",
                            alpha=0.3,
                        )
                        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
                        ax.tick_params(left=False, right=False, bottom=False, top=False)
                        
                        if num_domains > 100:
                            anchor = (0.5, 1.8)
                        else:
                            anchor = (0.5, 1.35)
                        
                        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_domains / 10))
                        plt.tight_layout()
                        if isinstance(trainer.logger, pl.loggers.WandbLogger):
                            wandb.log(
                                {f'{umap_keys[idxx]}_domain': wandb.Image(ax)},
                                commit=False,
                            )
                        plt.close()                   

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.sanity_checking:
            self.plot(trainer, module)


class OfflineUMAP:
    def __init__(self, color_palette: str = "hls"):
        """Offline UMAP helper.

        Args:
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
        """

        self.color_palette = color_palette

    def subplot(self, data, Y, num_classes, path, suffix):
      
        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = data[:, 0]
        df["feat_2"] = data[:, 1]
        df["Y"] = Y
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            legend="full",
            alpha=0.3,
        )
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig(path+suffix)
        plt.close()

    def plot(
        self,
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        plot_path: str,
    ):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the model.
        **Note: the model should produce features for the forward() function.

        Args:
            device (str): gpu/cpu device.
            model (nn.Module): current model.
            dataloader (torch.utils.data.Dataloader): current dataloader containing data.
            plot_path (str): path to save the figure.
        """

        data = []
        inst_data = []
        Y = []

        # set module to eval model and collect all feature representations
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats, inst_feats = model(x)
                data.append(feats.cpu())
                inst_data.append(inst_feats.cpu())
                Y.append(y.cpu())
        model.train()

        data = torch.cat(data, dim=0).numpy()
        inst_data = torch.cat(inst_data, dim=0).numpy()
        Y = torch.cat(Y, dim=0)
        num_classes = len(torch.unique(Y))
        Y = Y.numpy()

        print("Creating UMAP")
        data = umap.UMAP(n_components=2).fit_transform(data)
        inst_data = umap.UMAP(n_components=2).fit_transform(inst_data) 

        self.subplot(data, Y, num_classes, plot_path, suffix='_resfeats.pdf')
        self.subplot(inst_data, Y, num_classes, plot_path, suffix='_instnorm.pdf')
    
    def plot_projections(
        self,
        device: str,
        models: List[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        plot_path: str,
    ):

        backbone = models[0]
        content_proj, style_proj = models[1], models[2]
        data = []
        inst_data = []
        st_proj = []
        cnt_proj = []
        Y = []

        # set module to eval model and collect all feature representations
        backbone.eval()
        content_proj.eval()
        style_proj.eval()

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats, inst_feats = backbone(x)
                style_projections = style_proj(inst_feats)
                content_projections = content_proj(feats)

                data.append(feats.cpu())
                inst_data.append(inst_feats.cpu())
                st_proj.append(style_projections.cpu())
                cnt_proj.append(content_projections.cpu())
                Y.append(y.cpu())

        backbone.train()
        content_proj.train()
        style_proj.train()

        data = torch.cat(data, dim=0).numpy()
        inst_data = torch.cat(inst_data, dim=0).numpy()
        st_proj = torch.cat(st_proj, dim=0).numpy()
        cnt_proj = torch.cat(cnt_proj, dim=0).numpy()
        Y = torch.cat(Y, dim=0)
        num_classes = len(torch.unique(Y))
        Y = Y.numpy()

        print("Creating UMAP...")
        data = umap.UMAP(n_components=2).fit_transform(data)
        inst_data = umap.UMAP(n_components=2).fit_transform(inst_data) 
        st_proj = umap.UMAP(n_components=2).fit_transform(st_proj) 
        cnt_proj = umap.UMAP(n_components=2).fit_transform(cnt_proj) 

        self.subplot(data, Y, num_classes, plot_path, suffix='_bb_feats.pdf')
        self.subplot(st_proj, Y, num_classes, plot_path, suffix='_style_projection.pdf')
        self.subplot(cnt_proj, Y, num_classes, plot_path, suffix='_content_projection.pdf')