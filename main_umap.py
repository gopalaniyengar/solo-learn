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

import json
import os
from pathlib import Path

from solo.args.setup import parse_args_umap
from solo.methods import METHODS
from solo.utils.auto_umap import OfflineUMAP
from solo.data.classification_dataloader import prepare_data
from solo.backbones.resnet import inst_style_feats

def main():
    args = parse_args_umap()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    # prepare data
    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # build the model
    checkpoint = METHODS[method_args["method"]].load_from_checkpoint(ckpt_path, strict=False, **method_args)
    model = (checkpoint.backbone)
    style_proj = model = (checkpoint.style_projector)
    cont_proj = model = (checkpoint.projector)

    # move model to the gpu
    device = "cuda:0"

    model.cuda()
    style_proj.cuda()
    cont_proj.cuda()

    model = model.to(device)
    style_proj = style_proj.to(device)
    cont_proj = cont_proj.to(device)

    umap = OfflineUMAP()
    # umap.plot(device, model, train_loader, "/content/drive/MyDrive/BTP/UMAPs/train")
    # umap.plot(device, model, val_loader, "/content/drive/MyDrive/BTP/UMAPs/val")
    umap.plot_projections(device, [model, cont_proj, style_proj], train_loader, "/content/drive/MyDrive/BTP/Projection_UMAPs/train")
    umap.plot_projections(device, [model, cont_proj, style_proj], val_loader, "/content/drive/MyDrive/BTP/Projection_UMAPs/val")

if __name__ == "__main__":
    main()
