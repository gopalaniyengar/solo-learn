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

from typing import Dict, List, Sequence

import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)

def domainwise_acc(dom_logits: torch.Tensor, dom_targets: torch.Tensor, ddict: Dict) -> List[Dict]:
    """Computes the accuracy of predictions separately for each domain.

    Args:
        dom_logits (torch.Tensor): Domain classifier raw predictions.
        dom_targets (torch.Tensor): Domain actual labels.
        ddict (Dict): Domain label to domain name mapping

    Returns:
        List[Dict]: Domainwise accuracy
    """
    
    _, pred = dom_logits.topk(1, 1, True, True)
    preds = pred.t().squeeze()
    labels = dom_targets.squeeze()
    matches = preds.eq(labels)

    acc = [0 for c in range(len(ddict))]
    for c in range(len(ddict)):
        acc[c] = (matches * labels.eq(c)).sum() / max(int(labels.eq(c).sum()), 1)

    out = {f'val_acc_{ddict[i]}':acc[i].item() for i in range(len(ddict))}
    return out

if __name__ == '__main__':

    a = torch.tensor([[1.2, 2.4, 5.4, 3.6], [3.2, 2.6, 6.4, 5.6], [3.2, 1.4, 3.4, 6.6], [0.2, 3.4, 1.4, 2.6]])
    b = torch.tensor([0,3,3,1])
    ddict = {0: 'art', 1: 'clipart', 2: 'product', 3: 'realworld'}
    out = domainwise_acc(a,b,ddict)
    art = 'art'
    print(out)
    print(out[f'val_acc_{art}'])