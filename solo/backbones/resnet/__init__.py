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

from .resnet import resnet18_bn as default_resnet18
from .resnet import resnet50_bn as default_resnet50

from .resnet import resnet18_in as custom_resnet18
from .resnet import resnet50_in as custom_resnet50


def resnet18_bn(method, *args, **kwargs):
    return default_resnet18(*args, **kwargs)

def resnet50_bn(method, *args, **kwargs):
    return default_resnet50(*args, **kwargs)

def resnet18_in(method, *args, **kwargs):
    return custom_resnet18(*args, **kwargs)

def resnet50_in(method, *args, **kwargs):
    return custom_resnet50(*args, **kwargs)

__all__ = ["resnet18_in", "resnet50_in", "resnet18_bn", "resnet50_bn"]
