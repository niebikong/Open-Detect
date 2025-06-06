from .resnet import ResNet18Enc, ResNet18Dec  # 32*32的VAE模型

# from .t1 import ResNet18Enc, ResNet18Dec  # 40*40的VAE模型

__factory = {
    'resnet18': [ResNet18Enc, ResNet18Dec],
}


def names():
    return sorted(__factory.keys())

def net(name, *args, **kwargs):
    """
    Create a net.
    Parameters
    ----------
    name : str
        the name of network arch
    """
    if name not in __factory:
        raise KeyError("Unknown Network Arch:", name)
    encoder = __factory[name][0](*args, **kwargs)
    decoder = __factory[name][1](*args, **kwargs)
    return encoder, decoder

