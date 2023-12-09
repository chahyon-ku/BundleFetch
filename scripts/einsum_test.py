import torch


if __name__ == '__main__':
    eye = torch.eye(3)
    

    Jdi = torch.einsum('ab, phwi', eye, )