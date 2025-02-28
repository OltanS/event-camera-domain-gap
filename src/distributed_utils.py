import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

def ddp_setup(local_rank):
    '''
    Initialize distributed training environment
    args:
        local_rank (int): The rank of the current process
    '''
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")

def ddp_cleanup():
    '''Destroy distributed training environment'''
    destroy_process_group()

def ddp_model(model, local_rank):
    '''
    Wrap model for distributed training
    args:
        model (torch.nn.Module): The model to wrap
        local_rank (int): The rank of the current process
    '''
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])