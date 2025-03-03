# 导入库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
    os.environ['MASTER_PORT'] = '12355'      # 主节点端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置每个进程对应的 GPU


def train(rank,world_size):
    try:
        swanlab.init(project="case_learn",experiment_name="GPUS")
        setup(rank, world_size)
        batch_size=256
        learning_rate=0.01
        epochs=100
        # 数据处理
        tran_method=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(1.0,))])
        Minst_Data=torchvision.datasets.MNIST(root="datasets",train=True,transform=tran_method,download=False)# root只存放根目录
        # train_sampler = torch.utils.data.distributed.DistributedSampler(Minst_Data)
        train_sampler = DistributedSampler(
        Minst_Data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # 控制是否打乱数据
    )
        data_train_loader=DataLoader(Minst_Data,batch_size=batch_size,shuffle=False,sampler=train_sampler)
        # 网络结构搭建
        model=torchvision.models.resnet101(num_classes=10)
        model.conv1=torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model=model.cuda()
        model = DDP(model, device_ids=[rank])
        # 损失函数，优化器设置
        loss_function=nn.CrossEntropyLoss()
        opt=optim.Adam(model.parameters(),lr=learning_rate)
        if rank == 0:
            epoch_pbar = tqdm(total=epochs, desc="Total Epochs", position=0)
            batch_pbar = tqdm(total=len(data_train_loader), desc="Batches", position=1, leave=True)

        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss=0
            if rank == 0:
                batch_pbar.reset(total=len(data_train_loader))
                batch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            for batch_number,data in enumerate(data_train_loader):
                images,label=data
                images,label=images.cuda(),label.cuda()
                opt.zero_grad()
                label_pred=model(images)
                loss=loss_function(label_pred,label)
                loss.backward()
                opt.step()
                epoch_loss+=loss.item()
                            # 每10个batch记录日志
                if batch_number % 10 == 0 and rank == 0:
                    swanlab.log({"batch_loss": loss.item()})
                    batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                # 更新batch进度条（仅主进程）
                if rank == 0:
                    batch_pbar.update(1)

            # 更新epoch进度条和日志
            avg_epoch_loss = epoch_loss / len(data_train_loader)
            if rank == 0:
                swanlab.log({"epoch_loss": avg_epoch_loss})
                epoch_pbar.set_postfix({"epoch_loss": f"{avg_epoch_loss:.4f}"})
                epoch_pbar.update(1)
                batch_pbar.close()
        # 关闭进度条
        if rank == 0:
            epoch_pbar.close()
    finally:
        dist.destroy_process_group()
if __name__=="__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    if dist.is_initialized():
        dist.destroy_process_group()