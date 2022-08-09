import os
import glob
import argparse
import torch
import numpy as np
import tqdm
from torch import optim
from tensorboardX import SummaryWriter
from importlib import import_module
from models.model import Yolo
from dataset.dataset import MyDataset
from tools.utils import merge_cfg
from sklearn.model_selection import train_test_split
from dataprocess.transform import Transform
from torch.utils.data import DataLoader
from models.yolo_loss import YoloLoss
from models.yolo_loss2 import YoloLoss as YoloLoss2
from tools.box_transforms import convert_to_corners
from tools.visualization import save_eval_img


def train(epochs, lrs, num_gpu, model, optimizer, writer, train_loader, val_loader, criterion, device, workdir,
          eval_path):
    global_step = 0
    val_global_step = 0
    best_loss = 1e3
    for epoch in range(epochs):
        if epoch < 75:
            lr = lrs[0]
            lr += (epoch + 1) * lr * 9 / 75
        elif epoch <= 105:
            lr = lrs[1]
        else:
            lr = lrs[2]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        eval_outs = []
        eval_imgs = []

        model.train()
        progressbar = tqdm.tqdm(train_loader)
        epoch_losses = []
        for index, item in enumerate(progressbar):
            imgs, targets = item
            if num_gpu > 1:
                imgs = imgs.float().to(device=0)
                targets = targets.float().to(device=0)
            else:
                imgs = imgs.float().to(device)
                targets = targets.float().to(device)
            optimizer.zero_grad()

            outs = model(imgs)
            loss = criterion(outs, targets)

            loss.backward()
            optimizer.step()

            desc = f'Epochs: {epoch + 1}/{epochs} iters: {index + 1}/{len(train_loader)} loss: {loss.item()}'

            progressbar.set_description(desc)

            writer.add_scalars('Loss', {'train': loss}, global_step=global_step)
            writer.add_scalar('learning_rate', lr, global_step)

            global_step += 1

            epoch_losses.append(loss.item())

            if index in [0, 1]:
                eval_outs.append(outs.detach())
                eval_imgs.append(imgs)

        epoch_loss = np.mean(epoch_losses)
        if best_loss > epoch_loss:
            torch.save(model.state_dict(), os.path.join(workdir, 'yolo_best.pth'))

        torch.save(model.state_dict(), os.path.join(workdir, 'yolo_last.pth'))

        model.eval()
        progressbar = tqdm.tqdm(val_loader)

        for index, item in enumerate(progressbar):
            imgs, targets = item
            if num_gpu > 1:
                imgs = imgs.float().cuda(device=0)
                targets = targets.float().cuda(device=0)
            else:
                imgs = imgs.float().to(device)
                targets = targets.float().to(device)

            with torch.no_grad():
                outs = model(imgs)

            loss = criterion(outs, targets)
            desc = f'Epochs: {epoch + 1} / {epochs} iters: {index + 1} / {len(val_loader)} loss: {loss.item()}'

            progressbar.set_description(desc)

            writer.add_scalars('Val_Loss', {'val': loss}, global_step=val_global_step)
            val_global_step += 1

        if (epoch + 1) % 5 == 0:
            outs = torch.concat(eval_outs, dim=0)
            imgs = torch.concat(eval_imgs, dim=0)
            imgs = imgs.cpu().numpy()

            batch_boxes, _, _ = model.decoder(outs, device=device)
            n = batch_boxes.shape[0]
            for i in range(n):
                img = imgs[i]
                boxes = batch_boxes[i]
                num = boxes.shape[0]
                if num == 0:
                    continue
                h, w = img.shape[1:3]
                boxes = boxes * np.array([w, h, w, h])
                boxes = convert_to_corners(boxes)

                save_eval_img(img, boxes, os.path.join(eval_path, f'{epoch + 1}_{i}.jpg'))


def main(args):
    seed = 1212

    print('load settings')
    # setting params
    data_cfg = args.data
    hp_cfg = args.hp
    model_cfg = args.model
    cfg = merge_cfg([data_cfg, hp_cfg, model_cfg])

    # create work dir
    print('init work dir')
    workdir = args.workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    logs_dir = os.listdir(workdir)
    max_logs = 0
    for log_dir in logs_dir:
        if 'logs' not in log_dir:
            continue
        if len(log_dir) > 4:
            num = int(log_dir[4:])
            if max_logs < num:
                max_logs = num
        else:
            max_logs = 1
    logs_path = os.path.join(workdir, 'logs') if max_logs == 0 else os.path.join(workdir, 'logs{}'.format(max_logs + 1))
    weights_path = os.path.join(logs_path, 'weights')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    eval_path = os.path.join(workdir, 'eval')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    writer = SummaryWriter(logs_path)

    batch_size = cfg['batch_size']
    img_size = cfg['img_size']
    grid = cfg['grid']
    num_gpu = cfg['num_gpu']
    fcn = cfg['fcn']
    b = cfg['b']
    nc = cfg['nc']
    sync_bn = cfg['sync_bn']
    epochs = cfg['epochs']
    lrs = cfg['lr']
    weight_decay = cfg['weight_decay']
    momentum = cfg['momentum']
    opt = cfg['opt']
    gama_coord = cfg['gama_coord']
    gama_noobj = cfg['gama_noobj']

    batch_size = batch_size * num_gpu

    # load data
    print('load data')
    data_path = args.data_path
    if data_path is None:
        dataprocess = cfg['dataprocess']
        if dataprocess is not None:
            dataset_name = os.path.basename(data_cfg).split('.')[0]
            p_module = import_module('datapreprocess.' + dataprocess)
            process = p_module.DataPreprocess(cfg['path'], cfg['names'], f'data/{dataset_name}_data.txt')
            process.process()
    else:
        if not os.path.exists(data_path):
            raise ValueError(f'{data_path} does not existed !')

    lines = open(data_path).readlines()

    # split train val data
    print('split data for train and val')
    train_lines, val_lines = train_test_split(lines, test_size=.2, shuffle=True, random_state=seed)

    # load train data
    print('load train data and val data')
    train_dataset = MyDataset(train_lines + val_lines, img_size=img_size, grid=grid, b=b, num_classes=nc, transform=Transform())
    val_dataset = MyDataset(val_lines, img_size=img_size, grid=grid, b=b, num_classes=nc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_gpu = min(num_gpu, torch.cuda.device_count())
    if num_gpu > 1:
        device_ids = list(range(num_gpu))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('load model')
    backbone = os.path.basename(model_cfg).replace('.yaml', '')
    model = Yolo({'backbone': backbone, 'fcn': fcn, 'b': b, 'nc': nc})

    pre_weights = args.pre_weights
    if pre_weights and os.path.exists(pre_weights):
        print('load pre weights')
        model.load_state_dict(torch.load(pre_weights), strict=False)

    # support multi gpu
    if num_gpu > 1:
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        model.to(device=0)
    else:
        model.to(device)

    if num_gpu > 1 and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lrs[0], momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lrs[0], weight_decay=weight_decay)

    # load loss
    print('load loss')
    criterion = YoloLoss2(grid, b, gama_coord, gama_noobj, device=device)
    # criterion = YoloLoss(grid, b, gama_coord, gama_noobj, device=device)

    train(epochs, lrs, num_gpu, model, optimizer, writer, train_loader, val_loader, criterion, device, weights_path,
          eval_path)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOV1 Train')
    parser.add_argument('--data_path', default='data/voc_data.txt', type=str, help='dataset settings')
    parser.add_argument('--workdir', default='run', type=str, help='dataset settings')
    parser.add_argument('--pre_weights', default='run/logs/weights/yolo_best.pth', type=str, help='pre weights file')
    parser.add_argument('--data', default='configs/data/voc.yaml', type=str, help='dataset settings')
    parser.add_argument('--hp', default='configs/hp/hp.yaml', type=str, help='train hyper params')
    parser.add_argument('--model', default='configs/model/vgg16.yaml', type=str, help='model params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
