import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as tfs
from torchvision import models
from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
import os
import sys


voc_root = "./data/voc2012/VOCtrainval_11-May-2012"
mode_train = False
mode_predict = True


def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


# data, label = read_images()
# plt.figure()
# for i, pic_data in enumerate(data):
#     pic_data = np.array(Image.open(pic_data))
#     plt.subplot(121)
#     plt.imshow(pic_data)
#     pic_label = np.array(Image.open(label[i]))
#     plt.subplot(122)
#     plt.imshow(pic_label)
#     plt.pause(0.01)
# plt.close()


def rand_crop(data, label, height, width):
    i, j, h, w = tfs.RandomCrop.get_params(data, output_size=(height, width))
    data_crop = tfs.functional.crop(data, i, j, h, w)
    label_crop = tfs.functional.crop(label, i, j, h, w)
    return data_crop, label_crop


# data, label = read_images()
# plt.figure()
# for i, pic_data in enumerate(data):
#     pic_data_origin = np.array(Image.open(pic_data))
#     plt.subplot(221)
#     plt.imshow(pic_data_origin)
#     pic_label_origin = np.array(Image.open(label[i]))
#     plt.subplot(222)
#     plt.imshow(pic_label_origin)
#     pic_data_crop, pic_label_crop = rand_crop(Image.open(pic_data), Image.open(label[i]), 100, 100)
#     plt.subplot(223)
#     plt.imshow(pic_data_crop)
#     plt.subplot(224)
#     plt.imshow(pic_label_crop)
#     plt.pause(0.5)
# plt.close()


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0],[128, 192, 0], [0, 64, 128]]

rgb2class = np.zeros(256**3)
for i, rgb in enumerate(colormap):
    rgb2class[(rgb[0] * 256 + rgb[1]) * 256 + rgb[2]] = i


def label2class(label):
    label = np.array(label, dtype='int32')
    idx = (label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]
    return np.array(rgb2class[idx],dtype='int64')


# class_array = label2class(Image.open('./data/voc2012/VOCtrainval_11-May-2012/SegmentationClass'
#                                      '/2007_000033.png').convert('RGB'))
# print(class_array[140: 160, 240: 260])

def pre_transforms(data, label, crop_size):
    data, label = rand_crop(data, label, *crop_size)
    data_transforms = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = data_transforms(data)
    label = torch.from_numpy(label2class(label))
    return data, label


class SegmentationDataset(Dataset):
    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self.filter(data_list)
        self.label_list = self.filter(label_list)

    def filter(self, images):
        return [image for image in images if(Image.open(image).size[1] >= self.crop_size[0] and
                                             Image.open(image).size[0] >= self.crop_size[1])]

    def __getitem__(self, item):
        data = Image.open(self.data_list[item])
        label = Image.open(self.label_list[item]).convert('RGB')
        return self.transforms(data, label, self.crop_size)

    def __len__(self):
        return len(self.data_list)


crop_size = (320, 480)
train = SegmentationDataset(True, crop_size, pre_transforms)
valid = SegmentationDataset(False, crop_size, pre_transforms)
batch_size = 32
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_data = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_data = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[: kernel_size, : kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# data = np.array(Image.open('./data/voc2012/VOCtrainval_11-May-2012/JPEGImages/2007_005210.jpg'))
# plt.figure()
# plt.subplot(211)
# plt.imshow(data)
# data = torch.from_numpy(data.astype('float32')).permute(2, 0, 1).unsqueeze(0)
# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
# conv_trans.weight.data = bilinear_kernel(3, 3, 4)
# data_trans = conv_trans(data).data.squeeze().permute(1, 2, 0).numpy()
# plt.subplot(212)
# plt.imshow(data_trans.astype('uint8'))
# plt.show()
# plt.pause(1000)


pretrained_net = models.resnet34(pretrained=True)
num_classes = len(classes)
# print(pretrained_net)


class NaiveFCN(nn.Module):
    def __init__(self, num_classes):
        super(NaiveFCN, self).__init__()
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[: -4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x
        x = self.stage2(x)
        s2 = x
        x = self.stage3(x)
        s3 = x
        s3 = self.upsample_2x(self.scores1(s3))
        s2 = self.upsample_4x(self.scores2(s2) + s3)
        s1 = self.upsample_8x(self.scores3(s1) + s2)
        return s1


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).\
        reshape(n_class, n_class)


def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    hist += _fast_hist(label_trues.flatten(), label_preds.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    main_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum() # frequency weight IU
    return acc, acc_cls, main_iu, fwavacc


if mode_train:
    criterion = nn.NLLLoss2d()
    net = NaiveFCN(num_classes)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, last_epoch=-1)
    epoch = 80
    for e in range(epoch):
        scheduler.step()
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0
        prev_time = datetime.now()
        net = net.train()
        net = net.cuda()
        for item in train_data:
            data = item[0].cuda()
            label = item[1].cuda()
            out = net(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc

        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0
        with torch.no_grad():
            for data in valid_data:
                data = item[0].cuda()
                label = item[1].cuda()
                out = net(data)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.cpu().item()

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                    eval_acc += acc
                    eval_acc_cls + acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IoU: {:.5f}, '
                     'Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IoU: {:.5f} '.format(
            e + 1, train_loss / len(train_data), train_acc / len(train), train_mean_iu / len(train),
            eval_loss / len(valid_data), eval_acc / len(valid), eval_mean_iu / len(valid)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str + ' lr: {}'.format(scheduler.get_lr()))
    torch.save(net.state_dict(), './parameters/NaiveFCN.pt')


if mode_predict:
    net = NaiveFCN(num_classes)
    net.load_state_dict(torch.load('./parameters/NaiveFCN.pt'))
    net.eval()
    net.cuda()

    cm = np.array(colormap).astype('uint8')

    def predict(data, label):
        data = data.unsqueeze(0).cuda()
        out = net(data)
        pred = out.max(dim=1)[1].squeeze().data.cpu().numpy()
        return cm[pred], cm[label.numpy()]

    _, figs = plt.subplots(10, 3, figsize=(12, 8))
    valid_bias = 0
    for i in range(10):
        predict_data, predict_label = valid[i + valid_bias]
        pred, label = predict(predict_data, predict_label)
        figs[i, 0].imshow(np.array(Image.open(valid.data_list[i + valid_bias])))
        figs[i, 0].axes.get_xaxis().set_visible(False)
        figs[i, 0].axes.get_yaxis().set_visible(False)
        figs[i, 1].imshow(label)
        figs[i, 1].axes.get_xaxis().set_visible(False)
        figs[i, 1].axes.get_yaxis().set_visible(False)
        figs[i, 2].imshow(pred)
        figs[i, 2].axes.get_xaxis().set_visible(False)
        figs[i, 2].axes.get_yaxis().set_visible(False)
