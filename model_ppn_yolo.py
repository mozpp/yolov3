import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

ONNX_EXPORT = False


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc

    def forward(self, p, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.ng.repeat((1, self.na * self.nx * self.ny, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view((1, -1, 2)) / ngu

            p = p.view(-1, 5 + self.nc)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy[0]  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh[0]  # width, height
            p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
            return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            # p = p.view(1, -1, 5 + self.nc)
            # xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            # wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            # p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            # p_cls = p[..., 5:5 + self.nc]
            # # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            # p_cls = torch.exp(p_cls).permute((2, 1, 0))
            # p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            # p_cls = p_cls.permute(2, 1, 0)
            # return torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p


class ResNetTwoHead(nn.Module):

    def __init__(self, block, layers, anchors, num_classes=1, arc='default'):
        self.inplanes = 64
        super(ResNetTwoHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.layer_det = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.LeakyReLU(negative_slope=0.1),
                                       nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.LeakyReLU(negative_slope=0.1),
                                       nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(negative_slope=0.1),
                                       nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0))
        self.layer_yolo = YOLOLayer(anchors, num_classes, arc)
        # # 固定for循环以上的参数
        # for p in self.parameters():
        #     p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        img_size = x.shape[-2:]
        output = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_l4 = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x_pose = self.fc(x_l4)

        x_det = self.layer_det(x_l4)
        x_det = self.layer_yolo(x_det, img_size)

        x_pose = x_pose.permute(0, 2, 3, 1)
        # x_det = x_det.permute(0, 2, 3, 1)
        output.append(x_det)

        if self.training:
            return x_pose, output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return x_pose, torch.cat(io, 1), p


def resnet18_pose_and_det(anchors, arc, net_state_dict=None):
    model = ResNetTwoHead(BasicBlock, [2, 2, 2, 2], anchors, 1, arc)
    # if net_state_dict is None:
    #     print('Loading resnet18 pretrain weight...')
    #     print('debug, 不加载预训练模型')
    #     model.load_state_dict(torch.load('./model/pretrain_weight/resnet18-5c106cde.pth'))
    # 替换fc输出头，替换为pose_head
    model.fc = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(512),
                             nn.LeakyReLU(negative_slope=0.1),
                             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                             nn.LeakyReLU(negative_slope=0.1),
                             nn.Conv2d(512, 1311, kernel_size=1, stride=1, padding=0))
    # model.head_d = nn.Sequential(nn.Conv2d(512 + 96, 512, kernel_size=3, stride=1, padding=1),
    #                              nn.BatchNorm2d(512),
    #                              nn.LeakyReLU(negative_slope=0.1),
    #                              nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    #                              nn.LeakyReLU(negative_slope=0.1),
    #                              nn.Conv2d(512, 16, kernel_size=1, stride=1, padding=0))
    if net_state_dict is not None:
        try:
            print('Loading resnet18-3d checkpoint weight...')
            model.load_state_dict(net_state_dict)
            print('successfully loaded resnet18-3d checkpoint weight.')
        except Exception as e:
            print(e, "变更为：只load部分参数")
            model_dict = model.state_dict()
            # 将model_pretrained的建与自定义模型的建进行比较，剔除不同的
            pretrained_dict = {k: v for k, v in net_state_dict.items() if k in model_dict}
            # 更新现有的model_dict
            model_dict.update(pretrained_dict)

            # 加载我们真正需要的state_dict
            model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    anchor = np.array([1, 2, 3, 4, 5, 6])
    anchor = anchor.reshape((-1, 2))
    model = resnet18_pose_and_det(anchor)
    print(model)
    input = torch.from_numpy(np.ones((320, 320, 3)).transpose((2, 0, 1)).astype(np.float32))
    if input.ndimension() == 3:
        input = input.unsqueeze(0)
    output_pose, output_det = model(input)
    pass
