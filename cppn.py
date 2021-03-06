import os
import sys
import argparse
import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F
from imageio import imwrite
import imutil
import time


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--z', default=8, type=int, help='latent space width')
    parser.add_argument('--n', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--scale', default=10, type=int, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--net', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--exp', default='0', type=str, help='output fn')

    parser.add_argument('--walk', default=False, type=bool, help='Generate a smooth video interpolation through the latent space')
    parser.add_argument('--sample', default=False, type=bool, help='Sample images independently from the latent space')

    parser.add_argument('--render_video', default=False, type=bool, help='If walk mode is enabled, output an mp4 video')
    parser.add_argument('--interpolation', default='linear', type=str, help='One of: linear, sigmoid')
    parser.add_argument('--num_frames', default=50, type=int, help='Number of video frames to generate')

    args = parser.parse_args()
    return args


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        dim = self.x_dim * self.y_dim * self.batch_size
        self.linear_z = nn.Linear(self.z, self.net)
        self.linear_x = nn.Linear(1, self.net, bias=False)
        self.linear_y = nn.Linear(1, self.net, bias=False)
        self.linear_r = nn.Linear(1, self.net, bias=False)
        self.linear_h = nn.Linear(self.net, self.net)
        self.linear_out = nn.Linear(self.net, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        #print ('G in: ', x.shape)
        x, y, z, r = inputs
        n_points = self.x_dim * self.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)#.cuda()
        z_scaled = z.view(self.batch_size, 1, self.z) * ones * self.scale
        z_pt = self.linear_z(z_scaled.view(self.batch_size*n_points, self.z))
        x_pt = self.linear_x(x.view(self.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(self.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(self.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = torch.tanh(U)
        H = F.elu(self.linear_h(H))
        H = F.softplus(self.linear_h(H))
        H = torch.tanh(self.linear_h(H))
        #x = self.sigmoid(self.linear_out(H))
        x = .5 * torch.sin(self.linear_out(H)) + .5
        x = x.view(self.batch_size, self.c_dim, self.y_dim, self.x_dim)
        #print ('G out: ', x.shape)
        return x


def coordinates(args):
    x_dim, y_dim, scale = args.x_dim, args.y_dim, args.scale
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), args.batch_size).reshape(args.batch_size, n_points, 1)
    x_mat = torch.from_numpy(x_mat).float()#.cuda()
    y_mat = torch.from_numpy(y_mat).float()#.cuda()
    r_mat = torch.from_numpy(r_mat).float()#.cuda()
    return x_mat, y_mat, r_mat


def sample(args, netG, z):
    x_vec, y_vec, r_vec = coordinates(args)
    image = netG((x_vec, y_vec, z, r_vec))
    return image


def init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data)
    return model


def latent_walk(args, z1, z2, netG):
    states = []
    for i in range(args.num_frames):
        if args.interpolation == 'linear':
            theta = (i + 0.5) / (args.num_frames)
        elif args.interpolation == 'sigmoid':
            steepness = 12
            gamma = -0.5 + (i + 0.5) / (args.num_frames)
            theta = 1 / (1 + np.exp(-gamma * steepness))

        z = theta * z2 + (1 - theta) * z1
        if args.c_dim == 1:
            states.append(sample(args, netG, z)[0][0]*255)
        else:
            states.append(sample(args, netG, z)[0].view(
                args.x_dim, args.y_dim, args.c_dim)*255)
    states = torch.stack(states).detach().numpy()
    return states


def cppn(args):
    netG = init(Generator(args))
    print (netG)
    n_images = args.n
    zs = []
    for _ in range(args.n):
        zs.append(torch.zeros(1, args.z).uniform_(-1.0, 1.0))

    if args.walk:
        k = 0
        if args.render_video:
            filename = 'cppn_walk_{}.mp4'.format(int(time.time()))
            print('Writing video filename {}'.format(filename))
            vid = imutil.Video(filename)
        for i in range(args.n):
            print('Generating latent walk {}...'.format(i))
            z1, z2 = zs[i], zs[(i + 1) % args.n]
            images = latent_walk(args, z1, z2, netG)
            print('Writing {} frames...'.format(len(images)))
            for img in images:
                if args.render_video:
                    vid.write_frame(img)
                else:
                    imwrite('{}_{}.jpg'.format(args.exp, k), img)
                k += 1
            print('Walked through {}/{} control points, {}/{} frames'.format(
                i+1, args.n, k, args.n*args.num_frames))
        if args.render_video:
            vid.finish()

    if args.sample:
        zs, _ = torch.stack(zs).sort()
        for i, z in enumerate(zs):
            img = sample(args, netG, z).cpu().detach().numpy()
            if args.c_dim == 1:
                img = img[0][0]
            else:
                img = img[0].reshape((args.x_dim, args.y_dim, args.c_dim))
            imwrite('{}_{}.png'.format(args.exp, i), img*255)

if __name__ == '__main__':

    args = load_args()
    cppn(args)
