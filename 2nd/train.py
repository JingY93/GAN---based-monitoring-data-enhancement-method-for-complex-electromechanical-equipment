
import functools

import imlib as im
import numpy as np
import pylib as py
import tensorboardX
import torch
import torchlib
import torchprob as gan
import tqdm
import matplotlib.pyplot as plt
import data
import module
from sklearn import preprocessing
import DTW
import mmd
import sorttt
# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
py.arg('--dataset', default='bearing', choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom','bearing'])
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=400)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--z_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='1-gp', choices=['none', '1-gp', '0-gp', 'lp'])
py.arg('--gradient_penalty_sample_mode', default='dragan', choices=['line', 'real', 'fake', 'dragan'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='none')
py.arg('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])  # !!!
args = py.args()

# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
use_gpu = torch.cuda.is_available()
# device = torch.device("cuda" if use_gpu else "cpu")
device = torch.device( "cpu")


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    data_loader, shape = data.make_32x32_dataset(args.dataset, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset == 'celeba':  # 64x64
    img_paths = py.glob('data/img_align_celeba', '*.jpg')
    data_loader, shape = data.make_celeba_dataset(img_paths, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'anime':  # 64x64
    img_paths = py.glob('data/faces', '*.jpg')
    data_loader, shape = data.make_anime_dataset(args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'bearing':  # 64x64
    # img_paths = py.glob('data/faces', '*.jpg')
    data_loader, shape = data.make_bearingdata(args.batch_size)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset == 'custom':
    # ======================================
    # =               custom               =
    # ======================================
    img_paths = ...  # image paths of custom dataset
    data_loader = data.make_custom_dataset(img_paths, args.batch_size, pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = ...  # 3 for 32x32 and 4 for 64x64
    # ======================================
    # =               custom               =
    # ======================================


# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = module.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = module.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
print(D)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def train_G():
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z)

    x_fake_d_logit = D(x_fake)
    G_loss = g_loss_fn(x_fake_d_logit)

    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return {'g_loss': G_loss}


def train_D(x_real):
    G.train()
    D.train()

    z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    x_fake = G(z).detach()

    x_real_d_logit = D(x_real)
    x_fake_d_logit = D(x_fake)

    x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
    gp = gan.gradient_penalty(functools.partial(D), x_real, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)

    D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# load checkpoint if exists
ckpt_dir = py.join(output_dir, 'checkpoints')
py.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    D_optimizer.load_state_dict(ckpt['D_optimizer'])
    G_optimizer.load_state_dict(ckpt['G_optimizer'])
except:
    ep, it_d, it_g = 0, 0, 0

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(py.join(output_dir, 'summaries'))
z = torch.randn(100, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

for ep_ in tqdm.trange(args.epochs, desc='Epoch Loop'):
    if ep_ < ep:
        continue
    ep += 1

    # train for an epoch
    for x_real in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):
        x_real = x_real.to(device)

        D_loss_dict = train_D(x_real)
        it_d += 1
        for k, v in D_loss_dict.items():
            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

        if it_d % args.n_d == 0:
            G_loss_dict = train_G()
            it_g += 1
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

        # sample
        if it_g % 100 == 0:
            x_fake = sample(z)
            x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))
            img = im.immerge(x_fake, n_rows=10).squeeze()
            im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % it_g))

    # save checkpoint
    torchlib.save_checkpoint({'ep': ep, 'it_d': it_d, 'it_g': it_g,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'D_optimizer': D_optimizer.state_dict(),
                              'G_optimizer': G_optimizer.state_dict()},
                             py.join(ckpt_dir, 'Epoch_(%d).ckpt' % ep),
                             max_keep=1)
    # torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))
# 对比
# 原始数据
dataset = np.load('bearing_data.npy')
dataset = dataset.flatten()
dataset = preprocessing.StandardScaler().fit_transform(dataset.flatten()[:, np.newaxis])
dataset = dataset.reshape(-1, 64, 64)
# 生成数据
x_fake = sample(z)
x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 1, 2, 3))
# plt.imshow(x_fake[99][0], cmap="gray")
# plt.imshow(dataset[100,:,:], cmap="gray")
# plt.show()
# DTW验证
s1 = dataset[100,:,:].flatten()
s2 = dataset[50,:,:].flatten()
s3 = x_fake[99][0].flatten()
s3=sorttt.sortt(s1,s3)
plt.imshow(s3.reshape(64,64), cmap="gray")
plt.show()
#
# 原始算法
distance12, paths12, max_sub12 = DTW.TimeSeriesSimilarityImprove(s1, s2)
distance13, paths13, max_sub13 = DTW.TimeSeriesSimilarityImprove(s1, s3)

print("更新前s1和s2距离：" + str(distance12))
print("更新前s1和s3距离：" + str(distance13))

# 衰减系数
weight12 = DTW.calculate_attenuate_weight(len(s1), len(s2), max_sub12)
weight13 = DTW.calculate_attenuate_weight(len(s1), len(s3), max_sub13)

# 更新距离
print("更新后s1和s2距离：" + str(distance12 * weight12))
print("更新后s1和s3距离：" + str(distance13 * weight13))

# MMD验证
s1 =torch.from_numpy(s1.reshape(64,64))
s2 = torch.from_numpy(s2.reshape(64,64))
s3 = torch.from_numpy(s3.reshape(64,64))
dict1=mmd.mmd_rbf(s1,s2)
dict2=mmd.mmd_rbf(s1,s3)
print("s1和s2MMD距离：" + str(dict1))
print("s1和s3MMD距离：" + str(dict2))
