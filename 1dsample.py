import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

# 输入纯色图像，看是否能采样出纯色图像
# training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1
training_seq = torch.ones(64, 32, 128) # features are normalized from 0 to 1
#for i in range(64):
#    scale = i / 640
#    training_seq[i,:] = training_seq[i,:] * scale

# train
'''
loss = diffusion(training_seq)
loss.backward()
# Or using trainer
'''
trainer = Trainer1D(
    diffusion,
    dataset = training_seq,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 100,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

#采样4张
sampled_seq = diffusion.sample(batch_size = 4)
#shape回4张32通道128维向量
sampled_seq.shape # (4, 32, 128)
rand_seq = torch.ones(4, 32, 128)
#测试
#原始数据 mean 1 var 0
#纯随机   mean 0.5 var 0.08 
#7个训练步骤 mean 0.6 var 0.11 
#100个训练步骤 mean 0.86 var 0.04 更接近原始分布了，重采样也很稳定



print('锚点')