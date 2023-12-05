from configs.default_cifar10_configs import get_default_configs
from configs.ve.single_13M_cifar10_ncsnpp_continuous import get_config as get_13M
from ml_collections import ConfigDict

# 410MB

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.n_iters = 950001

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # evaluation
  evaluate = config.eval
  evaluate.begin_ckpt = 12
  evaluate.end_ckpt = 12
  evaluate.enable_sampling = True
  evaluate.enable_loss = False
  evaluate.batch_size = 2048
  evaluate.num_samples = 50000
  evaluate.multi_model_sampling = False
  evaluate.enable_time = False

  # multi model sampling
  mult = evaluate.multi = ConfigDict()
  mult.model_configs = [get_13M()]
  mult.state_paths = ['./assets/ve_cifar10_ncsnpp_continuous_13M/checkpoint_26.pth']
  mult.step_counts = [100,900]

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.conv_size = 3

  return config
