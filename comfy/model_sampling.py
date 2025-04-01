import torch
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
import math

def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        根据给定的噪声水平和是否最大化去噪的标志，对噪声进行缩放。

        参数:
        sigma (float): 输入噪声的标准差。
        noise (Tensor): 待处理的噪声张量。
        latent_image (Tensor): 原始潜在图像张量，用于与噪声相加。
        max_denoise (bool, 可选): 指示是否最大化去噪。默认为False。

        返回:
        Tensor: 缩放后的噪声张量，已与潜在图像相加。
        """
        # 根据是否最大化去噪来调整噪声的缩放因子
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma

        # 将缩放后的噪声与潜在图像相加，得到最终的输出
        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent

class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class EDM(V_PREDICTION):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class CONST:
    """
    该类定义了一些常量和方法，用于处理图像去噪和噪声缩放。

    方法:
    - calculate_input: 根据给定的噪声水平，计算输入图像中的噪声。
    - calculate_denoised: 计算去噪后的图像，通过减去模型输出与噪声水平的乘积。
    - noise_scaling: 根据噪声水平，对噪声和潜在图像进行加权混合。
    - inverse_noise_scaling: 根据噪声水平，反向调整潜在图像。
    """

    def calculate_input(self, sigma, noise):
        """
        计算输入图像中的噪声。

        参数:
        sigma (Tensor): 噪声水平。
        noise (Tensor): 输入图像中的噪声。

        返回:
        Tensor: 与噪声水平相关的噪声。
        """
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        """
        计算去噪后的图像。

        参数:
        sigma (Tensor): 噪声水平。
        model_output (Tensor): 模型的输出图像。
        model_input (Tensor): 模型的输入图像。

        返回:
        Tensor: 去噪后的图像，通过调整模型输出与噪声水平的乘积得到。
        """
        # 调整sigma的形状，使其与model_output匹配，以便进行元素-wise乘法
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        根据噪声水平，对噪声和潜在图像进行加权混合。

        参数:
        sigma (Tensor): 噪声水平。
        noise (Tensor): 输入图像中的噪声。
        latent_image (Tensor): 潜在的干净图像。
        max_denoise (bool): 是否使用最大去噪模式，默认为False。

        返回:
        Tensor: 根据噪声水平加权混合后的图像。
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        """
        根据噪声水平，反向调整潜在图像。

        参数:
        sigma (Tensor): 噪声水平。
        latent (Tensor): 潜在图像。

        返回:
        Tensor: 调整后的潜在图像。
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
        return latent / (1.0 - sigma)

class X0(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output

class IMG_TO_IMG(X0):
    def calculate_input(self, sigma, noise):
        return noise


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None, zsnr=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        timesteps = sampling_settings.get("timesteps", 1000)

        if zsnr is None:
            zsnr = sampling_settings.get("zsnr", False)

        self._register_schedule(given_betas=None, beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3, zsnr=zsnr)
        self.sigma_data = 1.0

    def _register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, zsnr=False):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        # 每个位置的新值是前面所有（含当前）元素的累积乘积。因为原值都是小于1的，对应位置元素值都变小了
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        # self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        # 元素值逐步递增
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if zsnr:
            sigmas = rescale_zero_terminal_snr_sigmas(sigmas)

        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        # 计算自然对数
        log_sigma = sigma.log()
        # 差值
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        # 计算绝对值差，并找到沿第一个维度的最小值索引
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        # 将时间步转换为浮点数并限制其范围，以适应self.log_sigmas的索引范围
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))

        # 计算时间步对应的较低索引
        low_idx = t.floor().long()

        # 计算时间步对应的较高索引
        high_idx = t.ceil().long()

        # 计算时间步的小数部分，用于后续的线性插值
        w = t.frac()

        # 使用线性插值计算log_sigma的值
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]

        # 返回sigma的值，即log_sigma的指数，并确保其在正确的设备上
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()

class ModelSamplingDiscreteEDM(ModelSamplingDiscrete):
    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

class ModelSamplingContinuousEDM(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        sigma_min = sampling_settings.get("sigma_min", 0.002)
        sigma_max = sampling_settings.get("sigma_max", 120.0)
        sigma_data = sampling_settings.get("sigma_data", 1.0)
        self.set_parameters(sigma_min, sigma_max, sigma_data)

    def set_parameters(self, sigma_min, sigma_max, sigma_data):
        self.sigma_data = sigma_data
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()

        self.register_buffer('sigmas', sigmas) #for compatibility with some schedulers
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp((math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min)


class ModelSamplingContinuousV(ModelSamplingContinuousEDM):
    def timestep(self, sigma):
        return sigma.atan() / math.pi * 2

    def sigma(self, timestep):
        return (timestep * math.pi / 2).tan()


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

class ModelSamplingDiscreteFlow(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)

class StableCascadeSampling(ModelSamplingDiscrete):
    def __init__(self, model_config=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(sampling_settings.get("shift", 1.0))

    def set_parameters(self, shift=1.0, cosine_s=8e-3):
        self.shift = shift
        self.cosine_s = torch.tensor(cosine_s)
        self._init_alpha_cumprod = torch.cos(self.cosine_s / (1 + self.cosine_s) * torch.pi * 0.5) ** 2

        #This part is just for compatibility with some schedulers in the codebase
        self.num_timesteps = 10000
        sigmas = torch.empty((self.num_timesteps), dtype=torch.float32)
        for x in range(self.num_timesteps):
            t = (x + 1) / self.num_timesteps
            sigmas[x] = self.sigma(t)

        self.set_sigmas(sigmas)

    def sigma(self, timestep):
        alpha_cumprod = (torch.cos((timestep + self.cosine_s) / (1 + self.cosine_s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod)

        if self.shift != 1.0:
            var = alpha_cumprod
            logSNR = (var/(1-var)).log()
            logSNR += 2 * torch.log(1.0 / torch.tensor(self.shift))
            alpha_cumprod = logSNR.sigmoid()

        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        return ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5

    def timestep(self, sigma):
        # 计算变量的初始公式，基于给定的sigma值
        var = 1 / ((sigma * sigma) + 1)

        # 将变量var的值限制在0到1之间，确保其在一个合理的范围内
        var = var.clamp(0, 1.0)

        # 准备cosine_s和_init_alpha_cumprod张量，确保它们与var在相同的设备上
        s, min_var = self.cosine_s.to(var.device), self._init_alpha_cumprod.to(var.device)

        # 计算最终的t值，使用反余弦和比例因子，这一步涉及到了三角函数和张量运算
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s

        # 返回计算得到的t值，它代表了根据当前变量计算出的结果
        return t

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0

        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent))


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)
