import math
import warnings
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from mamba_ssm import Mamba
    _HAVE_MAMBA = True
except Exception as _e:
    _HAVE_MAMBA = False
    _MAMBA_IMPORT_ERROR = _e


class IBN2d(nn.Module):
    def __init__(self, num_channels: int, in_ratio: float = 0.5, gn_groups: int = 4):
        super().__init__()
        split = int(round(num_channels * in_ratio))
        split = max(1, min(split, num_channels - 1))
        self.split = split
        self.inorm = nn.InstanceNorm2d(split, affine=True, track_running_stats=False)
        self.gnorm = nn.GroupNorm(gn_groups, num_channels - split)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.split, x.shape[1] - self.split], dim=1)
        return torch.cat([self.inorm(x1), self.gnorm(x2)], dim=1)


class DSConvDown(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 2, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSConv(nn.Module):

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class PlainConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        if cout % 8 == 0:
            g = 8
        elif cout % 4 == 0:
            g = 4
        elif cout % 2 == 0:
            g = 2
        else:
            g = 1
        self.norm = nn.GroupNorm(g, cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FrequencyAttention(nn.Module):

    def __init__(self, C: int, mode: str = "full"):
        super().__init__()
        self.C = C
        self.mode = mode
        self.eps = 1e-8

        if mode in ("full", "spatial_only"):
            self.spatial_weight = nn.Parameter(torch.ones(1, C, 1, 1) * 0.1)

        if mode in ("full", "freq_only"):
            self.freq_weight = nn.Parameter(torch.ones(1, C, 1, 1) * 0.1)
            self.tanh_in_scale = nn.Parameter(torch.tensor(5.0))
            self.tanh_out_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        B, C, H, W = x.shape

        if self.mode == "spatial_only":
            spatial_mean = x.mean(dim=(-2, -1), keepdim=True).clamp(-5, 5)
            spatial_std = x.std(dim=(-2, -1), keepdim=True).clamp(min=self.eps, max=5)
            spatial_input = torch.clamp(self.spatial_weight * (spatial_mean + spatial_std), -5, 5)
            spatial_attention = torch.sigmoid(spatial_input)
            return x * spatial_attention

        x_scaled = torch.tanh(x / self.tanh_in_scale.to(x.dtype)) * self.tanh_out_scale.to(x.dtype)

        try:
            x_freq = torch.fft.fft2(x_scaled, dim=(-2, -1), norm='ortho')
            x_freq_mag = torch.abs(x_freq)
            freq_mean = x_freq_mag.mean(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
            freq_std = x_freq_mag.std(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        except Exception:
            freq_mean = x.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
            freq_std = x.std(dim=(-2, -1), keepdim=True).clamp(min=self.eps)

        freq_input = torch.clamp(self.freq_weight * (freq_mean + freq_std), -5, 5)
        freq_attention = torch.sigmoid(freq_input)

        if self.mode == "freq_only":
            combined_attention = torch.clamp(freq_attention, 0.1, 2.0)
            return x * combined_attention

        spatial_mean = x.mean(dim=(-2, -1), keepdim=True).clamp(-5, 5)
        spatial_std = x.std(dim=(-2, -1), keepdim=True).clamp(min=self.eps, max=5)
        spatial_input = torch.clamp(self.spatial_weight * (spatial_mean + spatial_std), -5, 5)
        spatial_attention = torch.sigmoid(spatial_input)

        combined_attention = freq_attention * spatial_attention
        combined_attention = torch.clamp(combined_attention, 0.1, 2.0)

        return x * combined_attention


class CSA(nn.Module):

    def __init__(self, C: int, reduction: int = 8):
        super().__init__()
        self.C = C
        mid = max(C // reduction, 4)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, C, 1, bias=False),
            nn.Sigmoid()
        )


        self.spatial_conv = nn.Conv2d(2, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        ca = self.channel_att(x)
        x_ca = x * ca

        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = torch.sigmoid(self.spatial_conv(spatial_input))

        return x_ca * sa


class PlainASPP(nn.Module):
    def __init__(self, C: int, rates=(1, 2, 3), drop: float = 0.1):
        super().__init__()
        self.C = C
        mid = max(C // 6, 8)

        self.branches = nn.ModuleList()
        for r in rates:
            branch = nn.Sequential(
                nn.Conv2d(C, C, 3, 1, r, dilation=r, groups=C, bias=False),
                nn.Conv2d(C, mid, 1, 1, 0, bias=False),
                nn.GELU()
            )
            self.branches.append(branch)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, mid, 1, bias=False),
            nn.GELU()
        )

        total_mid = mid * (len(rates) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_mid, C, 1, bias=False),
            nn.Dropout2d(drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        features = []
        for branch in self.branches:
            feat = branch(x)
            features.append(feat)

        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=True)
        features.append(global_feat)

        fused_features = torch.cat(features, dim=1)
        out = self.fuse(fused_features)
        out = out + x
        return out


class FSDA_ASPP(nn.Module):

    def __init__(self, C: int, rates=(1, 2, 3), drop: float = 0.1, freq_mode: str = "full"):
        super().__init__()
        self.C = C
        self.freq_mode = freq_mode
        mid = max(C // 6, 8)

        self.branches = nn.ModuleList()
        for r in rates:
            branch = nn.Sequential(
                nn.Conv2d(C, C, 3, 1, r, dilation=r, groups=C, bias=False),
                nn.Conv2d(C, mid, 1, 1, 0, bias=False),
                nn.GELU()
            )
            self.branches.append(branch)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, mid, 1, bias=False),
            nn.GELU()
        )

        self.freq_attention = FrequencyAttention(C, mode=freq_mode)
        self.cs_attention = CSA(mid * (len(rates) + 1), reduction=4)

        total_mid = mid * (len(rates) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_mid, C, 1, bias=False),
            nn.Dropout2d(drop)
        )

        self.adaptive_recalibration = True

    def _adaptive_recalibration(self, x_orig, x_enhanced):

        with torch.no_grad():

            x_orig_norm = torch.clamp(x_orig, -10, 10)
            x_enhanced_norm = torch.clamp(x_enhanced, -10, 10)


            orig_flat = x_orig_norm.view(x_orig_norm.size(0), x_orig_norm.size(1), -1)
            enhanced_flat = x_enhanced_norm.view(x_enhanced_norm.size(0), x_enhanced_norm.size(1), -1)


            orig_norm = F.normalize(orig_flat, p=2, dim=2, eps=1e-8)
            enhanced_norm = F.normalize(enhanced_flat, p=2, dim=2, eps=1e-8)


            similarity = (orig_norm * enhanced_norm).sum(dim=2, keepdim=True).unsqueeze(-1)
            confidence = torch.sigmoid(similarity * 2.0)
            confidence = torch.clamp(confidence, 0.1, 0.9)


        alpha = 0.3
        result = (1 - alpha) * x_orig + alpha * x_enhanced
        return torch.clamp(result, -5, 5)

    def forward(self, x):
        B, C, H, W = x.shape

        x_enhanced = self.freq_attention(x)

        features = []
        for branch in self.branches:
            feat = branch(x_enhanced)
            features.append(feat)

        global_feat = self.global_branch(x_enhanced)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=True)
        features.append(global_feat)

        fused_features = torch.cat(features, dim=1)

        attended_features = self.cs_attention(fused_features)

        out = self.fuse(attended_features)

        if self.adaptive_recalibration:
            out = self._adaptive_recalibration(x, out)
        else:
            out = out + x

        return out




class DLESkip(nn.Module):

    def __init__(self, channels: int, mid_ratio: int = 8, att_mid: int = 8):
        super().__init__()
        C = channels
        self.struct_dw = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.struct_bn = nn.BatchNorm2d(C)
        self.struct_act = nn.GELU()

        mid = max(att_mid, C // mid_ratio, 8)
        self.sem_reduce = nn.Conv2d(C, mid, 1, bias=False)
        self.sem_bn = nn.BatchNorm2d(mid)
        self.sem_act = nn.GELU()
        self.sem_expand = nn.Conv2d(mid, C, 1, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1, C, 1, 1))

        fuse_mid = mid
        self.fuse_reduce = nn.Conv2d(2 * C, fuse_mid, 1, bias=False)
        self.fuse_act = nn.GELU()
        self.fuse_expand = nn.Conv2d(fuse_mid, C, 1, bias=False)

        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = skip.shape

        s = self.struct_dw(skip)
        s = self.struct_bn(s)
        s = self.struct_act(s)

        with torch.no_grad():
            gray = skip.mean(dim=1, keepdim=True)
            gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
            gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
            grad = (gx + gy).clamp(0, 1)
        s = s * (1.0 + 0.15 * grad)

        x = self.sem_reduce(skip)
        x = self.sem_bn(x)
        x = self.sem_act(x)
        x = self.sem_expand(x)

        dec_pool = F.adaptive_avg_pool2d(dec, 1)
        skip_pool = F.adaptive_avg_pool2d(x, 1)
        sim = (dec_pool * skip_pool).sum(dim=1, keepdim=True)
        sim = torch.sigmoid(sim)
        sem = x * (0.7 + 0.6 * sim)

        alpha = torch.sigmoid(self.alpha)
        blended = s * (1.0 - alpha) + sem * alpha

        cat = torch.cat([blended, s], dim=1)
        f = self.fuse_reduce(cat)
        f = self.fuse_act(f)
        f = self.fuse_expand(f)
        f = self.out_bn(f)
        return f

class RMSNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class _SharedMamba(nn.Module):
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not _HAVE_MAMBA:
            raise ImportError(
                f"[mamba_ssm] 未安装或导入失败：{_MAMBA_IMPORT_ERROR}\n"
                f"请先 `pip install mamba-ssm`."
            )
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x_tokens):
        return self.mamba(x_tokens)


class GSS_Mamba(nn.Module):

    def __init__(self, input_dim, output_dim, d_state=8, d_conv=4, expand=1.5, r: int = 4, g: int = 2,
                 per_channel_out_res: bool = False):
        super().__init__()

        r = max(1, r)
        assert input_dim % r == 0, f"input_dim({input_dim}) 必须能被 r({r}) 整除"
        mid = input_dim // r

        g = min(g, max(1, mid // 4))
        assert mid % g == 0, f"mid({mid}) 必须能被 g({g}) 整除"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mid = mid
        self.g = g
        self.group_dim = mid // g
        self.per_channel_out_res = bool(per_channel_out_res)


        self.norm_in = nn.GroupNorm(min(4, input_dim//4), input_dim) if input_dim >= 8 else nn.Identity()


        self.reduce = DSConv(input_dim, mid, k=1, s=1, p=0)
        self.expand = DSConv(mid, output_dim, k=1, s=1, p=0)


        self.shared_mamba = _SharedMamba(self.group_dim, d_state=d_state, d_conv=d_conv, expand=expand)


        self.res_scale = nn.Parameter(torch.tensor(0.5))
        if self.per_channel_out_res:
            self.out_res_scale = nn.Parameter(torch.full((1, output_dim, 1, 1), 0.1))
        else:
            self.out_res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C, H, W = x.shape


        x_norm = self.norm_in(x) if not isinstance(self.norm_in, nn.Identity) else x
        z = self.reduce(x_norm)



        if self.g > 1:
            groups = torch.chunk(z, self.g, dim=1)
        else:
            groups = [z]

        T = H * W
        outs = []
        for g in groups:
            tok = g.reshape(B, self.group_dim, T).transpose(1, 2).contiguous()
            tok_out = self.shared_mamba(tok)

            tok_out = tok_out + self.res_scale * tok
            g_out = tok_out.transpose(1, 2).reshape(B, self.group_dim, H, W).contiguous()
            outs.append(g_out)

        z_out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        y = self.expand(z_out)


        if self.input_dim == self.output_dim:
            y = y + self.out_res_scale.to(y.dtype) * x

        return y




class FSM_UNet(nn.Module):

    def __init__(self, num_classes: int = 1, input_channels: int = 3, c_list=None,
                 use_ibn_early: bool = True, ibn_ratio: float = 0.5,
                 use_ds_down: bool = True,
                 use_dle_skips: bool = True, dle_mid_ratio: int = 8, dle_att_mid: int = 8,
                 deep_supervision: bool = True,
                 use_gssm_blocks: bool = True, use_fsda_aspp: bool = True,
                 aspp_variant: str = "D",
                 **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f'FSM_UNet: 忽略未使用的参数: {list(kwargs.keys())}', UserWarning)
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]

        self.use_ibn_early = use_ibn_early
        self.use_ds_down = use_ds_down
        self.use_dle_skips = bool(use_dle_skips)
        self.deep_supervision = deep_supervision
        self.use_gssm_blocks = bool(use_gssm_blocks)
        self.use_fsda_aspp = bool(use_fsda_aspp)
        self.aspp_variant = aspp_variant.upper()
        assert self.aspp_variant in ("A", "B", "C", "D"), \
            f"aspp_variant must be 'A', 'B', 'C', or 'D', got {aspp_variant}"

        self.gssm_r = kwargs.get('gssm_r', 4)
        self.gssm_g = kwargs.get('gssm_g', 2)
        self.gssm_expand = kwargs.get('gssm_expand', 1.5)

        self.stem = nn.Conv2d(input_channels, c_list[0], 3, 1, 1)
        self.enc2 = DSConv(c_list[0], c_list[1], 3, 1, 1)
        self.enc3 = DSConv(c_list[1], c_list[2], 3, 1, 1)


        if self.use_gssm_blocks:
            self.enc4 = GSS_Mamba(input_dim=c_list[2], output_dim=c_list[3], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
            self.enc5 = GSS_Mamba(input_dim=c_list[3], output_dim=c_list[4], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
            self.enc6 = GSS_Mamba(input_dim=c_list[4], output_dim=c_list[5], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
        else:
            self.enc4 = PlainConvBlock(c_list[2], c_list[3], k=3, s=1, p=1)
            self.enc5 = PlainConvBlock(c_list[3], c_list[4], k=3, s=1, p=1)
            self.enc6 = PlainConvBlock(c_list[4], c_list[5], k=3, s=1, p=1)

        if use_ibn_early:
            self.n1 = IBN2d(c_list[0], in_ratio=ibn_ratio, gn_groups=4)
            self.n2 = IBN2d(c_list[1], in_ratio=ibn_ratio, gn_groups=4)
            self.n3 = IBN2d(c_list[2], in_ratio=ibn_ratio, gn_groups=4)
        else:
            self.n1 = nn.GroupNorm(4, c_list[0])
            self.n2 = nn.GroupNorm(4, c_list[1])
            self.n3 = nn.GroupNorm(4, c_list[2])
        self.n4 = nn.GroupNorm(4, c_list[3])
        self.n5 = nn.GroupNorm(4, c_list[4])

        if use_ds_down:
            self.d1 = DSConvDown(c_list[0]); self.d2 = DSConvDown(c_list[1]); self.d3 = DSConvDown(c_list[2])
            self.d4 = DSConvDown(c_list[3]); self.d5 = DSConvDown(c_list[4])
        else:
            self.d1 = self.d2 = self.d3 = self.d4 = self.d5 = None

        if self.use_fsda_aspp:
            if self.aspp_variant == "A":
                self.aspp = PlainASPP(c_list[5], rates=(1, 2, 3), drop=0.1)
            elif self.aspp_variant == "B":
                self.aspp = FSDA_ASPP(c_list[5], rates=(1, 2, 3), drop=0.1, freq_mode="freq_only")
            elif self.aspp_variant == "C":
                self.aspp = FSDA_ASPP(c_list[5], rates=(1, 2, 3), drop=0.1, freq_mode="spatial_only")
            else:
                self.aspp = FSDA_ASPP(c_list[5], rates=(1, 2, 3), drop=0.1, freq_mode="full")
        else:
            self.aspp = nn.Identity()

        if self.use_gssm_blocks:
            self.dec1 = GSS_Mamba(input_dim=c_list[5], output_dim=c_list[4], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
            self.dec2 = GSS_Mamba(input_dim=c_list[4], output_dim=c_list[3], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
            self.dec3 = GSS_Mamba(input_dim=c_list[3], output_dim=c_list[2], r=self.gssm_r, g=self.gssm_g, expand=self.gssm_expand, per_channel_out_res=True)
        else:
            self.dec1 = PlainConvBlock(c_list[5], c_list[4], k=3, s=1, p=1)
            self.dec2 = PlainConvBlock(c_list[4], c_list[3], k=3, s=1, p=1)
            self.dec3 = PlainConvBlock(c_list[3], c_list[2], k=3, s=1, p=1)
        self.dec4 = DSConv(c_list[2], c_list[1], 3, 1, 1)
        self.dec5 = DSConv(c_list[1], c_list[0], 3, 1, 1)

        if self.use_dle_skips:
            self.dle5 = DLESkip(c_list[4], mid_ratio=dle_mid_ratio, att_mid=dle_att_mid)
            self.dle4 = DLESkip(c_list[3], mid_ratio=dle_mid_ratio, att_mid=dle_att_mid)
            self.dle3 = DLESkip(c_list[2], mid_ratio=dle_mid_ratio, att_mid=dle_att_mid)
        else:
            self.dle5 = self.dle4 = self.dle3 = None

        self.head_main = nn.Conv2d(c_list[0], num_classes, 1)
        self.head_aux8 = nn.Conv2d(c_list[2], num_classes, 1)
        self.head_aux4 = nn.Conv2d(c_list[1], num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02);
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def _down(self, x: torch.Tensor, conv: nn.Module, norm: nn.Module, down: Optional[nn.Module]) -> torch.Tensor:
        x = conv(x); x = norm(x); x = F.silu(x)
        x = F.max_pool2d(x, 2, 2) if down is None else down(x)
        return x

    def _forward_once(self, x: torch.Tensor, do_refine: bool = False) -> tuple:
        e1 = self._down(x, self.stem, self.n1, self.d1)
        e2 = self._down(e1, self.enc2, self.n2, self.d2)
        e3 = self._down(e2, self.enc3, self.n3, self.d3)
        e4 = self._down(e3, self.enc4, self.n4, self.d4)
        e5 = self._down(e4, self.enc5, self.n5, self.d5)
        b  = F.silu(self.enc6(e5))
        b  = self.aspp(b)


        d5 = F.silu(self.dec1(b))
        e5g = e5
        e5a = self.dle5(e5g, d5) if self.dle5 is not None else e5g
        d5 = d5 + e5a

        d4_pre = F.silu(F.interpolate(self.dec2(d5), scale_factor=(2,2), mode='bilinear', align_corners=True))
        e4g = e4
        e4a = self.dle4(e4g, d4_pre) if self.dle4 is not None else e4g
        d4 = d4_pre + e4a

        d3_pre = F.silu(F.interpolate(self.dec3(d4), scale_factor=(2,2), mode='bilinear', align_corners=True))
        e3g = e3
        e3a = self.dle3(e3g, d3_pre) if self.dle3 is not None else e3g
        d3 = d3_pre + e3a


        d2 = F.silu(F.interpolate(self.dec4(d3), scale_factor=(2,2), mode='bilinear', align_corners=True))
        d2 = d2 + e2
        d1 = F.silu(F.interpolate(self.dec5(d2), scale_factor=(2,2), mode='bilinear', align_corners=True))
        d1 = d1 + e1

        logit = F.interpolate(self.head_main(d1), scale_factor=(2,2), mode='bilinear', align_corners=True)


        logit = torch.clamp(logit, -10, 10)

        seg = torch.sigmoid(logit)

        seg = torch.clamp(seg, 1e-7, 1.0 - 1e-7)

        if self.deep_supervision and self.training:

            aux8_logit = F.interpolate(self.head_aux8(d3), scale_factor=(2,2), mode='bilinear', align_corners=True)
            aux8_logit = torch.clamp(aux8_logit, -10, 10)
            aux8 = torch.sigmoid(aux8_logit)
            aux8 = torch.clamp(aux8, 1e-7, 1.0 - 1e-7)

            aux4_logit = self.head_aux4(d2)
            aux4_logit = torch.clamp(aux4_logit, -10, 10)
            aux4 = torch.sigmoid(aux4_logit)
            aux4 = torch.clamp(aux4, 1e-7, 1.0 - 1e-7)

            return (seg, aux4, aux8)
        return seg

    def forward(self, x: torch.Tensor):
        if self.training:
            return self._forward_once(x, do_refine=False)
        else:
            return self._forward_once(x, do_refine=False)
