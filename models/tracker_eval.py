import torch.nn.init
import kornia
from models.common import *


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.f1 = nn.Linear(dim_in, dim_out)
        self.f3 = nn.LeakyReLU()
        self.f4 = nn.Dropout()
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.f1(x)
        x = self.f3(x)
        x = self.f4(x)
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        x = self.e2v(E, H)
        return x


class HGNNPN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.hgconv1 = HyPConv(dim_in, dim_out)
        self.hgconv2 = HyPConv(dim_in, dim_out)

    def forward(self, x, path_pos, path_neg):
        x1 = self.hgconv1(x, path_pos)
        x2 = self.hgconv1(x, path_neg)
        x1 = x + x1 * 0.1 - x2 * 0.1
        return x1


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(FPNEncoder, self).__init__()

        self.conv_bottom_0 = ConvBlock(in_channels=in_channels,out_channels=32,n_convs=2,kernel_size=1,padding=0,downsample=False,)
        self.conv_bottom_1 = ConvBlock(in_channels=32,out_channels=64,n_convs=2,kernel_size=5,padding=0,downsample=False,)
        self.conv_bottom_2 = ConvBlock(in_channels=64,out_channels=128,n_convs=2,kernel_size=5,padding=0,downsample=False,)
        self.conv_bottom_3 = ConvBlock(in_channels=128,out_channels=256,n_convs=2,kernel_size=3,padding=0,downsample=True,)
        self.conv_bottom_4 = ConvBlock(in_channels=256,out_channels=out_channels,n_convs=2,kernel_size=3,padding=0,downsample=False,)

        self.conv_lateral_3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_2 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_0 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, bias=True)

        self.conv_dealias_3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_1 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_0 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_out = nn.Sequential(ConvBlock(in_channels=out_channels,out_channels=out_channels,n_convs=1,kernel_size=3,padding=1,downsample=False,),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,),)

        self.conv_bottleneck_out = nn.Sequential(ConvBlock(in_channels=out_channels,out_channels=out_channels,n_convs=1,kernel_size=3,padding=1,downsample=False,),
                                                 nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,),)

    def reset(self):
        return

    def forward(self, x):
        c0 = self.conv_bottom_0(x)
        c11 = c0
        c1 = self.conv_bottom_1(c11)
        c2 = self.conv_bottom_2(c1)
        c3 = self.conv_bottom_3(c2)
        c4 = self.conv_bottom_4(c3)

        p4 = c4
        p3 = self.conv_dealias_3(self.conv_lateral_3(c3) + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear"))
        p2 = self.conv_dealias_2(self.conv_lateral_2(c2) + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear"))
        p1 = self.conv_dealias_1(self.conv_lateral_1(c1)+ F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear"))
        p0 = self.conv_dealias_0(self.conv_lateral_0(c0)+ F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode="bilinear"))

        return self.conv_out(p0), self.conv_bottleneck_out(c4)


class FPNEncoder1(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(FPNEncoder1, self).__init__()

        self.conv_bottom_0 = ConvBlock(in_channels=in_channels,out_channels=32,n_convs=2,kernel_size=1,padding=0,downsample=False,)
        self.conv_bottom_0_1 = ConvBlock(in_channels=in_channels,out_channels=32,n_convs=2,kernel_size=1,padding=0,downsample=False,)
        self.conv_bottom_1 = ConvBlock(in_channels=32,out_channels=64,n_convs=2,kernel_size=5,padding=0,downsample=False,)
        self.conv_bottom_2 = ConvBlock(in_channels=64,out_channels=128,n_convs=2,kernel_size=5,padding=0,downsample=False,)
        self.conv_bottom_3 = ConvBlock(in_channels=128,out_channels=256,n_convs=2,kernel_size=3,padding=0,downsample=True,)
        self.conv_bottom_4 = ConvBlock(in_channels=256,out_channels=out_channels,n_convs=2,kernel_size=3,padding=0,downsample=False,)

        self.conv_lateral_3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_2 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv_lateral_0 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, bias=True)

        self.conv_dealias_3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_1 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_dealias_0 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,)
        self.conv_out = nn.Sequential(ConvBlock(in_channels=out_channels,out_channels=out_channels,n_convs=1,kernel_size=3,padding=1,downsample=False,),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,),)

        self.conv_bottleneck_out = nn.Sequential(ConvBlock(in_channels=out_channels,out_channels=out_channels,n_convs=1,kernel_size=3,padding=1,downsample=False,),
                                                 nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=True,),)

    def reset(self):
        return

    def forward(self, x, x1):
        c0 = self.conv_bottom_0(x)
        c0_1 = self.conv_bottom_0_1(x1)
        c11 = (c0 + c0_1) / 2
        c1 = self.conv_bottom_1(c11)
        c2 = self.conv_bottom_2(c1)
        c3 = self.conv_bottom_3(c2)
        c4 = self.conv_bottom_4(c3)

        p4 = c4
        p3 = self.conv_dealias_3(self.conv_lateral_3(c3) + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear"))
        p2 = self.conv_dealias_2(self.conv_lateral_2(c2) + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear"))
        p1 = self.conv_dealias_1(self.conv_lateral_1(c1)+ F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear"))
        p0 = self.conv_dealias_0(self.conv_lateral_0(c0)+ F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode="bilinear"))

        return self.conv_out(p0), self.conv_bottleneck_out(c4)


class JointEncoder(nn.Module):
    def __init__(self, in_channels):
        super(JointEncoder, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, n_convs=2, downsample=True)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, n_convs=2, downsample=True)
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, n_convs=2, downsample=True)
        self.conv4 = ConvBlock(in_channels=256,out_channels=256,kernel_size=3,padding=0,n_convs=1,downsample=False)

        self.flatten = nn.Flatten()

        embed_dim = 256

        self.prev_x_res = None
        self.gates = nn.Linear(2 * embed_dim, embed_dim)

        self.fusion_layer0 = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim),nn.LeakyReLU(0.1),nn.Linear(embed_dim, embed_dim),nn.LeakyReLU(0.1))
        self.output_layers = nn.Sequential(nn.Linear(embed_dim, 512), nn.LeakyReLU(0.1))

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x1 = x

        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))
        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))
        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight + x * (1 - gate_weight)

        self.prev_x_res = x

        x = self.output_layers(x)
        return x, x1


class PatchMixStereo(nn.Module):
    def __init__(self, max_disp=160, ds_rate=1, group=8, recurr=False, hgnn=True, bs=None, num_fea=None):
        super().__init__()
        self.max_disp = max_disp
        self.group = group
        self.k = 3
        self.th = 16

        self.recurr = recurr
        self.hgnn = hgnn
        if bs is not None:
            self.pos = torch.repeat_interleave(torch.repeat_interleave(torch.arange(0, max_disp//ds_rate)[None, :, None], bs, dim=0), self.k, dim=2).cuda()
        if recurr:
            self.gru = ConvLSTM1D(self.group, self.group*2, 3)
            self.corr_stem_1 = nn.Conv1d(self.group*2, 1, kernel_size=3, padding=1)
        else:
            self.corr_stem_1 = nn.Conv1d(self.group, 1, kernel_size=3, padding=1)

        if self.hgnn:
            self.HGNNP = HGNNPN(self.group, self.group)

        self.ds_rate = ds_rate

    def reset(self):
        if self.recurr:
            self.gru.reset()
        return

    def forward(self, left_feat, right_feat, start_left):
        bn, _, p, p = left_feat.shape

        cost_volume_1 = build_gwc_volume(left_feat, right_feat, self.max_disp, self.group, start_left, self.ds_rate)
        mask = (cost_volume_1.sum(dim=1, keepdim=True) == 0)

        if self.hgnn:
            feature = cost_volume_1.transpose(1, 2).contiguous()
            _, row_idx_k = torch.topk(square_distance(feature, feature), self.k, dim=2, sorted=False, largest=False)

            if hasattr(self, 'pos'):
                position = self.pos
            else:
                position = torch.repeat_interleave(torch.repeat_interleave(torch.arange(0, row_idx_k.shape[1])[None, :, None], row_idx_k.shape[0], dim=0), row_idx_k.shape[2], dim=2).cuda()

            if position.shape[0] != row_idx_k.shape[0]:
                position = torch.repeat_interleave(torch.repeat_interleave(torch.arange(0, row_idx_k.shape[1])[None, :, None], row_idx_k.shape[0], dim=0), row_idx_k.shape[2], dim=2).cuda()

            space_dist = torch.abs(row_idx_k - position)
            mask_pos = space_dist < self.th
            neg_idx = row_idx_k.clone()
            neg_idx[mask_pos] = row_idx_k.shape[1]
            row_idx_k[~mask_pos] = row_idx_k.shape[1]

            batch_idx_k = torch.arange(bn).unsqueeze(1).unsqueeze(2).repeat(1, feature.shape[1], self.k).contiguous()
            col_idx_k = torch.arange(feature.shape[1]).unsqueeze(1).repeat(1, self.k).repeat(bn, 1, 1).contiguous()
            hg_pos = torch.zeros([bn, feature.shape[1] + 1, feature.shape[1]]).cuda()
            hg_neg = torch.zeros([bn, feature.shape[1] + 1, feature.shape[1]]).cuda()
            hg_pos[batch_idx_k, row_idx_k, col_idx_k] = 1
            hg_neg[batch_idx_k, neg_idx, col_idx_k] = 1
            hg_pos = hg_pos[:, :-1, :]
            hg_neg = hg_neg[:, :-1, :]
            if hg_neg.sum() + hg_pos.sum() != row_idx_k.shape[0]*row_idx_k.shape[1]*row_idx_k.shape[2]:
                print('Warning')
            cost_volume_1 = self.HGNNP(feature, hg_pos, hg_neg).transpose(1, 2)


        cost_volume_agg = self.corr_stem_1(cost_volume_1)

        cost_volume_agg[mask] = -1e9

        prob = F.softmax(cost_volume_agg, dim=2).reshape(bn, -1)
        init_disp = disparity_regression(prob, self.max_disp // self.ds_rate) * self.ds_rate

        return init_disp


class TrackerNetEval(nn.Module):
    def __init__(self, feature_dim=1024, patch_size=31, input_channels=None, channels_in_per_patch=10, hgnn=False, bs=None):
        super(TrackerNetEval, self).__init__()

        self.feature_dim = feature_dim
        self.redir_dim = 128

        self.reference_encoder = FPNEncoder1(1, feature_dim)
        self.target_encoder = FPNEncoder(channels_in_per_patch, feature_dim)

        self.reference_redir = nn.Conv2d(feature_dim, self.redir_dim, kernel_size=3, padding=1)
        self.target_redir = nn.Conv2d(feature_dim, self.redir_dim, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=2)
        self.joint_encoder = JointEncoder(in_channels=1 + 2 * self.redir_dim)

        self.predictor = nn.Linear(in_features=512, out_features=2, bias=False)
        self.flatten = nn.Flatten()
        self.patch_size = patch_size
        self.learn = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Linear(128, 5))

        self.f_ref, self.d_ref = None, None

        self.conv_bottom_0 = ConvBlock(in_channels=channels_in_per_patch, out_channels=32, n_convs=2, kernel_size=1, padding=0, downsample=False)
        self.stem_stereo = nn.Sequential(BasicConv_IN(32, 48, kernel_size=3, stride=1, padding=1), nn.Conv2d(48, 48, 3, 1, 1, bias=False), nn.InstanceNorm2d(48), nn.ReLU())
        self.stereo_matching = PatchMixStereo(max_disp=160, ds_rate=4, recurr=False, group=8, hgnn=hgnn, bs=bs, num_fea=48)

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc_out.weight)

    def reset(self):
        self.d_ref, self.f_ref = None, None
        self.joint_encoder.reset()

    def forward(self, ev_frame_left, ev_frame_right, ref, pos, pos_r, pred=None):
        self.reference_encoder = self.reference_encoder.eval()
        self.target_encoder = self.target_encoder.eval()
        self.reference_redir = self.reference_redir.eval()
        self.target_redir = self.target_redir.eval()
        self.joint_encoder = self.joint_encoder.eval()
        self.predictor = self.predictor.eval()

        ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4])
        u_centers_l = pos
        # u_centers_r = pos_r
        b, n, _ = u_centers_l.shape
        u_centers_l = u_centers_l.reshape(-1, 2)
        # u_centers_r = u_centers_r.reshape(-1, 2)
        _, c, h, w = ev_frame_left.shape
        ev_frame_left = ev_frame_left.unsqueeze(1).repeat(1, n, 1, 1, 1)
        ev_frame_left = ev_frame_left.reshape(-1, c, h, w)
        # ev_frame_right_copy = ev_frame_right.unsqueeze(1).repeat(1, n, 1, 1, 1)
        # ev_frame_right_copy = ev_frame_right_copy.reshape(-1, c, h, w)

        patch_size = self.patch_size

        if pred is None:
            ref1 = ref
        else:
            [scale, angle, sx, sy] = pred
            a = kornia.geometry.transform.get_affine_matrix2d(torch.zeros_like(scale), torch.zeros_like(scale), 1.0/scale, angle, sx=sx, sy=sy)
            grid = F.affine_grid(a[:, :2, :], size=ref.shape)
            ref1 = F.grid_sample(ref, grid)

        x = extract_glimpse(ev_frame_left, (patch_size, patch_size), u_centers_l + 0.5)
        # x_r = extract_glimpse(ev_frame_right_copy, (patch_size, patch_size), u_centers_r + 0.5)   # (bs*N_fea), C, H, W

        f0, _ = self.target_encoder(x)
        # f0_r, _ = self.target_encoder(x_r)

        self.f_ref, self.d_ref = self.reference_encoder(ref, ref1)
        self.f_ref = self.reference_redir(self.f_ref)

        f_corr = (f0 * self.d_ref).sum(dim=1, keepdim=True)
        # f_corr_r = (f0_r * self.d_ref.detach()).sum(dim=1, keepdim=True)

        f_corr = self.softmax(f_corr.view(-1, 1, self.patch_size * self.patch_size)).view(-1, 1, self.patch_size, self.patch_size)
        # f_corr_r = self.softmax(f_corr_r.view(-1, 1, self.patch_size * self.patch_size)).view(-1, 1, self.patch_size, self.patch_size)

        tf = self.target_redir(f0)
        # tf_r = self.target_redir(f0_r)

        f = torch.cat([f_corr, tf, self.f_ref], dim=1)
        # f_r = torch.cat([f_corr_r, tf_r, self.f_ref], dim=1)
        # cat_f = torch.cat([f, f_r], dim=0)

        f, f1 = self.joint_encoder(f)

        f = self.predictor(f)
        f_left = f[:b*n].reshape(b, n, 2)
        # f_right = f[b*n:].reshape(b, n, 2)


        f1 = f1[:b*n]
        pred = F.sigmoid(self.learn(f1))
        scale = pred[:, :2] + 0.5
        angle = (pred[:,2] - 0.5) * 15
        sx = pred[:,3] - 0.5
        sy = pred[:,4] - 0.5

        f0_st = self.stem_stereo(self.conv_bottom_0(x))
        f0_r_sframe = self.stem_stereo(self.conv_bottom_0(ev_frame_right))

        init_disp = self.stereo_matching(f0_st, f0_r_sframe, u_centers_l)

        return f_left, init_disp.reshape(b, -1), [scale, angle, sx, sy]


