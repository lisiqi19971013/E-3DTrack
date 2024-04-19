from pathlib import Path
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from utils.data_preprocess import generate_subseq


torch.multiprocessing.set_sharing_strategy("file_system")


def extract_glimpse(input, size, offsets, centered=False, normalized=False, mode="nearest", padding_mode="zeros"):
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W / 2, H / 2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError("Invalid parameter that offsets centered but not normlized")

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype, device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype, device=input.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)
    offsets_grid = offsets[:, None, None, :] + grid[None, ...]
    offsets_grid = (offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])) / offsets_grid.new_tensor([W / 2, H / 2])

    return torch.nn.functional.grid_sample(input, offsets_grid.float(), mode=mode, align_corners=True, padding_mode=padding_mode)


class RecurrentStereoSubseq(Dataset):
    def __init__(self, root_dir='./dataset', sequence_name='', patch_size=31):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size

        self.dt, self.dt_us = 0.004, 0.004 * 1e6
        self.sequence_dir = self.root_dir / self.sequence_name
        self.sequence_name = sequence_name
        self.device = 'cpu'
        self.x_ref = torch.zeros(1)

        self.width = 346
        self.height = 260
        self.frame_dir = self.root_dir / sequence_name / "image_680"

        self.trigger_ts_arr = np.load(str(self.sequence_dir / "ts_trigger_crop.npy"))
        self.frame_ts_arr = np.load(str(self.sequence_dir / "ts_680.npy"))
        self.n_frames = len(self.frame_ts_arr)
        self.Qmat = torch.from_numpy(np.load(str(self.sequence_dir / "Qmat.npy")))
        len_seq = int(self.trigger_ts_arr.shape[0])

        self.track_3d = torch.from_numpy(np.load(str(self.sequence_dir / "gt_track.npy"))[:, -len_seq:])
        self.track_2d_left = torch.from_numpy(np.load(str(self.sequence_dir / "gt_2d_680.npy"))[-len_seq:])
        self.track_2d_right = torch.from_numpy(np.load(str(self.sequence_dir / "gt_2d_688.npy"))[-len_seq:])

        self.track_2d_right[:, :, 1] = self.track_2d_left[:, :, 1]

        self.u_centers_left_init = torch.tensor(self.track_2d_left[0])
        self.u_centers_right_init = torch.tensor(self.track_2d_right[0])

        self.channels_in_per_patch = 10

        self.current_idx = 0
        self.t_init = self.trigger_ts_arr[0] * 1e-6
        self.t_end = self.trigger_ts_arr[-1] * 1e-6
        self.t_now = self.t_init
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt)) - 1
        self.n_feat = self.track_2d_left.shape[1]

        self.read_num_feat = np.arange(self.n_feat)

        self.u_centers_left = self.u_centers_left_init[self.read_num_feat]
        self.u_centers_right = self.u_centers_right_init[self.read_num_feat]

        self.frame_first = cv2.imread(str(self.frame_dir / (f"{0}".zfill(4) + ".png")),cv2.IMREAD_GRAYSCALE)
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        self.initialize(u_centers=self.u_centers_left)

    def __len__(self):
        return 1

    def initialize(self, u_centers=None):
        self.initialize_keypoints(u_centers=u_centers)
        self.initialize_reference_patches()

    def initialize_keypoints(self, u_centers):
        self.u_centers = u_centers
        self.u_centers = self.u_centers.to(dtype=torch.float32)
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]
        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

    def initialize_reference_patches(self):
        ref_input = (torch.from_numpy(self.frame_first.astype(np.float32) / 255).unsqueeze(0).unsqueeze(0))
        self.x_ref = self.get_patches(ref_input)

    def reprojectImageTo3D_ph(self, disp, pos):
        Qr = torch.cat([pos.T, torch.ones_like(pos.T)])
        Q = self.Qmat
        Qr[2] = disp
        xyzw = (Q @ Qr)
        xyz = (xyzw[:3] * torch.reciprocal(xyzw[3])).T.reshape(pos.shape[0], 3)
        return xyz

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)

    def get_patches(self, f):
        if f.device != self.device:
            self.device = f.device
            self.move_centers()
        return extract_glimpse(f.repeat(self.u_centers.size(0), 1, 1, 1), (self.patch_size, self.patch_size), self.u_centers.detach()+0.5, mode="nearest")

    def __getitem__(self, idx):
        read_num_feat = self.read_num_feat
        data = {}
        target = {}

        # gt_track_2d_l = []
        # gt_track_2d_r = []
        # gt_disp = []
        gt_track_3d = []

        seq_len_read = self.n_events

        op_tl = torch.zeros((seq_len_read, self.channels_in_per_patch, self.height, self.width))
        op_tr = torch.zeros((seq_len_read, self.channels_in_per_patch, self.height, self.width))

        time_surface_left, time_surface_right = generate_subseq(str(self.sequence_dir))

        for unroll in range(seq_len_read):
            cur_idx = unroll

            input_ep_tl = np.transpose(time_surface_left[cur_idx+1], (2, 0, 1))
            input_ep_tl = torch.from_numpy(input_ep_tl)
            op_tl[unroll] = input_ep_tl

            input_ef_tr = np.transpose(time_surface_right[cur_idx+1], (2, 0, 1))
            input_ef_tr = torch.from_numpy(input_ef_tr)
            op_tr[unroll] = input_ef_tr

            # gt_track_2d_r.append(self.track_2d_right[cur_idx+1, read_num_feat])
            # gt_track_2d_l.append(self.track_2d_left[cur_idx+1, read_num_feat])
            # gt_disp.append(self.track_2d_left[cur_idx+1, read_num_feat, 0] - self.track_2d_right[cur_idx+1, read_num_feat, 0])
            gt_track_3d.append(self.track_3d[read_num_feat, cur_idx+1])

        # gt_disp = torch.stack(gt_disp, dim=1)
        # gt_track_2d_l = torch.stack(gt_track_2d_l, dim=1)
        # gt_track_2d_r = torch.stack(gt_track_2d_r, dim=1)
        gt_track_3d = torch.stack(gt_track_3d, dim=1)

        u_centers = self.track_2d_left[0, read_num_feat, :]
        # u_centers_r = self.track_2d_right[0, read_num_feat, :]

        data['ev_frame_left'] = op_tl
        data['ev_frame_right'] = op_tr
        # target['disp'] = gt_disp

        data['ref_img'] = self.x_ref
        data['u_centers_l'] = u_centers
        # data['u_centers_r'] = u_centers_r
        # target['track_l'] = gt_track_2d_l
        # target['track_r'] = gt_track_2d_r
        target['track_3d'] = gt_track_3d

        return self.trigger_ts_arr[:int(seq_len_read)], data, target


def getDataset(data_folder='/data/lisiqi/event_3d_raw/', train=True):
    dataset_list = []

    if train:
        with open(os.path.join(data_folder, 'train.txt'), 'r') as f:
            lines = f.readlines()
    else:
        with open(os.path.join(data_folder, 'test.txt'), 'r') as f:
            lines = f.readlines()

    for l in lines:
        f1, seq_num = os.path.split(l.strip('\n'))
        root, subdir = os.path.split(f1)
        subdataset = RecurrentStereoSubseq(root_dir=data_folder, sequence_name=f'{subdir}/{seq_num}')
        dataset_list.append(subdataset)
        print(f"Loading...\tseq_name: {subdir}\tseq_idx: {seq_num}\tnum_feat:{subdataset.n_feat}")
    return dataset_list