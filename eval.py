import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.append('..')
from utils.dataset import getDataset
from models.tracker_eval import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    import numpy as np
    import torch

    ckpt_path = './ckpt.pth'

    model = TrackerNetEval(feature_dim=384, hgnn=True)
    model_info = torch.load(ckpt_path)
    model.load_state_dict(model_info["state_dict"])
    model = model.cuda()
    print(f"Loaded from: {ckpt_path}")

    testDataset = getDataset(data_folder='./data', train=False)

    with torch.no_grad():
        model.eval()
        for n, td in enumerate(testDataset):
            DL = DataLoader(td, 1)
            model.reset()
            opFolder = os.path.join('./test/output', td.sequence_name)
            os.makedirs(opFolder, exist_ok=True)

            for i, (ts, data, gt) in enumerate(DL):
                for k, v in data.items():
                    data[k] = v.cuda()
                for k, v in gt.items():
                    gt[k] = v.cuda()

                current_pos_l = data['u_centers_l']
                # current_pos_r = data['u_centers_r']
                ref_patch = data['ref_img']

                pred = None
                pos_l = []
                disp = []
                pos_3d = []
                for unroll in range(data['ev_frame_left'].shape[1]):
                    ev_frame_l = data['ev_frame_left'][:, unroll]
                    ev_frame_r = data['ev_frame_right'][:, unroll]
                    # gt_flow_l = (gt['track_l'][:, :, unroll] - current_pos_l)
                    # gt_disp = gt['disp'][:, :, unroll]

                    flow_l_pred, disp_pred, pred = model(ev_frame_l, ev_frame_r, ref_patch, current_pos_l, None, pred=pred)

                    current_pos_l += flow_l_pred.detach()
                    pos = td.reprojectImageTo3D_ph(disp_pred[0].cpu(), current_pos_l[0].cpu())

                    disp.append(disp_pred)
                    pos_l.append(current_pos_l.clone())
                    pos_3d.append(pos)

                pos_3d = torch.stack(pos_3d)
                np.save(os.path.join(opFolder, 'pos_3d_pred.npy'), np.array(pos_3d.cpu()))
                np.save(os.path.join(opFolder, 'pos_3d_gt.npy'), np.array(gt['track_3d'][0].transpose(1,0).cpu()))
                message = f'Test, Sequence: [{n}]/[{len(testDataset)}]'
                print(message)

