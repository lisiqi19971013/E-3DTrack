import numpy as np
import os


_width = 346
_height = 260


def metric_3d(pred, gt, err_thr=[50., 100., 200.]):
    assert pred.shape == gt.shape
    track_err = gt - pred
    tracking_error = np.linalg.norm(track_err, axis=2)
    fa = []
    fs = []
    for err in err_thr:
        fa_list = []
        fs_list = []
        for idx in range(gt.shape[0]):
            if (np.argwhere(tracking_error[idx, :] > err)).any():
                idx_end = np.min(np.argwhere(tracking_error[idx, :] > err))
            else:
                idx_end = pred.shape[1]
            fa_list.append(idx_end)
            fs_list.append((tracking_error[idx, :] <= err).sum())
        fa_nz = np.array(fa_list)/gt.shape[1]
        fs_nz = np.array(fs_list)/gt.shape[1]
        fa.append(fa_nz.mean())
        fs.append(fs_nz.mean())

    rmse = tracking_error[tracking_error<5e3].mean() / 1000
    return fa, fs, rmse


if __name__ == '__main__':
    root_path = r'./output/'

    err_thr = [100, 150, 200]
    err_keys = ['fa_'+str(x) for x in err_thr] + ['fs_'+str(x) for x in err_thr] + ['mse']

    seq_dict = {}

    err_avg_total = {}

    for k in err_keys:
        err_avg_total[k] = []

    for seq_name in os.listdir(root_path):
        err_avg_seq = {}
        for k in err_keys:
            err_avg_seq[k] = []

        for slice in os.listdir(os.path.join(root_path, seq_name)):
            path = os.path.join(root_path, seq_name, slice)

            track_pred_3D = np.load(os.path.join(path,'pos_3d_pred.npy'))
            gt_track = np.load(os.path.join(path,'pos_3d_gt.npy'))

            gt_track_3D = np.swapaxes(gt_track, 0, 1)
            track_pred_3D = np.swapaxes(track_pred_3D, 0, 1)
            fa, fs, rmse = metric_3d(track_pred_3D*1e3, gt_track_3D, err_thr)

            for i in range(len(err_thr)):
                err_avg_seq[f'fa_{err_thr[i]}'].append(fa[i])
                err_avg_seq[f'fs_{err_thr[i]}'].append(fs[i])
                err_avg_total[f'fa_{err_thr[i]}'].append(fa[i])
                err_avg_total[f'fs_{err_thr[i]}'].append(fs[i])

            err_avg_seq['mse'].append(rmse)
            err_avg_total['mse'].append(rmse)

        for k, v in err_avg_seq.items():
            err_avg_seq[k] = sum(v) / len(v)
        seq_dict[seq_name] = err_avg_seq

        # break

    for k, v in err_avg_total.items():
        err_avg_total[k] = sum(v) / len(v)

    for k, v in seq_dict.items():
        print('--------------')
        print('seq:', k)
        print(v)
    print('--------------')
    print('Total')
    print(err_avg_total)

    # with open(os.path.join(root_path, 'res.txt'), 'w') as f:
    #     for k, v in seq_dict.items():
    #         f.writelines(f"mse:{v['mse']:.4f}\tfa_100:{v['fa_100']:.4f}\tfa_150:{v['fa_150']:.4f}\tfa_200:{v['fa_200']:.4f}\t"
    #                      f"fs_100:{v['fs_100']:.4f}\tfs_150:{v['fs_150']:.4f}\tfs_200:{v['fs_200']:.4f}\tSeq: {k}\n")
    #     f.writelines('\n')
    #     f.writelines(f"mse:{err_avg_total['mse']:.4f}\tfa_100:{err_avg_total['fa_100']:.4f}\tfa_150:{err_avg_total['fa_150']:.4f}\tfa_200:{err_avg_total['fa_200']:.4f}\t"
    #                  f"fs_100:{err_avg_total['fs_100']:.4f}\tfs_150:{err_avg_total['fs_150']:.4f}\tfs_200:{err_avg_total['fs_200']:.4f}\tTotal")