import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
if sys.path[0] != '':
    sys.path.insert(0, '')

IMG_H = 260
IMG_W = 346


def first_element_greater_than(values, req_value):
    i = np.searchsorted(values, req_value)
    val = values[i] if i < len(values) else None
    return (i, val)


def generate_subseq(path):
    input_dir = Path(path)

    assert input_dir.exists()

    gt_timestamp = np.load(os.path.join(input_dir, "ts_trigger_crop.npy"))
    image_timestamps = np.load(os.path.join(input_dir, "ts_680.npy"))

    events_680 = np.load(os.path.join(input_dir, "events_680.npy"))
    events_688 = np.load(os.path.join(input_dir, "events_688.npy"))

    ref_end_time = gt_timestamp[-1]

    start_ev_idx_680 = np.searchsorted(events_680[:,0], image_timestamps[0])
    end_ev_idx_680 = np.searchsorted(events_680[:,0], ref_end_time)
    start_ev_idx_688 = np.searchsorted(events_688[:, 0], image_timestamps[0])
    end_ev_idx_688 = np.searchsorted(events_688[:, 0], ref_end_time)

    events_680 = events_680[start_ev_idx_680:end_ev_idx_680]
    events_688 = events_688[start_ev_idx_688:end_ev_idx_688]

    dt_us = 4000
    n_pack = 1

    n_bins = 5
    dt_bin_us = dt_us * n_pack / n_bins

    time_surface_680, time_surface_688 = [], []

    for i, t1 in tqdm(enumerate(np.arange(gt_timestamp[0+n_pack-1], gt_timestamp[-1] + dt_us, dt_us*n_pack)), total=int((gt_timestamp[-1] - gt_timestamp[0]) / (dt_us * n_pack) ), desc="Generating time surfaces...",):
        time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float)
        t0 = t1 - dt_us * n_pack

        for i_bin in range(n_bins):
            t0_bin = t0 + i_bin * dt_bin_us
            t1_bin = t0_bin + dt_bin_us
            time = events_680[:, 0]
            first_idx = np.searchsorted(time, t0_bin, side="left")
            last_idx_p1 = np.searchsorted(time, t1_bin, side="right")

            out = {
                "x": np.rint(np.asarray(events_680[:,1][first_idx:last_idx_p1])).astype(int),
                "y": np.rint(np.asarray(events_680[:,2][first_idx:last_idx_p1])).astype(int),
                "p": np.asarray(events_680[:,3][first_idx:last_idx_p1]),
                "t": time[first_idx:last_idx_p1],}
            n_events = out["x"].shape[0]
            for i in range(n_events):
                time_surface[out["y"][i], out["x"][i], 2 * i_bin + int(out["p"][i])] = (out["t"][i] - t0)
        time_surface_680.append(np.divide(time_surface, dt_us * n_pack))

        time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float)
        for i_bin in range(n_bins):
            t0_bin = t0 + i_bin * dt_bin_us
            t1_bin = t0_bin + dt_bin_us
            time = events_688[:, 0]
            first_idx = np.searchsorted(time, t0_bin, side="left")
            last_idx_p1 = np.searchsorted(time, t1_bin, side="right")
            out = {
                "x": np.rint(np.asarray(events_688[:,1][first_idx:last_idx_p1])).astype(int),
                "y": np.rint(np.asarray(events_688[:,2][first_idx:last_idx_p1])).astype(int),
                "p": np.asarray(events_688[:,3][first_idx:last_idx_p1]),
                "t": time[first_idx:last_idx_p1]}
            n_events = out["x"].shape[0]
            for i in range(n_events):
                time_surface[out["y"][i], out["x"][i], 2 * i_bin + int(out["p"][i]) ] = (out["t"][i] - t0)
        time_surface_688.append(np.divide(time_surface, dt_us * n_pack))

    return time_surface_680, time_surface_688