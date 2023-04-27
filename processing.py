import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Force-Aware Interface via Electromyography for Natural VR/AR Interaction')
FLAGS = parser.parse_args()

# Raw data info
emg_fps = 2000
file_ids = list(range(11))
session_ids = list(range(1, 28))
division_lines = []
division_lines += [[[0, 93, 123, 147], [0, 138, 157, 179]]] * 3
division_lines += [[[0, 88, 110, 130], [0, 136, 153, 173]]] * 3
division_lines += [[[0, 92, 115, 144], [0, 136, 153, 173]]] * 3
division_lines += [[[0, 98, 126, 148], [0, 136, 153, 173]]] * 3
division_lines += [[[0, 88, 115, 143], [0, 136, 156, 180]]] * 3
division_lines += [[[0, 97, 126, 159], [0, 136, 156, 180]]] * 3
division_lines += [[[0, 78, 106, 128], [0, 127, 144, 160]]] * 3
division_lines += [[[0, 89, 119, 141], [0, 138, 150, 179]]] * 3
division_lines += [[[0, 94, 117, 140], [0, 137, 160, 181]]] * 3
# Data processing
start_time = 4.0
duration = 30.0
window_length = 256
hop_length = 32
# Dataset generation
num_frames = 32
hop_length_train = 4
hop_length_test = 8


def location2index(x, lines):
    if x <= lines[0]:
        return 1
    elif lines[0] < x <= lines[1]:
        return 2
    elif lines[1] < x <= lines[2]:
        return 3
    elif lines[2] < x <= lines[3]:
        return 4
    else:
        return 5


def process_emg_and_force(args):
    for sid in session_ids:
        for fid in file_ids:
            # EMG data
            emg = np.loadtxt(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.csv".format(fid)), dtype=np.float32, delimiter='\t')
            emg = emg[:, emg[0] != 0.0]  # remove trailing empty entries
            begin = 0
            end = 0
            for timestamp in emg[0]:
                if timestamp < start_time:
                    begin += 1
                else:
                    break
            for timestamp in emg[0][::-1]:
                if timestamp > start_time + duration:
                    end -= 1
                else:
                    break
            emg = emg[1:, begin:end] if end < 0 else emg[1:, begin:]  # remove the starting and ending few frames
            num_frames_original = emg.shape[1] // hop_length - window_length // hop_length + 1
            fps = num_frames_original / float(duration)
            emg = emg.transpose()[:(num_frames_original + window_length // hop_length - 1) * hop_length]
            print("EMG data shape:", emg.shape)
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.npy".format(fid)), emg)
            # Force data
            sensel = np.loadtxt(os.path.join(args.data_path, "Session{:d}".format(sid), "sensel_{:d}.csv".format(fid)), dtype=np.float32, delimiter='\t')
            force = np.zeros(((sensel[:, 1] == 0).sum(), 6), dtype=np.float32)
            current = -1
            for entry in sensel:
                if entry[1] == 0:
                    current += 1
                    force[current, 0] = entry[0]
                    force[current, location2index(entry[3], division_lines[sid - 1][0] if fid < 6 else division_lines[sid - 1][1])] = entry[5]
                else:
                    force[current, location2index(entry[3], division_lines[sid - 1][0] if fid < 6 else division_lines[sid - 1][1])] = entry[5]
            assert current == force.shape[0] - 1
            # Resample force data
            timestamps = np.linspace(start_time + window_length / (2.0 * emg_fps), duration + start_time - window_length / (2.0 * emg_fps), num=num_frames_original, endpoint=True, dtype=np.float32)
            force_downsampled = force[np.abs(force[:, 0].reshape(1, -1) - timestamps.reshape(-1, 1)).argmin(1)]
            force_downsampled[np.abs(force[:, 0].reshape(1, -1) - timestamps.reshape(-1, 1)).min(1) >= (1.0 / fps)] = 0
            force = force_downsampled[:, 1:]
            if fid >= 6:
                force[:, 0] = force[:, 1:].sum(1) / 2.0
            force_class = np.asarray(force >= 1, dtype=np.int64)
            print("Force data shape:", force.shape)
            scale = 1.0e3
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_{:d}.npy".format(fid)), force / scale)
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_{:d}.npy".format(fid)), force_class)


def build_dataset(args):
    emg_train = []
    force_train = []
    force_class_train = []
    for sid in session_ids:
        for fid in file_ids:
            emg = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.npy".format(fid)))
            force = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_{:d}.npy".format(fid)))
            force_class = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_{:d}.npy".format(fid)))
            if sid in args.train_sessions:
                for cid in range(0, force.shape[0] - num_frames + 1, hop_length_train):
                    emg_train.append(emg[cid * hop_length:cid * hop_length + (num_frames + window_length // hop_length - 1) * hop_length])
                    force_train.append(force[cid:cid + num_frames])
                    force_class_train.append(force_class[cid:cid + num_frames])
                print("Training file {:d} from session {:d} done!".format(fid, sid))
            elif sid in args.test_sessions:
                emg_test = []
                force_test = []
                force_class_test = []
                for cid in range(0, force.shape[0] - num_frames + 1, hop_length_test):
                    emg_test.append(emg[cid * hop_length:cid * hop_length + (num_frames + window_length // hop_length - 1) * hop_length])
                    force_test.append(force[cid:cid + num_frames])
                    force_class_test.append(force_class[cid:cid + num_frames])
                emg_test = np.stack(emg_test, axis=0).transpose(0, 2, 1)
                force_test = np.stack(force_test, axis=0).transpose(0, 2, 1)
                force_class_test = np.stack(force_class_test, axis=0).transpose(0, 2, 1)
                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_test_{:d}.npy".format(fid)), emg_test)
                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_test_{:d}.npy".format(fid)), force_test)
                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_test_{:d}.npy".format(fid)), force_class_test)
                print("Evaluation file {:d} from session {:d} done!".format(fid, sid))
    emg_train = np.stack(emg_train, axis=0).transpose(0, 2, 1)
    force_train = np.stack(force_train, axis=0).transpose(0, 2, 1)
    force_class_train = np.stack(force_class_train, axis=0).transpose(0, 2, 1)
    print("Training EMG data shape:", emg_train.shape)
    print("Training force shape:", force_train.shape)
    print("Training force class shape:", force_class_train.shape)
    np.save(os.path.join(args.dataset_path, "emg_train"), emg_train)
    np.save(os.path.join(args.dataset_path, "force_train"), force_train)
    np.save(os.path.join(args.dataset_path, "force_class_train"), force_class_train)


def main(args):
    args.data_path = os.path.join(os.getcwd(), 'Data')
    args.dataset_path = os.path.join(os.getcwd(), 'Dataset')
    if not os.path.exists(args.data_path):
        raise Exception("Data not found!")
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    args.train_sessions = list(range(1, 28, 3)) + list(range(2, 28, 3))
    args.test_sessions = list(range(3, 28, 3))

    process_emg_and_force(args)
    build_dataset(args)


if __name__ == '__main__':
    main(FLAGS)
