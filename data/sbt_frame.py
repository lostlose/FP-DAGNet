import os
import random
import numpy as np
import tqdm
from prophesee import dat_events_tools, npy_events_tools
from numpy.lib import recfunctions as rfn
import argparse

class Gen1(object):
    def __init__(self, root, object_classes, height, width, nr_events_window=-1, augmentation=False, mode='train',
                 ms_per_frame=10, frame_per_sequence=5, T=5, shuffle=True, sbt_method='before'):
        """
        Creates an iterator over the Gen1 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param nr_events_window: number of events in a sliding window histogram, -1 corresponds to all events
        :param augmentation: flip, shift and random window start for training
        :param mode: 'train', 'test' or 'val'
        """

        file_dir = os.path.join('detection_dataset_duration_60s_ratio_1.0', mode)
        self.files = os.listdir(os.path.join(root, file_dir))
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files
                      if time_seq_name[-3:] == 'npy']

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.window_time = ms_per_frame * 1000 * frame_per_sequence * T
        if nr_events_window == -1:
            self.nr_events_window = 250000
        else:
            self.nr_events_window = nr_events_window

        self.sbt_method = sbt_method
        self.max_nr_bbox = 15

        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['Car', "Pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes

        self.sequence_start = []
        self.sequence_time_start = []
        self.labels = []
        self.events_num = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files, self.sequence_start))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.sequence_start = zip(*zipped_lists)

    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
            unique indices file.
        """
        file_name_bbox_id = []
        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(self.files):
            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            event_file = os.path.join(self.root, file_name + '_td.dat')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            labels = np.stack(
                [dat_bbox['t'], dat_bbox['x'], dat_bbox['y'], dat_bbox['w'], dat_bbox['h'], dat_bbox['class_id']],
                axis=1)
            if len(self.labels) == 0:
                self.labels = labels
            else:
                self.labels = np.concatenate((self.labels, labels), axis=0)

            unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
            for unique_time in unique_ts:
                sequence_start_end = self.searchEventSequence(event_file, unique_time, time_interval=self.window_time)
                self.sequence_start.append(sequence_start_end)
                self.events_num.append(sequence_start_end[1] - sequence_start_end[0])
                if self.sbt_method == 'mid':
                    self.sequence_time_start.append(unique_time - self.window_time // 2 + 1)
                else:
                    self.sequence_time_start.append(unique_time - self.window_time + 1)

            file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]
            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id

    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        bbox_file = os.path.join(self.root, self.files[idx][0] + '_bbox.npy')
        event_file = os.path.join(self.root, self.files[idx][0] + '_td.dat')

        # Bounding Box
        f_bbox = open(bbox_file, "rb")
        # dat_bbox types (v_type):
        # [('ts', 'uint64'), ('x', 'float32'), ('y', 'float32'), ('w', 'float32'), ('h', 'float32'), (
        # 'class_id', 'uint8'), ('confidence', 'float32'), ('track_id', 'uint32')]
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        bbox_time_idx = self.files[idx][1]

        # Get bounding boxes at current timestep
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = dat_bbox[v_type[0][0]].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx + 1]

        bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]

        # Required Information ['t', 'x', 'y', 'w', 'h', 'class_id']
        np_bbox = rfn.structured_to_unstructured(bboxes)[:, [5, 1, 2, 3, 4]]
        np_bbox = self.cropToFrame(np_bbox)

        const_size_bbox = np.zeros([np_bbox.shape[0], 5])
        const_size_bbox[:np_bbox.shape[0], :] = np_bbox
        sequence_start_end = self.searchEventSequence(event_file, unique_ts[bbox_time_idx],
                                                      time_interval=self.window_time)
        # Events
        events = self.readEventFile(event_file, sequence_start_end, nr_window_events=self.nr_events_window)
        frame = self.sbt_frame(events, self.sequence_time_start[idx], self.height, self.width, self.ms_per_frame,
                               self.frame_per_sequence, self.T)
        frame = frame.reshape(-1, self.height, self.width)

        return events, const_size_bbox.astype(np.int16), frame.astype(np.int8)

    def sbt_frame(self, events, start_time, height, width, ms_per_frame=10, frame_per_sequence=5, T=5):
        final_frame = np.zeros((T, frame_per_sequence, height, width))
        num_events = events.shape[0]
        for i in range(num_events):
            total_index = (events[i, 2] - start_time) // (ms_per_frame * 1000)
            frame_index = int(total_index % frame_per_sequence)
            sequence_index = int(total_index // frame_per_sequence)
            # print(total_index, sequence_index, frame_index, events[i, 1], events[i, 0])
            final_frame[sequence_index, frame_index, events[i, 1], events[i, 0]] += events[i, 3]
        return np.sign(final_frame)

    def searchEventSequence(self, event_file, bbox_time, time_interval=250000):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        if self.sbt_method == 'mid':
            start_time = max(0, bbox_time - time_interval // 2 + 1)
            end_time = min(59999999, bbox_time + time_interval // 2)
        else:
            start_time = max(0, bbox_time - time_interval + 1)
            end_time = bbox_time
        nr_events = dat_events_tools.count_events(event_file)
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        low = 0
        high = nr_events
        file_handle.seek(ev_start + low * ev_size)
        buffer = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=nr_events)["ts"]
        start_position = np.searchsorted(buffer, start_time, side='left')
        end_position = np.searchsorted(buffer, end_time, side='right')

        file_handle.close()
        # we now know that it is between low and high
        return start_position, end_position

    def readEventFile(self, event_file, file_position, nr_window_events=250000):
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        file_handle.seek(ev_start + file_position[0] * ev_size)
        dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')],
                                count=file_position[1] - file_position[0])
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        y = np.right_shift(
            np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        p[p == 0] = -1
        events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)

        return events_np

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        pt1 = [np_bbox[:, 1], np_bbox[:, 2]]
        pt2 = [np_bbox[:, 1] + np_bbox[:, 3], np_bbox[:, 2] + np_bbox[:, 4]]
        pt1[0] = np.clip(pt1[0], 0, self.width - 1)
        pt1[1] = np.clip(pt1[1], 0, self.height - 1)
        pt2[0] = np.clip(pt2[0], 0, self.width - 1)
        pt2[1] = np.clip(pt2[1], 0, self.height - 1)
        np_bbox[:, 1] = pt1[0]
        np_bbox[:, 2] = pt1[1]
        np_bbox[:, 3] = pt2[0] - pt1[0]
        np_bbox[:, 4] = pt2[1] - pt1[1]

        return np_bbox

parser = argparse.ArgumentParser(description='SBT Frame')
# basic
parser.add_argument('--root', default='/',
                    help='raw data root')
parser.add_argument('-t', '--time_steps', default=3, type=int,
                    help='snn time steps')
parser.add_argument('-tf', '--time_per_frame', default=3, type=int,
                    help='sbt ms per frame')
parser.add_argument('-fs', '--frame_per_stack', default=20, type=int,
                    help='frames per stack')
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = Gen1(args.root, object_classes='all', height=240, width=304, augmentation=False, mode='train',
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, shuffle=False, sbt_method='before')
    val_dataset = Gen1(args.root, object_classes='all', height=240, width=304, augmentation=False, mode='val',
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, shuffle=False, sbt_method='before')
    test_dataset = Gen1(args.root, object_classes='all', height=240, width=304, augmentation=False, mode='test',
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, shuffle=False, sbt_method='before')
    save_dir = os.path.join(args.root, 'gen1/sbt_{}ms_{}frame_{}stack_{}'.format(args.time_per_frame, args.frame_per_stack, args.time_steps, 'before'))
    train_save_dir = os.path.join(save_dir, 'train')
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    val_save_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)
    test_save_dir = os.path.join(save_dir, 'test')
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    print('start save sbt trainset frame and label')
    pbar = tqdm.tqdm(total=len(train_dataset), unit='File', unit_scale=True)
    for i in range(len(train_dataset)):
        _, label, frame = train_dataset[i]
        frame_file_path = os.path.join(train_save_dir, 'sample{}_frame.npy'.format(i))
        label_file_path = os.path.join(train_save_dir, 'sample{}_label.npy'.format(i))
        np.save(frame_file_path, frame)
        np.save(label_file_path, label)
        pbar.update(1)
    pbar.close()
    pbar = tqdm.tqdm(total=len(val_dataset), unit='File', unit_scale=True)
    for i in range(len(val_dataset)):
        _, label, frame = val_dataset[i]
        frame_file_path = os.path.join(val_save_dir, 'sample{}_frame.npy'.format(i))
        label_file_path = os.path.join(val_save_dir, 'sample{}_label.npy'.format(i))
        np.save(frame_file_path, frame)
        np.save(label_file_path, label)
        pbar.update(1)
    pbar.close()
    pbar = tqdm.tqdm(total=len(test_dataset), unit='File', unit_scale=True)
    for i in range(len(test_dataset)):
        _, label, frame = test_dataset[i]
        frame_file_path = os.path.join(test_save_dir, 'sample{}_frame.npy'.format(i))
        label_file_path = os.path.join(test_save_dir, 'sample{}_label.npy'.format(i))
        np.save(frame_file_path, frame)
        np.save(label_file_path, label)
        pbar.update(1)
    pbar.close()



