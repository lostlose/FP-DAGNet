import os
import numpy as np
import cv2

class Gen1_sbt(object):
    def __init__(self, root, object_classes, height, width, mode='train', ms_per_frame = 10, frame_per_sequence=5, T = 5, transform=None, sbt_method='mid'):
        self.file_dir = os.path.join(root, 'sbt_{}ms_{}frame_{}stack_{}'.format(ms_per_frame, frame_per_sequence, T, sbt_method), mode)
        self.files = os.listdir(self.file_dir)
        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.transform = transform
        self.sbt_method = sbt_method
        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['car', "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes
    
    def __len__(self):
        return len(self.files) // 2


    def __getitem__(self, idx):
        """
        returns frame and label, loading them from files
        :param idx:
        :return: x,y,  label
        """
        frame = np.load(os.path.join(self.file_dir, 'sample{}_frame.npy'.format(idx))).astype(np.float32)
        label = np.load(os.path.join(self.file_dir, 'sample{}_label.npy'.format(idx))).astype(np.float32)
        if self.transform is not None:
            resized_frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = resized_frame.shape[1], resized_frame.shape[2]
            resized_frame = resized_frame.reshape(self.T, self.frame_per_sequence, h, w)
            return resized_frame, resized_label, label
        return frame, label, label

class Gen1_sbn(object):
    def __init__(self, root, object_classes, height, width, mode='train', events_per_frame = 25000, frame_per_sequence=1, T = 2, transform=None, sbn_method='mid'):
        self.file_dir = os.path.join(root, 'sbn_{}events_{}frame_{}stack_{}'.format(events_per_frame, frame_per_sequence, T, sbn_method), mode)
        self.files = os.listdir(self.file_dir)
        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.events_per_frame = events_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.transform = transform
        self.sbn_method = sbn_method
        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['car', "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes
    
    def __len__(self):
        return len(self.files) // 2


    def __getitem__(self, idx):
        """
        returns frame and label, loading them from files
        :param idx:
        :return: x,y,  label
        """
        frame = np.load(os.path.join(self.file_dir, 'sample{}_frame.npy'.format(idx))).astype(np.float32)
        label = np.load(os.path.join(self.file_dir, 'sample{}_label.npy'.format(idx))).astype(np.float32)
        if self.transform is not None:
            resized_frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = resized_frame.shape[1], resized_frame.shape[2]
            resized_frame = resized_frame.reshape(self.T, self.frame_per_sequence, h, w)
            return resized_frame, resized_label, label
        return frame, label, label

class Resize_frame(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, frame, label):
        frame = frame.transpose(1, 2, 0)
        h_original, w_original = frame.shape[0], frame.shape[1]
        r = min(self.input_size / h_original, self.input_size / w_original)
        h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
        if r > 1:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
        else:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
        h_pad, w_pad =  self.input_size - h_resize, self.input_size - w_resize
        h_pad /= 2
        w_pad /= 2
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
        final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        final_label = np.zeros_like(label)
        final_label[:, 0] = label[:, 0]
        final_label[:, 1:] = np.round(label[:, 1:] * r)
        final_label[:, 1] = np.round(final_label[:, 1] + w_pad)
        final_label[:, 2] = np.round(final_label[:, 2] + h_pad)
        if len(final_frame.shape) == 2:
            final_frame = np.expand_dims(final_frame, axis=-1)
        return final_frame.transpose(2, 0, 1), final_label



