import json
import logging
import multiprocessing as mp

import cv2
import ffmpeg
import numpy as np
import torch
from munch import Munch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from exp import ex
from model import get_extractor

from .utils import (FastLoader, load_by_segment, save_by_segment,
                    serial_load_by_segment, serial_save_by_segment)


"""
Directory structure of the PanoNut360 dataset (default expectation)
├── splits/{train,val,test}.txt        # video identifiers for each split
├── videos/{split}/{video_id}.mp4      # omnidirectional video files
├── saliency/{video_id}/{frame}.npy    # per-frame saliency maps (optional split sub-folders)
└── viewport/{video_id}.npy            # viewport trajectories (optional)

The folder and extension names can be overridden via the dataset
configuration (see `configs/dataset/panonut360.json`).
"""


def _as_list(value, default=None):
    if value is None:
        return list(default) if default is not None else []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


class PanoNut360(Dataset):
    @ex.capture()
    def __init__(self, data_path, cache_path, rebuild_cache, num_workers,
                 clip_length, extractor_name, feature_norm, model_config,
                 eval_res, mode, dataset_args):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.split = mode

        model_config = Munch(model_config)
        dataset_args = Munch(dataset_args) if dataset_args is not None else Munch()

        # Dataset specific configuration ------------------------------------------------------
        self.dataset_root = data_path / dataset_args.get('root_dir', 'panonut360')
        self.name = dataset_args.get('dataset_name', 'PanoNut360')

        self.split_files = dataset_args.get('split_files', {})
        self.video_dir = dataset_args.get('video_dir', 'videos')
        self.video_dir_has_split = dataset_args.get('video_dir_has_split', True)
        self.video_ext = _as_list(dataset_args.get('video_ext', ['.mp4', '.mkv', '.mov']))

        self.gt_dir = dataset_args.get('gt_dir', 'saliency')
        self.gt_dir_has_split = dataset_args.get('gt_dir_has_split', False)
        self.gt_ext = _as_list(dataset_args.get('gt_ext', ['.npy', '.npz', '.png', '.jpg']))

        self.viewport_dir = dataset_args.get('viewport_dir', None)
        self.viewport_ext = _as_list(dataset_args.get('viewport_ext', ['.npy', '.npz', '.json']))

        self.viewport_requires_split = dataset_args.get('viewport_dir_has_split', False)

        # ------------------------------------------------------------------------------------
        self.cache_path = cache_path
        self.data_path = data_path
        self.rebuild_cache = rebuild_cache
        self.num_workers = num_workers

        self.input_resolution = model_config.input_resolution
        self.model_resolution = model_config.model_resolution
        self.patch_size = model_config.patch_size
        self.train_type = model_config.train_type
        self.clip_length = clip_length

        self.eval_res = (eval_res * 2, eval_res)

        self.extractor_name = extractor_name
        self.feature_norm = feature_norm

        split_file = self._resolve_split_file(mode)
        with open(split_file) as f:
            self.video_list = [line.strip() for line in f.readlines() if line.strip()]

        self.clips = self.get_video()

    # ----------------------------------------------------------------------------------
    # Dataset helpers
    # ----------------------------------------------------------------------------------
    def _resolve_split_file(self, mode):
        if mode in self.split_files:
            split_file = self.split_files[mode]
        else:
            split_file = f'{mode}.txt'

        split_path = self.dataset_root / split_file
        if not split_path.is_file():
            raise FileNotFoundError(f'Split file for mode "{mode}" not found: {split_path}')
        return split_path

    def _resolve_data_dir(self, subdir, has_split):
        base = self.dataset_root / subdir
        if has_split:
            base = base / self.split
        return base

    def _match_file(self, directory, stem, extensions):
        for ext in extensions:
            candidate = directory / f'{stem}{ext}'
            if candidate.exists():
                return candidate
        return None

    def _resolve_video_path(self, video_id):
        video_dir = self._resolve_data_dir(self.video_dir, self.video_dir_has_split)
        video_path = self._match_file(video_dir, video_id, self.video_ext)
        if video_path is None:
            raise FileNotFoundError(f'Video file for id "{video_id}" was not found under {video_dir}')
        return video_path

    def _resolve_gt_dir(self, video_id):
        gt_dir = self._resolve_data_dir(self.gt_dir, self.gt_dir_has_split)
        gt_candidate = gt_dir / video_id
        if gt_candidate.is_dir():
            return gt_candidate
        matched = self._match_file(gt_dir, video_id, self.gt_ext)
        if matched is not None:
            return matched
        if not gt_dir.exists():
            self.logger.warning('Ground-truth directory %s does not exist. Saliency supervision will be empty.', gt_dir)
        return None

    def _resolve_viewport_path(self, video_id):
        if self.viewport_dir is None:
            return None
        viewport_dir = self._resolve_data_dir(self.viewport_dir, self.viewport_requires_split)
        matched = self._match_file(viewport_dir, video_id, self.viewport_ext)
        if matched is None:
            self.logger.debug('Viewport annotation for %s not found in %s', video_id, viewport_dir)
        return matched

    # ----------------------------------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------------------------------
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]

        data = {
            'frame': clip['frame'],
            'cls': clip['cls'],
        }

        label = {
            'mask': torch.where(torch.norm(clip['frame'], dim=(-2, -1)) > 1e-6, 1., 0.)
        }

        if len(clip['frame'].size()) > 3:
            label['mask'] = torch.where(torch.norm(clip['frame'], dim=(-2, -1)).sum(-1) > 1e-5, 1., 0.)

        for key in ['gt', 'video', 'viewport']:
            if key in clip:
                label[key] = clip[key]

        meta = {
            'width': clip['width'],
            'height': clip['height'],
            'video_id': clip['video_id'],
            'segment_id': clip['segment_id']
        }

        return data, label, meta

    # ----------------------------------------------------------------------------------
    # Video / annotation loading
    # ----------------------------------------------------------------------------------
    def get_video(self):
        cache_file = (
            f"{self.name.lower()}_{self.split}_f{self.clip_length}_"
            f"r{self.input_resolution}_{self.extractor_name}_"
            f"{self.model_resolution}_p{self.patch_size}_{self.train_type}.pkl"
        )
        cache_file = self.cache_path / cache_file

        if self.rebuild_cache and len(list(cache_file.parent.glob(f'{cache_file.stem}*'))) > 0:
            for part_file in cache_file.parent.glob(f'{cache_file.stem}*'):
                part_file.unlink()

        if len(list(cache_file.parent.glob(f'{cache_file.stem}*'))) > 0:
            if self.input_resolution > 224 or self.patch_size == 8:
                clip_list = serial_load_by_segment(load_dir=cache_file)
            else:
                clip_list = load_by_segment(load_dir=cache_file)
        else:
            video_items = {}
            with mp.Pool(self.num_workers) as pool:
                iterator = [self._resolve_video_path(x) for x in self.video_list]
                for video_id, frames, width, height in tqdm(
                        pool.imap_unordered(self._get_video, iterator),
                        total=len(iterator), desc=f'{self.name} VID'):
                    video_items[video_id] = {
                        'frame': frames,
                        'width': width,
                        'height': height,
                    }

            if self.split != 'train':
                self._attach_annotations(video_items)

            clip_list = self._build_clips(video_items)

            if self.input_resolution > 224 or self.patch_size == 8:
                serial_save_by_segment(data=clip_list, save_dir=cache_file)
            else:
                save_by_segment(data=clip_list, save_dir=cache_file)

        return clip_list

    def _attach_annotations(self, video_items):
        gt_root = self._resolve_data_dir(self.gt_dir, self.gt_dir_has_split)

        if not gt_root.exists():
            self.logger.warning('Saliency ground truth directory %s is missing. Skipping GT loading.', gt_root)
            return

        for video_id in self.video_list:
            if video_id not in video_items:
                continue

            gt_entry = self._resolve_gt_dir(video_id)
            if gt_entry is None:
                continue

            gt_tensor = self._load_saliency(gt_entry)
            if gt_tensor is not None:
                video_items[video_id]['gt'] = gt_tensor

            viewport_path = self._resolve_viewport_path(video_id)
            if viewport_path is not None:
                viewport_tensor = self._load_viewport(viewport_path)
                if viewport_tensor is not None:
                    video_items[video_id]['viewport'] = viewport_tensor

    def _build_clips(self, video_items):
        clip_list = []

        loader = DataLoader(FastLoader(video_items))
        feature_extractor = get_extractor().cuda()
        feature_extractor.eval()

        for k, v in tqdm(loader, desc=f'Feature extraction ({self.extractor_name})'):
            v = {key: value[0] for key, value in v.items()}
            video_len = v['frame'].size()[0]

            for i in range(video_len // self.clip_length + 1):
                if i * self.clip_length >= video_len:
                    continue

                frame = v['frame'][i * self.clip_length:(i + 1) * self.clip_length]

                if self.feature_norm:
                    if self.train_type in ['dino']:
                        frame = frame / 255.
                        frame -= torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)).double()
                        frame /= torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)).double()
                    else:
                        frame = ((frame / 255.) - 0.5) / 0.5

                clip = {
                    'video_id': k,
                    'segment_id': i,
                    'frame': feature_extractor(frame.cuda()).detach().cpu(),
                    'width': v['width'],
                    'height': v['height']
                }

                if clip['frame'].size()[0] != self.clip_length:
                    zero_vid = torch.zeros_like(clip['frame'][0])
                    if len(clip['frame'].size()) == 3:
                        zero_vid = zero_vid.repeat(self.clip_length - clip['frame'].size()[0], 1, 1)
                    else:
                        zero_vid = zero_vid.repeat(self.clip_length - clip['frame'].size()[0], 1, 1, 1)
                    clip['frame'] = torch.cat((clip['frame'], zero_vid), 0)

                if self.split != 'train':
                    if 'frame' in v:
                        raw_video = v['frame'][i * self.clip_length:(i + 1) * self.clip_length].numpy()
                        raw_video = [np.transpose(raw_video[j], (1, 2, 0)) for j in range(raw_video.shape[0])]
                        raw_video = np.array([cv2.resize(x, self.eval_res, cv2.INTER_LANCZOS4) for x in raw_video])
                        raw_video = torch.from_numpy(np.transpose(raw_video, (0, 3, 1, 2)))
                        if raw_video.size()[0] != self.clip_length:
                            zero_vid = torch.zeros_like(raw_video[0])
                            zero_vid = zero_vid.repeat(self.clip_length - raw_video.size()[0], 1, 1, 1)
                            clip['video'] = torch.cat((raw_video, zero_vid), 0)
                        else:
                            clip['video'] = raw_video

                    if 'gt' in v:
                        gt = v['gt'][i * self.clip_length:(i + 1) * self.clip_length]
                        if gt.size()[0] != self.clip_length:
                            zero_vid = torch.zeros_like(gt[0])
                            zero_vid = zero_vid.repeat(self.clip_length - gt.size()[0], 1, 1)
                            clip['gt'] = torch.cat((gt, zero_vid), 0)
                        else:
                            clip['gt'] = gt

                    if 'viewport' in v:
                        viewport = v['viewport'][i * self.clip_length:(i + 1) * self.clip_length]
                        if viewport.size()[0] != self.clip_length:
                            zero_vid = torch.zeros_like(viewport[0])
                            zero_vid = zero_vid.repeat(self.clip_length - viewport.size()[0], 1, 1)
                            clip['viewport'] = torch.cat((viewport, zero_vid), 0)
                        else:
                            clip['viewport'] = viewport

                if len(clip['frame'].size()) == 3:
                    clip['cls'] = clip['frame'][:, 0]
                    clip['frame'] = clip['frame'][:, 1:]
                else:
                    clip['cls'] = clip['frame'][:, 0]

                clip_list.append(clip)

        return clip_list

    # ----------------------------------------------------------------------------------
    # File loading utilities
    # ----------------------------------------------------------------------------------
    def _get_video(self, vid_path):
        video_id = vid_path.stem
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])

        width = self.input_resolution * 2
        height = self.input_resolution

        cmd = ffmpeg.input(str(vid_path)).filter('scale', width, height)
        out, _ = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)

        video = np.frombuffer(out, np.uint8)
        video = video.reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)

        return video_id, video, orig_width, orig_height

    def _load_saliency(self, gt_entry):
        if gt_entry is None:
            return None

        if gt_entry.is_dir():
            frame_paths = sorted(gt_entry.glob('*'))
            if len(frame_paths) <= 0:
                return None
            tensors = [self._load_saliency_frame(frame_path) for frame_path in frame_paths]
        else:
            tensors = self._load_saliency_file(gt_entry)

        if len(tensors) <= 0:
            return None

        return torch.stack(tensors)

    def _load_saliency_file(self, path):
        ext = path.suffix.lower()
        if ext in ['.npy', '.npz']:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data[list(data.files)[0]]
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f'Unsupported saliency file extension: {path}')

        if isinstance(data, list):
            data = np.array(data)

        if data.ndim == 2:
            return [self._resize_saliency(data)]
        if data.ndim == 3:
            return [self._resize_saliency(frame) for frame in data]

        raise ValueError(f'Unexpected saliency data shape {data.shape} in {path}')

    def _load_saliency_frame(self, path):
        ext = path.suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        elif ext in ['.npy', '.npz']:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data[list(data.files)[0]]
        else:
            raise ValueError(f'Unsupported saliency file extension: {path}')

        if data.ndim == 3:
            # Allow (H, W, 1)
            data = data.squeeze()

        return self._resize_saliency(data)

    def _resize_saliency(self, array):
        resized = cv2.resize(array, self.eval_res, cv2.INTER_LANCZOS4)
        return torch.from_numpy(resized.astype('float32'))

    def _load_viewport(self, viewport_path):
        ext = viewport_path.suffix.lower()
        if ext in ['.npy', '.npz']:
            data = np.load(viewport_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data[list(data.files)[0]]
        elif ext == '.json':
            with open(viewport_path) as f:
                data = json.load(f)
            data = np.array(data)
        else:
            raise ValueError(f'Unsupported viewport annotation extension: {viewport_path}')

        tensor = torch.from_numpy(data.astype('float32'))
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        return tensor
