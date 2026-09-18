"""CIFAR-10 datasets for the memorization experiments, plus seeding and loader helpers.

Four upstream tasks share these builders:
  - random label   : real CIFAR-10 images, labels drawn once from ``label_seed``
  - random pixel   : uint8 pixels drawn from ``data_seed``, labels from ``label_seed``
  - resample label : random labels redrawn every epoch (CIFAR10RandomLabels.resample_labels)
  - resample pixel : random pixels redrawn every epoch (RandomPixelCIFAR10.resample_images)
Test splits always keep the real CIFAR-10 labels.

There are two transforms because the two datasets enter the pipeline differently: the real-image
sets go through ``cifar_transform`` (ToTensor + Normalize, from a PIL image); the random-pixel set
already yields a float tensor in [0, 1] from its own ``__getitem__`` and only needs
``pixel_transform`` (Normalize). No augmentation in either -- batch order is deliberately the only
source of run-to-run variation (see build_train_loader / make_train_eval_loader).
"""
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from common.config import CIFAR10_MEAN, CIFAR10_STD, DATA_DIR, NUM_CLASSES


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch and pin cuDNN to deterministic, non-benchmarking kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding derived from torch's per-worker base seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cifar_transform() -> T.Compose:
    """PIL image -> normalized tensor, for the real-image datasets (random / resampled labels)."""
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def pixel_transform() -> T.Normalize:
    """Normalize only, for RandomPixelCIFAR10 (its __getitem__ already yields float [0, 1])."""
    return T.Normalize(CIFAR10_MEAN, CIFAR10_STD)


class CIFAR10RandomLabels(Dataset):
    """Real CIFAR-10 images with random training labels.

    Training labels are drawn from ``label_seed`` (fixed random labels). Pass ``label_seed=None``
    to draw a fresh seed from ``os.urandom`` and call :meth:`resample_labels` each epoch for the
    resample-label task; ``current_label_seed`` records whatever seed was actually used. The test
    split keeps the real labels.
    """

    def __init__(self, train, transform=None, label_seed=None, root=None, num_classes=NUM_CLASSES):
        super().__init__()
        root = str(DATA_DIR if root is None else root)
        self.base = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.train = train
        self.num_classes = num_classes
        self.rand_labels = None
        self.current_label_seed = None
        if train:
            self.resample_labels(label_seed=label_seed)

    @staticmethod
    def _random_seed() -> int:
        return int.from_bytes(os.urandom(8), byteorder="little", signed=False)

    def resample_labels(self, label_seed=None):
        """Redraw the random label vector; returns the seed used (None on the test split)."""
        if not self.train:
            return None
        seed = self._random_seed() if label_seed is None else int(label_seed)
        g = torch.Generator()
        g.manual_seed(seed)
        self.rand_labels = torch.randint(0, self.num_classes, (len(self.base),), generator=g)
        self.current_label_seed = seed
        return seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y_true = self.base[idx]
        if self.rand_labels is None:
            return x, y_true
        return x, int(self.rand_labels[idx])


class RandomPixelCIFAR10(Dataset):
    """Uniform random-pixel images (uint8 from ``data_seed``) with random labels (``label_seed``).

    Call :meth:`resample_images` each epoch for the resample-pixel task. The test split keeps real
    CIFAR-10 labels (and gets its own ``data_seed``, conventionally the train seed + 1).
    """

    def __init__(self, train, data_seed, label_seed, size, image_shape=(3, 32, 32),
                 transform=None, root=None, num_classes=NUM_CLASSES):
        super().__init__()
        root = str(DATA_DIR if root is None else root)
        self.train = train
        self.transform = transform
        self.image_shape = image_shape
        self.current_data_seed = None
        self.current_label_seed = int(label_seed) if train else None

        base = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        if size != len(base.targets):
            raise ValueError(f"size must match the CIFAR-10 split size, got {size}, expected {len(base.targets)}")

        if train:
            g = torch.Generator()
            g.manual_seed(int(label_seed))
            self.labels = torch.randint(0, num_classes, (size,), generator=g, dtype=torch.int64)
        else:
            self.labels = torch.tensor(base.targets, dtype=torch.int64)

        self.images = None
        self.resample_images(data_seed)

    def resample_images(self, data_seed):
        """Redraw the whole random-pixel tensor from ``data_seed``; returns the seed used."""
        g = torch.Generator()
        g.manual_seed(int(data_seed))
        self.images = torch.randint(0, 256, (len(self.labels), *self.image_shape),
                                    generator=g, dtype=torch.uint8)
        self.current_data_seed = int(data_seed)
        return self.current_data_seed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.images is None:
            raise RuntimeError("random-pixel images have not been initialized")
        x = self.images[idx].to(torch.float32).div_(255.0)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[idx])


def build_data_seed_schedule(schedule_seed: int, num_epochs: int) -> list[int]:
    """Deterministic per-epoch data seeds for the resample-pixel task (one seed per epoch)."""
    g = torch.Generator()
    g.manual_seed(int(schedule_seed))
    return torch.randint(0, 2 ** 63 - 1, (num_epochs,), generator=g, dtype=torch.int64).tolist()


def build_split_indices(num_samples, d1_size, split_seed):
    """Partition indices [0, num_samples) into (D1, D2) by a ``split_seed``-seeded permutation.

    The random-label map is drawn before this split, so D1 and D2 share it and the split only
    decides which samples each stage sees. Deterministic in ``split_seed``: stage 2 can rebuild
    the same D2 from the seed alone, without reading stage 1's saved indices (Fig 2b).
    """
    if d1_size >= num_samples:
        raise ValueError(f"d1_size must be < num_samples, got {d1_size} vs {num_samples}")
    g = torch.Generator()
    g.manual_seed(int(split_seed))
    perm = torch.randperm(num_samples, generator=g)
    return perm[:d1_size].clone(), perm[d1_size:].clone()


def build_train_loader(dataset, seed, batch_size=128, num_workers=0, pin_memory=False) -> DataLoader:
    """Shuffled training loader whose per-epoch permutation sequence is fixed by ``seed``.

    ``persistent_workers`` is forced False on purpose: persistent workers reuse the iterator across
    epochs and skip the per-epoch permutation draw from the generator, silently changing the shuffle
    order from epoch 2 on. Only the training loop should iterate this loader; any read-only pass over
    the training set must go through :func:`make_train_eval_loader`.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g,
                      persistent_workers=False)


def make_train_eval_loader(train_loader: DataLoader) -> DataLoader:
    """Eval-only view of ``train_loader``'s dataset that cannot disturb the training data order.

    Same dataset and settings but ``shuffle=False`` and no generator, so iterating it draws no
    randomness. Evaluating the training set through the shuffled loader would consume a permutation
    and shift every later epoch by one -- worth ~1 downstream epoch of t0 on this steep transition.
    Cheap to build (holds no state), so call it at the point of use.
    """
    return DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False,
                      num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
                      worker_init_fn=train_loader.worker_init_fn, persistent_workers=False)
