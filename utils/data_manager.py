import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import (
    iCIFAR100,
    iIMAGENET_R,
    iIMAGENET_A,
    iCUB,
    iDomainNet,
    iIDomainNet,
    iCIFAR10,
    iMULTI,
)


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        # For multi-dataset mode, use increments provided by the underlying
        # iMULTI implementation (computed per-dataset and per-task) instead of
        # deriving them from init_cls / increment.
        if dataset_name.lower() == "multi_dataset":
            if hasattr(self, "_multi_increments"):
                self._increments = self._multi_increments
            else:
                raise ValueError("Multi-dataset requires precomputed increments.")
            return

        # Special handling for iDomainNet when a tasks-per-domain setting is
        # provided. This approximates DC-LoRA's idomainnet_tasks_per_domain
        # semantics by:
        #   - Creating 6 * tasks_per_domain incremental tasks.
        #   - Assigning each class to exactly one such task.
        #   - Associating each task with a single domain so that
        #     DataManager.get_dataset can later filter samples by both
        #     class and domain.
        #
        # Note: unlike DC-LoRA, where each class appears in multiple
        # domain-specific tasks, here each class is still seen only once
        # overall (class-incremental constraint), but each task operates on
        # a single domain.
        if dataset_name.lower() == "idomainnet" and args is not None:
            tpd = args.get("idomainnet_tasks_per_domain", None)
            if tpd is not None:
                tpd = int(tpd)
                if tpd <= 0:
                    raise ValueError("idomainnet_tasks_per_domain must be positive.")

                num_classes = len(self._class_order)
                num_tasks = 6 * tpd
                if num_tasks > num_classes:
                    raise ValueError(
                        "iDomainNet: total tasks (6 * idomainnet_tasks_per_domain) "
                        "exceeds number of classes ({}).".format(num_classes)
                    )

                base = num_classes // num_tasks
                remainder = num_classes - base * num_tasks
                increments = [base] * num_tasks
                # Distribute any remainder classes over the first few tasks.
                for i in range(remainder):
                    increments[i] += 1

                self._increments = increments

                # Build per-task domain assignment in domain-first order,
                # mirroring DC-LoRA's IDOMAINNET_DOMAIN_ORDER.
                domains = iIDomainNet.IDOMAINNET_DOMAIN_ORDER
                task_domains = []
                for dom in domains:
                    for _ in range(tpd):
                        task_domains.append(dom)
                # Truncate in case num_tasks < len(task_domains).
                self._task_domains = task_domains[:num_tasks]

                # Map each class index to its owning task, using the
                # computed increments. This will later allow get_dataset
                # to infer the task (and hence domain) from the class
                # indices passed in.
                class_to_task = np.empty(num_classes, dtype=int)
                task_class_bounds = []
                start = 0
                for task_idx, size in enumerate(increments):
                    end = min(start + size, num_classes)
                    task_class_bounds.append((start, end))
                    class_to_task[start:end] = task_idx
                    start = end

                self._class_to_task = class_to_task
                self._task_class_bounds = task_class_bounds

                logging.info(
                    "iDomainNet: using idomainnet_tasks_per_domain=%d -> %d tasks, "
                    "increments=%s, task_domains=%s",
                    tpd,
                    num_tasks,
                    self._increments,
                    self._task_domains,
                )
                return

        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        # For iDomainNet with an explicit tasks-per-domain setting, delegate to
        # a domain-aware dataset constructor that filters by both class and
        # domain.
        if (
            self.dataset_name.lower() == "idomainnet"
            and self.args is not None
            and self.args.get("idomainnet_tasks_per_domain", None) is not None
        ):
            return self._get_idomainnet_domain_dataset(
                indices, source, mode, appendent=appendent, ret_data=ret_data
            )

        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])

        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Optional domain annotations (for datasets like iIDomainNet).
        if hasattr(idata, "train_domains") and hasattr(idata, "test_domains"):
            self._train_domains = idata.train_domains
            self._test_domains = idata.test_domains
        else:
            self._train_domains = None
            self._test_domains = None

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

        # For multi-dataset mode, keep track of per-task increments computed by
        # the underlying iData implementation (iMULTI).
        if hasattr(idata, "increments"):
            self._multi_increments = idata.increments

    def _get_idomainnet_domain_dataset(
        self, indices, source, mode, appendent=None, ret_data=False
    ):
        """Construct an iDomainNet dataset filtered by both class and domain.

        This is used only when idomainnet_tasks_per_domain is specified.
        Each incremental task is mapped to a single domain according to
        self._task_domains; given the class indices for that task, we
        infer the task id and corresponding domain, then filter samples
        accordingly.
        """

        if source == "train":
            x, y, d = self._train_data, self._train_targets, self._train_domains
        elif source == "test":
            x, y, d = self._test_data, self._test_targets, self._test_domains
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if d is None:
            raise ValueError(
                "iDomainNet domain-aware loader requires domain annotations, "
                "but none were found."
            )

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        indices = list(indices)

        # If no explicit class indices are provided, fall back to the
        # standard behavior (used e.g. for exemplar construction).
        if not indices:
            data, targets = x, y
        else:
            # Infer task id (and thus domain) from the class indices.
            task_ids = []
            for idx in indices:
                if not hasattr(self, "_class_to_task") or not hasattr(
                    self, "_task_domains"
                ):
                    raise ValueError(
                        "iDomainNet domain-aware mode requires _class_to_task and "
                        "_task_domains to be initialized."
                    )

                if idx < 0 or idx >= len(self._class_to_task):
                    raise ValueError(
                        "Class index {} outside valid range [0, {}).".format(
                            idx, len(self._class_to_task)
                        )
                    )

                task_idx = int(self._class_to_task[idx])
                task_ids.append(task_idx)

            domains = [self._task_domains[task_idx] for task_idx in task_ids]
            unique_domains = list(set(domains))

            indices_arr = np.array(indices, dtype=int)
            class_mask = np.isin(y, indices_arr)
            domain_mask = np.zeros_like(d, dtype=bool)
            for domain in unique_domains:
                domain_mask = np.logical_or(domain_mask, d == domain)
            mask = np.logical_and(class_mask, domain_mask)
            data = x[mask]
            targets = y[mask]

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data = (
                np.concatenate([data, appendent_data]) if data.size else appendent_data
            )
            targets = (
                np.concatenate([targets, appendent_targets])
                if targets.size
                else appendent_targets
            )

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar100":
        return iCIFAR100(args)
    elif name == "cifar10":
        return iCIFAR10(args)
    elif name == "imagenet_r":
        return iIMAGENET_R(args)
    elif name == "domainnet":
        return iDomainNet(args)
    elif name == "idomainnet":
        return iIDomainNet(args)
    elif name == "imagenet_a":
        return iIMAGENET_A(args)
    elif name == "cub":
        return iCUB(args)
    elif name == "multi_dataset":
        return iMULTI(args)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
