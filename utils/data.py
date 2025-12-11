import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import ipdb
import yaml
from PIL import Image
from shutil import move, rmtree
import torch


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t


class iCUB(iData):
    use_path = True

    train_trsf = [
        transforms.RandomResizedCrop(
            224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [transforms.ToTensor()]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/cub/train/"
        test_dir = "data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    test_trsf = [transforms.Resize(224), transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ]

    class_order = np.arange(10).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(10).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(
            self.args["data_path"], train=True, download=True
        )
        test_dataset = datasets.cifar.CIFAR10(
            self.args["data_path"], train=False, download=True
        )
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(224),
    ]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ]

    # train_trsf = [
    #     transforms.RandomResizedCrop(224, interpolation=3),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=63/255)
    # ]
    # test_trsf = [
    #     transforms.Resize(256, interpolation=3),
    #     transforms.CenterCrop(224),
    # ]

    # common_trsf = [
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        # Set download=False since compute nodes have no internet access
        # Data should be pre-downloaded to data_path
        train_dataset = datasets.cifar.CIFAR100(
            self.args["data_path"], train=True, download=False
        )
        test_dataset = datasets.cifar.CIFAR100(
            self.args["data_path"], train=False, download=False
        )
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iIMAGENET_R(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    common_trsf = [
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # load splits from config file
        # Skip splitting if train/test folders already exist
        train_exists = os.path.exists(os.path.join(self.args["data_path"], "train"))
        test_exists = os.path.exists(os.path.join(self.args["data_path"], "test"))

        if not train_exists or not test_exists:
            self.dataset = datasets.ImageFolder(self.args["data_path"], transform=None)

            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size

            train, val = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )
            train_idx, val_idx = train.indices, val.indices

            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()

        train_data_config = datasets.ImageFolder(
            os.path.join(self.args["data_path"], "train")
        ).samples
        test_data_config = datasets.ImageFolder(
            os.path.join(self.args["data_path"], "test")
        ).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])

    def split(self):
        train_folder = os.path.join(self.args["data_path"], "train")
        test_folder = os.path.join(self.args["data_path"], "test")

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))

        for path in self.train_file_list:
            if "\\" in path:
                path = path.replace("\\", "/")
            src = path
            dst = os.path.join(train_folder, "/".join(path.split("/")[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if "\\" in path:
                path = path.replace("\\", "/")
            src = path
            dst = os.path.join(test_folder, "/".join(path.split("/")[-2:]))
            move(src, dst)

        for c in self.dataset.classes:
            path = os.path.join(self.args["data_path"], c)
            rmtree(path)


class iIMAGENET_A(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(
            224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [transforms.ToTensor()]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet-a/train/"
        test_dir = "data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = [
            "clipart",
            "infograph",
            "painting",
            "quickdraw",
            "real",
            "sketch",
        ]

    def download_data(self):
        # load splits from config file
        train_data_config = yaml.load(
            open("dataloaders/splits/domainnet_train.yaml", "r"), Loader=yaml.Loader
        )
        test_data_config = yaml.load(
            open("dataloaders/splits/domainnet_test.yaml", "r"), Loader=yaml.Loader
        )

        # If data_path is provided in args, treat it as the root of the
        # DomainNet directory containing the six domain folders
        # (clipart/infograph/painting/quickdraw/real/sketch). The YAML
        # files store paths like "data/DomainNet/domain/class/file.jpg";
        # we strip the leading prefix and join the remainder to data_path
        # so that we can reuse the same dataset root as DC-LoRA.
        data_root = os.path.expanduser(self.args.get("data_path", ""))

        def _remap_domainnet_path(p):
            # Only remap when data_root is an absolute path; for relative
            # defaults we keep the original YAML paths for backward
            # compatibility.
            if not data_root or not os.path.isabs(data_root):
                return p

            prefixes = [
                "data/DomainNet/",
                "DomainNet/",
                "data/domainnet/",
                "domainnet/",
            ]

            for pre in prefixes:
                if p.startswith(pre):
                    rel = p[len(pre) :]
                    return os.path.join(data_root, rel)

            # If p is already absolute, keep as-is; otherwise join with data_root
            if os.path.isabs(p):
                return p
            return os.path.join(data_root, p)

        train_paths = [_remap_domainnet_path(p) for p in train_data_config["data"]]
        test_paths = [_remap_domainnet_path(p) for p in test_data_config["data"]]

        self.train_data = np.array(train_paths)
        self.train_targets = np.array(train_data_config["targets"])
        self.test_data = np.array(test_paths)
        self.test_targets = np.array(test_data_config["targets"])


class iIDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        # Project root is four levels above this file:
        # dc_inc/baselines/InfLoRA/utils/data.py -> dc_inc
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        splits_dir = os.path.join(base_dir, "splits", "idomainnet")

        classes_file = os.path.join(splits_dir, "idomainnet_classes.txt")
        train_file = os.path.join(splits_dir, "idomainnet_train.txt")
        test_file = os.path.join(splits_dir, "idomainnet_test.txt")

        with open(classes_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

        class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.class_order = list(range(len(class_names)))

        data_root = os.path.expanduser(self.args.get("data_path", ""))
        if not data_root:
            data_root = "data/domainnet"
        data_root = os.path.expanduser(data_root)

        def _load_split(txt_path):
            paths = []
            labels = []
            with open(txt_path, "r") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    parts = p.split("/")
                    if len(parts) < 3:
                        continue
                    cls_name = parts[1]
                    if cls_name not in class_to_idx:
                        continue
                    label = class_to_idx[cls_name]
                    full_path = os.path.join(data_root, p)
                    paths.append(full_path)
                    labels.append(label)
            return np.array(paths), np.array(labels)

        self.train_data, self.train_targets = _load_split(train_file)
        self.test_data, self.test_targets = _load_split(test_file)


class iMULTI(iData):
    """Multi-dataset incremental setting for CIFAR-100, ImageNet-R, and CUB-200.

    This mirrors the high-level behavior of DC-LoRA's build_multi_dataset_sequence
    by:
      - Parsing dataset_sequence, tasks_per_dataset, and data_roots from args.
      - Loading each dataset's train/test splits.
      - Applying a global class offset per dataset.
      - Concatenating all data and exposing per-task increments via
        self.increments so DataManager can reuse its standard API.
    """

    use_path = False

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ]

    def __init__(self, args):
        self.args = args
        self.class_order = None
        self.increments = []

    def _parse_multi_info(self):
        dataset_seq = self.args.get("dataset_sequence", "")
        tasks_str = self.args.get("tasks_per_dataset", "")
        if (not dataset_seq) or (not tasks_str):
            raise ValueError(
                "dataset_sequence and tasks_per_dataset must be provided for multi_dataset."
            )

        datasets_list = [d.strip().lower() for d in dataset_seq.split(",") if d.strip()]
        tasks_list = [int(t.strip()) for t in tasks_str.split(",") if str(t).strip()]

        roots_raw = self.args.get("data_roots", "").strip()
        if roots_raw:
            roots_list = [r.strip() for r in roots_raw.split(",")]
        else:
            base_root = self.args.get("data_path", "data/")
            roots_list = [base_root] * len(datasets_list)

        if len(datasets_list) != len(tasks_list):
            raise ValueError(
                f"dataset_sequence has {len(datasets_list)} items but tasks_per_dataset has {len(tasks_list)}"
            )
        if len(datasets_list) != len(roots_list):
            raise ValueError(
                f"dataset_sequence has {len(datasets_list)} items but data_roots has {len(roots_list)}"
            )

        return datasets_list, tasks_list, roots_list

    def download_data(self):
        datasets_list, tasks_list, roots_list = self._parse_multi_info()

        train_data_list = []
        train_targets_list = []
        test_data_list = []
        test_targets_list = []
        increments = []
        class_offset = 0

        for ds_name, num_tasks, root in zip(datasets_list, tasks_list, roots_list):
            root = os.path.expanduser(root)

            if ds_name == "cifar100":
                train_ds = datasets.cifar.CIFAR100(root, train=True, download=False)
                test_ds = datasets.cifar.CIFAR100(root, train=False, download=False)

                def _resize_cifar_images(images):
                    resized = []
                    for img in images:
                        pil_img = Image.fromarray(img)
                        pil_img = pil_img.resize((224, 224))
                        resized.append(np.array(pil_img))
                    return np.stack(resized, axis=0)

                train_imgs = _resize_cifar_images(train_ds.data)
                train_labels = np.array(train_ds.targets)
                test_imgs = _resize_cifar_images(test_ds.data)
                test_labels = np.array(test_ds.targets)
                num_classes = 100
            elif ds_name == "imagenet_r":
                train_root = os.path.join(root, "train")
                test_root = os.path.join(root, "test")

                train_folder = datasets.ImageFolder(train_root)
                test_folder = datasets.ImageFolder(test_root)
                train_samples = train_folder.samples
                test_samples = test_folder.samples

                train_imgs = np.array(
                    [jpg_image_to_array(p) for (p, _) in train_samples]
                )
                train_labels = np.array([lab for (_, lab) in train_samples])
                test_imgs = np.array([jpg_image_to_array(p) for (p, _) in test_samples])
                test_labels = np.array([lab for (_, lab) in test_samples])
                num_classes = len(train_folder.classes)
            elif ds_name in ("cub200", "cub"):
                # CUB-200-2011: use official split files under root, same as
                # DC-LoRA's CUB200Dataset. Root should point to CUB_200_2011.

                # 1) Read image paths (relative to images/)
                images_file = os.path.join(root, "images.txt")
                image_id_to_path = {}
                with open(images_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_id = int(parts[0])
                            img_path = parts[1]
                            image_id_to_path[img_id] = img_path

                # 2) Read class labels (1-indexed -> 0-indexed)
                labels_file = os.path.join(root, "image_class_labels.txt")
                image_id_to_label = {}
                with open(labels_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_id = int(parts[0])
                            label = int(parts[1]) - 1
                            image_id_to_label[img_id] = label

                # 3) Read train/test split
                split_file = os.path.join(root, "train_test_split.txt")
                image_id_to_split = {}
                with open(split_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_id = int(parts[0])
                            is_train = int(parts[1]) == 1
                            image_id_to_split[img_id] = is_train

                train_paths = []
                train_labels_list = []
                test_paths = []
                test_labels_list = []

                for img_id in sorted(image_id_to_path.keys()):
                    is_train = image_id_to_split.get(img_id, True)
                    img_rel = image_id_to_path[img_id]
                    img_path = os.path.join(root, "images", img_rel)
                    label = image_id_to_label[img_id]

                    if is_train:
                        train_paths.append(img_path)
                        train_labels_list.append(label)
                    else:
                        test_paths.append(img_path)
                        test_labels_list.append(label)

                train_imgs = np.array([jpg_image_to_array(p) for p in train_paths])
                train_labels = np.array(train_labels_list)
                test_imgs = np.array([jpg_image_to_array(p) for p in test_paths])
                test_labels = np.array(test_labels_list)
                num_classes = 200
            else:
                raise NotImplementedError(
                    f"Unsupported dataset in multi-dataset sequence: {ds_name}"
                )

            # Apply global class offset for this dataset.
            train_labels_off = train_labels + class_offset
            test_labels_off = test_labels + class_offset

            train_data_list.append(train_imgs)
            train_targets_list.append(train_labels_off)
            test_data_list.append(test_imgs)
            test_targets_list.append(test_labels_off)

            # Per-dataset class-incremental split, DC-LoRA style.
            classes_per_task = num_classes // num_tasks
            for _ in range(num_tasks):
                increments.append(classes_per_task)

            class_offset += num_classes

        self.train_data = np.concatenate(train_data_list)
        self.train_targets = np.concatenate(train_targets_list)
        self.test_data = np.concatenate(test_data_list)
        self.test_targets = np.concatenate(test_targets_list)

        total_classes = class_offset
        self.class_order = list(range(total_classes))
        self.increments = increments


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr
