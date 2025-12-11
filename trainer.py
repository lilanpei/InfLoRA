import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.whitened_ncm_head import WhitenedNCMClassifier


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    device = device.split(",")

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _train(args):
    if args["model_name"] in [
        "InfLoRA",
        "InfLoRA_domain",
        "InfLoRAb5_domain",
        "InfLoRAb5",
        "InfLoRA_CA",
        "InfLoRA_CA1",
    ]:
        logdir = "logs/{}/{}_{}_{}/{}/{}/{}/{}_{}-{}".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["net_type"],
            args["model_name"],
            args["optim"],
            args["rank"],
            args["lamb"],
            args["lame"],
            args["lrate"],
        )
    else:
        logdir = "logs/{}/{}_{}_{}/{}/{}".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["net_type"],
            args["model_name"],
            args["optim"],
        )

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, "{}".format(args["seed"]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if not os.path.exists(logfilename):
        os.makedirs(logfilename)
    print(logfilename)
    save_ckpt = args.get("save_checkpoints", True)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)

    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = (
        {"top1": []},
        {"top1": []},
        {"top1": []},
        {"top1": []},
    )
    # Track per-task accuracies for forgetting computation (DC-LoRA style)
    # best_accuracy_per_task[t]: best accuracy for task t over all evaluations
    # current_accuracy_per_task[t]: current accuracy for task t at the end
    num_tasks = data_manager.nb_tasks
    best_accuracy_per_task = [0.0 for _ in range(num_tasks)]
    current_accuracy_per_task = [0.0 for _ in range(num_tasks)]

    use_wncm = args.get("use_wncm", args["dataset"] == "multi_dataset")
    wncm = None
    wncm_cumulative_accuracies = []
    best_accuracy_per_task_wncm = [0.0 for _ in range(num_tasks)]
    current_accuracy_per_task_wncm = [0.0 for _ in range(num_tasks)]
    avg_forgetting_history_wncm = []
    max_forgetting_history_wncm = []
    task_classes_by_id = []
    start_class = 0
    for task_id in range(num_tasks):
        task_size = data_manager.get_task_size(task_id)
        task_classes = list(range(start_class, start_class + task_size))
        task_classes_by_id.append(task_classes)
        start_class += task_size
    if use_wncm:
        wncm_lambda = args.get("wncm_lambda", 0.07)
        wncm = WhitenedNCMClassifier(
            feature_dim=model._network.feature_dim,
            shrinkage_lambda=wncm_lambda,
            device="cpu",
            use_whitening=True,
        )
    seen_classes = []

    for task in range(num_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info("Time:{}".format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()
        logging.info("Time:{}".format(time_end - time_start))
        # raise Exception
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve_with_task["top1"].append(cnn_accy_with_task["top1"])
        cnn_curve_task["top1"].append(cnn_accy_task)
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top1 with task curve: {}".format(cnn_curve_with_task["top1"]))
        logging.info("CNN top1 task curve: {}".format(cnn_curve_task["top1"]))

        # Update per-task accuracy tracking for forgetting (running best over time)
        grouped = cnn_accy["grouped"]
        # Extract per-task accuracies from grouped dict (keys like '00-09', '10-19', etc.)
        for t in range(task + 1):
            key = "{:02d}-{:02d}".format(
                t * args["increment"], (t + 1) * args["increment"] - 1
            )
            if key in grouped:
                acc_t = float(grouped[key])
                current_accuracy_per_task[t] = acc_t
                if acc_t > best_accuracy_per_task[t]:
                    best_accuracy_per_task[t] = acc_t

        if use_wncm:
            seen_classes.extend(task_classes_by_id[task])
            seen_classes = sorted(set(seen_classes))
            indices_current = np.array(task_classes_by_id[task])
            train_eval_dataset = data_manager.get_dataset(
                indices_current,
                source="train",
                mode="test",
            )
            train_eval_loader = DataLoader(
                train_eval_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                num_workers=args["num_workers"],
            )
            features_list = []
            labels_list = []
            model._network.eval()
            for _, (_, inputs, targets) in enumerate(train_eval_loader):
                inputs = inputs.to(model._device)
                with torch.no_grad():
                    if hasattr(model._network, "extract_vector"):
                        feats = model._network.extract_vector(inputs)
                    else:
                        outputs = model._network(inputs)
                        if isinstance(outputs, dict) and "features" in outputs:
                            feats = outputs["features"]
                        else:
                            feats = outputs
                features_list.append(feats.cpu())
                labels_list.append(targets.cpu())
            if features_list:
                feats_np = torch.cat(features_list, dim=0).numpy().astype(np.float32)
                labels_np = torch.cat(labels_list, dim=0).numpy().astype(np.int64)
                wncm.update(feats_np, labels_np, task)
                wncm.post_update_hook(task)
            test_indices = np.array(seen_classes)
            test_dataset = data_manager.get_dataset(
                test_indices,
                source="test",
                mode="test",
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                num_workers=args["num_workers"],
            )
            feats_eval_list = []
            labels_eval_list = []
            model._network.eval()
            for _, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(model._device)
                with torch.no_grad():
                    if hasattr(model._network, "extract_vector"):
                        feats = model._network.extract_vector(inputs)
                    else:
                        outputs = model._network(inputs)
                        if isinstance(outputs, dict) and "features" in outputs:
                            feats = outputs["features"]
                        else:
                            feats = outputs
                feats_eval_list.append(feats.cpu())
                labels_eval_list.append(targets.cpu())
            if feats_eval_list:
                feats_eval_np = (
                    torch.cat(feats_eval_list, dim=0).numpy().astype(np.float32)
                )
                labels_eval_np = (
                    torch.cat(labels_eval_list, dim=0).numpy().astype(np.int64)
                )
                preds_eval_np = wncm.predict(feats_eval_np)
                grouped_wncm = {}
                task_accuracy_strings = []
                for past_idx, past_classes in enumerate(task_classes_by_id[: task + 1]):
                    class_array = np.array(past_classes, dtype=np.int64)
                    task_mask = np.isin(labels_eval_np, class_array)
                    if task_mask.sum() == 0:
                        continue
                    wncm_task_acc = float(
                        (preds_eval_np[task_mask] == labels_eval_np[task_mask]).mean()
                        * 100.0
                    )
                    current_accuracy_per_task_wncm[past_idx] = wncm_task_acc
                    if wncm_task_acc > best_accuracy_per_task_wncm[past_idx]:
                        best_accuracy_per_task_wncm[past_idx] = wncm_task_acc
                    label = "{:02d}-{:02d}".format(past_classes[0], past_classes[-1])
                    grouped_wncm[label] = wncm_task_acc
                    summary = "T{}: W-NCM {:.2f}% (best {:.2f}%)".format(
                        past_idx + 1,
                        wncm_task_acc,
                        best_accuracy_per_task_wncm[past_idx],
                    )
                    task_accuracy_strings.append(summary)
                if grouped_wncm:
                    ave_acc_wncm = float(np.mean(list(grouped_wncm.values())))
                else:
                    ave_acc_wncm = 0.0
                wncm_cumulative_accuracies.append(ave_acc_wncm)
                forgetting_values_wncm = []
                for idx in range(len(task_classes_by_id[: task + 1]) - 1):
                    f = max(
                        best_accuracy_per_task_wncm[idx]
                        - current_accuracy_per_task_wncm[idx],
                        0.0,
                    )
                    forgetting_values_wncm.append(f)
                if forgetting_values_wncm:
                    avg_forgetting_wncm = float(np.mean(forgetting_values_wncm))
                    max_forgetting_wncm = float(np.max(forgetting_values_wncm))
                else:
                    avg_forgetting_wncm = 0.0
                    max_forgetting_wncm = 0.0
                avg_forgetting_history_wncm.append(avg_forgetting_wncm)
                max_forgetting_history_wncm.append(max_forgetting_wncm)
                logging.info("W-NCM: {}".format(grouped_wncm))
                logging.info("Ave Acc (W-NCM): {:.2f}%".format(ave_acc_wncm))
                logging.info(
                    "Per-task accuracies (W-NCM): {}".format(
                        "; ".join(task_accuracy_strings)
                    )
                )
                logging.info(
                    "Average forgetting (W-NCM): {:.2f}% | Max forgetting (W-NCM): {:.2f}%".format(
                        avg_forgetting_wncm, max_forgetting_wncm
                    )
                )

        # if task >= 3: break

        if save_ckpt:
            torch.save(
                model._network.state_dict(),
                os.path.join(logfilename, "task_{}.pth".format(int(task))),
            )

    # Compute and log final metrics (similar to DC-LoRA)
    final_avg_acc = cnn_curve["top1"][-1] if cnn_curve["top1"] else 0.0
    avg_acc_over_tasks = float(np.mean(cnn_curve["top1"])) if cnn_curve["top1"] else 0.0

    # Compute forgetting: for each task except the last, forgetting = best - current
    forgetting_values = []
    for idx in range(num_tasks - 1):
        forgetting = max(
            best_accuracy_per_task[idx] - current_accuracy_per_task[idx], 0.0
        )
        forgetting_values.append(forgetting)

    final_avg_forgetting = (
        float(np.mean(forgetting_values)) if forgetting_values else 0.0
    )
    final_max_forgetting = (
        float(np.max(forgetting_values)) if forgetting_values else 0.0
    )

    logging.info("\n===== Summary =====")
    logging.info("Final average accuracy: {:.2f}%".format(final_avg_acc))
    logging.info("Average accuracy over tasks: {:.2f}%".format(avg_acc_over_tasks))
    logging.info("Final average forgetting: {:.2f}%".format(final_avg_forgetting))
    logging.info("Final max forgetting: {:.2f}%".format(final_max_forgetting))
    if use_wncm and wncm_cumulative_accuracies:
        final_avg_acc_wncm = wncm_cumulative_accuracies[-1]
        avg_acc_over_tasks_wncm = float(np.mean(wncm_cumulative_accuracies))
        if avg_forgetting_history_wncm:
            final_avg_forgetting_wncm = avg_forgetting_history_wncm[-1]
            final_max_forgetting_wncm = max_forgetting_history_wncm[-1]
        else:
            final_avg_forgetting_wncm = 0.0
            final_max_forgetting_wncm = 0.0
        logging.info("W-NCM final average accuracy: {:.2f}%".format(final_avg_acc_wncm))
        logging.info(
            "W-NCM average accuracy over tasks: {:.2f}%".format(avg_acc_over_tasks_wncm)
        )
        logging.info(
            "W-NCM final average forgetting: {:.2f}%".format(final_avg_forgetting_wncm)
        )
        logging.info(
            "W-NCM final max forgetting: {:.2f}%".format(final_max_forgetting_wncm)
        )


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(args):
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
