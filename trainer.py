import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


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

        # if task >= 3: break

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
