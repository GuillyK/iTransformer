import cProfile
import datetime
import gc
import os
import pstats
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import ConfusionMatrixDisplay
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.metrics import accuracy_over_time, metric, visualize_attention
from utils.tensorboard_monitors import WeightChangeMonitor
from utils.tools import EarlyStopping, adjust_learning_rate, visual


warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # if args.is_training:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        # + "_" + timestamp
        if args.is_training:
            log_file_name = "runs/" + args.model_id + ".log"
        else:
            log_file_name = "runs/test/" + args.model_id + ".log"
        self.writer = SummaryWriter(log_dir=log_file_name)
        # self.writer.add_hparams(vars(self.args), {})

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            print("use multi gpu")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return model_optim

    def _select_criterion(self, class_weights):
        # criterion = nn.MSELoss()
        if isinstance(class_weights, np.ndarray):
            class_weights = torch.from_numpy(class_weights)
        class_weights = class_weights.float().to(self.device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.05
        )
        return criterion

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        class_weights = torch.from_numpy(vali_loader.dataset.class_weights)
        criterion = self._select_criterion(class_weights)
        dates = vali_loader.dataset.get_dates()

        # Validation loop
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                padding_mask,
                seq_end_lengths,
            ) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                class_weights = vali_loader.dataset.class_weights

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.output_attention:
                    outputs, attns = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )
                else:
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )

                outputs = outputs.float()

                pred = outputs
                true = batch_y
                loss = criterion(pred, true)
                loss = loss.detach().cpu().numpy()
                total_loss.append(loss)
                pred.detach().cpu().numpy()
                true.detach().cpu().numpy()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # get all data loaders
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # setup parameters
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=0.0001
        )

        model_optim = self._select_optimizer()

        class_weights = train_loader.dataset.get_class_weights()
        criterion = self._select_criterion(class_weights)
        scheduler = lr_scheduler.ExponentialLR(model_optim, gamma=0.9)
        dates = train_loader.dataset.get_dates()

        preds = []
        trues = []
        seq_end_lengths_list = []
        steps = 0
        self.model.apply(self._weights_init)
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Number of trainable parameters: {num_trainable_params}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                padding_mask,
                seq_end_lengths,
            ) in enumerate(train_loader):
                # reset model to correct layer size based on seq_len
                if steps == 0:
                    new_seq_len = batch_x.size(1)
                    print(f"{new_seq_len=}")
                    self.model.reset(new_seq_len)


                iter_count += 1
                steps += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                padding_mask = padding_mask.to("cpu")

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)




                if self.args.output_attention:
                    outputs, attns = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )

                else:
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )


                outputs = outputs.float()
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # log the loass per step to tensorboard
                self.writer.add_scalar("Loss/train_steps", loss, steps)

            	# Apply softmax to get class probabilities
                probabilities = F.softmax(outputs, dim=1)
                probabilities = probabilities.detach().cpu().numpy()

                one_hot_prob = np.eye(len(probabilities[0]))[
                    np.argmax(probabilities[0])
                ]
                true = batch_y.detach().cpu().numpy()
                preds.append(one_hot_prob)
                trues.append(true[0])
                seq_end_lengths_list.append(seq_end_lengths)


                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = (
                        speed
                        * ((self.args.train_epochs - epoch) * train_steps - i)
                        / 3600
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}h".format(
                            speed, left_time
                        )
                    )
                    # set attention in tensorboard
                    if self.args.output_attention:
                        visualize_attention(
                            attention_weights=attns,
                            input_tokens=len(batch_x[0]),
                            writer=self.writer,
                            global_step=steps,
                        )

                    # Get class probabilities and plot them
                    fig = plt.figure(figsize=(4, 4))
                    plt.bar(range(len(probabilities[0])), probabilities[0])
                    plt.xlabel("Classes")
                    plt.ylabel("Frequency")
                    plt.text(
                        0.1,
                        0.9,
                        f"True Label: {true[0]}",
                        transform=plt.gca().transAxes,
                    )
                    self.writer.add_figure(
                        "Probability Distribution", fig, global_step=steps
                    )
                    plt.close(fig)
                    iter_count = 0
                    time_now = time.time()


                loss.backward()
                model_optim.step()
                self.writer.add_scalar("Loss/train", loss, epoch)
                # Monitor some gradient/loss histograms with change
                if epoch == 0:
                    weight_monitor = WeightChangeMonitor(self.model, top_k=5)
                layers_to_monitor = weight_monitor.get_layers_to_monitor(
                    self.model, weight_monitor
                )
                for name in layers_to_monitor:
                    param = dict(self.model.named_parameters())[name]
                    self.writer.add_histogram(
                        f"Weights/{name}", param.data, epoch
                    )
                    if param.grad is not None:
                        self.writer.add_histogram(
                            f"gradients/{name}", param.grad, epoch
                        )

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            # Print memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (
                1024 * 1024 * 1024
            )  # in GB
            print("Memory Usage: {:.2f} GB".format(memory_usage))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Add values to tensorboard
            self.writer.add_scalar("Loss/vali", vali_loss, epoch)
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            # log learning rate
            self.writer.add_scalar(
                "Learning Rate", model_optim.param_groups[0]["lr"], epoch
            )
            acc_time, acc_time_figure = accuracy_over_time(
                preds, trues, seq_end_lengths_list, dates
            )

            self.writer.add_figure(
                "Accuracy over time", acc_time_figure, epoch
            )
            plt.close(acc_time_figure)
            target_names = train_data.target
            (
                acc,
                conf_matrix,
                prec,
                prec_micro,
                rec,
                rec_micro,
                F1,
                F1_micro,
                metrics_df_per_class,
            ) = metric(preds, trues, target_names, dates)

            print(
                "acc:{}, prec:{}, prec_micro:{}, recall:{}, recall_micro{}, F1:{}, F1_micro{}".format(
                    acc, prec, prec_micro, rec, rec_micro, F1, F1_micro
                )
            )
            # Add metrics to SummaryWriter
            self.writer.add_scalar("Accuracy", acc, epoch)
            self.writer.add_scalar("Precision", prec, epoch)
            self.writer.add_scalar("Precision_micro", prec_micro, epoch)
            self.writer.add_scalar("Recall", rec, epoch)
            self.writer.add_scalar("Recall_micro", rec_micro, epoch)
            self.writer.add_scalar("F1", F1, epoch)
            self.writer.add_scalar("F1_micro", F1_micro, epoch)
            self.writer.add_figure("Confusion Matrix", conf_matrix, epoch)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            scheduler.step()

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            batch_x, _, _, _, _, _ = next(iter(test_loader))
            new_seq_len = batch_x.size(1)
            print(f"{new_seq_len=}")
            self.model.reset(new_seq_len)
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints/" + setting, "checkpoint.pth")
                )
            )
        preds = []
        trues = []
        seq_end_lengths_list = []
        dates = test_loader.dataset.get_dates()

        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                padding_mask,
                seq_end_lengths,
            ) in enumerate(tqdm(test_loader)):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.output_attention:
                    outputs, attns = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )

                else:
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        batch_y,
                        batch_y_mark,
                        padding_mask,
                    )

                outputs = outputs.float()
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)

                # Detach and move to CPU
                probabilities = probabilities.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                true = batch_y

                if i % 100 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.squeeze(0)
                        ).reshape(shape)
                    gt = true
                    pd = probabilities
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

                one_hot_prob = np.eye(len(probabilities[0]))[
                    np.argmax(probabilities[0])
                ]

                preds.append(one_hot_prob)
                trues.append(true[0])
                seq_end_lengths_list.append(seq_end_lengths)

        preds = np.array(preds)  # [Batch, no_classes]
        trues = np.array(trues)

        print("test shape:", preds.shape, trues.shape)


        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        target_names = test_data.target

        acc_time, acc_time_figure = accuracy_over_time(
            preds, trues, seq_end_lengths_list, dates
        )

        self.writer.add_figure("test/Accuracy over time", acc_time_figure)
        plt.close(acc_time_figure)
        acc, conf_matrix, prec, prec_micro, rec, rec_micro, F1, F1_micro, metrics_df_per_class = (
            metric(preds, trues, target_names, dates)
        )

        print(
            "acc:{}, prec:{}, prec_micro:{}, recall:{}, recall_micro{}, F1:{}, F1_micro{}".format(
                acc, prec, prec_micro, rec, rec_micro, F1, F1_micro
            )
        )
        self.writer.add_scalar("test/Accuracy", acc)
        self.writer.add_scalar("test/Precision", prec)
        self.writer.add_scalar("test/Precision_micro", prec_micro)
        self.writer.add_scalar("test/Recall", rec)
        self.writer.add_scalar("test/Recall_micro", rec_micro)
        self.writer.add_scalar("test/F1", F1)
        self.writer.add_scalar("test/F1_micro", F1_micro)
        self.writer.add_figure("test/Confusion Matrix", conf_matrix)

        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("acc:{}, prec:{}, recall:{}, F1:{}".format(acc, prec, rec, F1))
        f.write("\n")
        f.write("\n")
        f.close()

        metrics_df_per_class_html = metrics_df_per_class.to_html()
        self.writer.add_text("test/Metrics per class", metrics_df_per_class_html)
        metrics_df_per_class.to_csv(folder_path + "metrics_per_class.csv")
        np.save(folder_path + "metrics.npy", np.array([acc, prec, rec, F1]))
        np.save(folder_path + "confusion_matrix.npy", conf_matrix)
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        conf_matrix.savefig(folder_path + "confusion_matrix.png")
        plt.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                
                if self.args.output_attention:
                    outputs, attns = self.model(
                        batch_x, batch_x_mark, batch_y, batch_y_mark
                    )
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, batch_y, batch_y_mark
                    )
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(
                        outputs.squeeze(0)
                    ).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
