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
from utils.metrics import accuracy_over_time, metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # if args.is_training:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        # + "_" + timestamp
        log_file_name = "runs/" + args.model_id  + ".log"
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

    # def _write_graph(self, model, train_loader):
    #     temp_x, temp_y, temp_x_mark, temp_y_mark, padding_mask = next(iter(train_loader))

    #     # Move the data to the correct device
    #     temp_x = temp_x.float().to(self.device)
    #     temp_y = temp_y.float().to(self.device)
    #     temp_x_mark = temp_x_mark.float().to(self.device)
    #     temp_y_mark = temp_y_mark.float().to(self.device)
    #     padding_mask = padding_mask.to(self.device)

    #     # If your model is wrapped in DataParallel, access the original model with .module
    #     original_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    #     # Add the model graph to TensorBoard
    #     self.writer.add_graph(original_model, [temp_x, temp_x_mark, temp_y, temp_y_mark, padding_mask])
    #     if self.args.use_multi_gpu:
    #         self.model = nn.DataParallel(original_model, device_ids=self.args.device_ids)
    #         print("use multi gpu after graph writing")
    #     return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        class_weights = torch.from_numpy(vali_loader.dataset.class_weights)
        # class_weights = class_weights.float().to(self.device)
        criterion = self._select_criterion(class_weights)
        dates = vali_loader.dataset.get_dates()

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
                # class_weights = class_weights.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # Decoder turned off for classification, dec_inp=batch_y

                # dec_inp = torch.zeros_like(
                #     batch_y[:, -self.args.pred_len :, :]
                # ).float()
                # dec_inp = (
                #     torch.cat(
                #         [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                #     )
                #     .float()
                #     .to(self.device)
                # )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
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
                f_dim = -1 if self.args.features == "MS" else 0
                # outputs = outputs[:, -self.args.pred_len :, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                #     self.device
                # )
                outputs = outputs.float()

                pred = outputs
                true = batch_y
                # if not pred.device == self.device or not true.device == self.device:
                #     print(pred.device, true.device)
                #     print("Warning: pred and true are not on the same device as criterion")
                loss = criterion(pred, true)
                loss = loss.detach().cpu().numpy()
                total_loss.append(loss)
                pred.detach().cpu().numpy()
                true.detach().cpu().numpy()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # profiler = cProfile.Profile()
        # profiler.enable()
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=-0.05
        )

        model_optim = self._select_optimizer()

        class_weights = train_loader.dataset.get_class_weights()
        criterion = self._select_criterion(class_weights)
        scheduler = lr_scheduler.ExponentialLR(model_optim, gamma=0.9)
        dates = train_loader.dataset.get_dates()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        preds = []
        trues = []
        seq_end_lengths_list = []
        steps = 0
        self.model.apply(self._weights_init)

        # self.model = self._write_graph(self.model, train_loader)
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

                # for j in sequences:
                iter_count += 1
                steps += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                padding_mask = padding_mask.to("cpu")
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        print("using amp")
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:

                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            batch_y,
                            batch_y_mark,
                            padding_mask,
                        )
                        # self.writer.add_graph(
                        #     self.model,
                        #     [
                        #         batch_x,
                        #         batch_x_mark,
                        #         batch_y,
                        #         batch_y_mark,
                        #         padding_mask,
                        #     ],
                        # )
                    else:
                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            batch_y,
                            batch_y_mark,
                            padding_mask,
                        )
                        # self.writer.add_graph(
                        #     self.model,
                        #     [
                        #         batch_x,
                        #         batch_x_mark,
                        #         batch_y,
                        #         batch_y_mark,
                        #         padding_mask,
                        #     ],
                        # )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs.float()
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    # log the loass per step to tensorboard
                    self.writer.add_scalar("Loss/train_steps", loss, steps)

                    probabilities = F.softmax(outputs, dim=1)
                    probabilities = probabilities.detach().cpu().numpy()

                    one_hot_prob = np.eye(len(probabilities[0]))[
                        np.argmax(probabilities[0])
                    ]
                    true = batch_y.detach().cpu().numpy()
                    preds.append(one_hot_prob)
                    trues.append(true[0])
                    seq_end_lengths_list.append(seq_end_lengths)
                    # Detach and move to CPU

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
                    # profiler.disable()
                    # stats = pstats.Stats(profiler).sort_stats("cumtime")
                    # stats.strip_dirs()
                    # stats.print_stats()
                    # stats.dump_stats("./test_results/test.prof")

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                self.writer.add_scalar("Loss/train", loss, epoch)
                for name, param in self.model.named_parameters():
                    # print(name, param)
                    # print(i)
                    self.writer.add_histogram(
                        "Weights/" + name, param.data, epoch
                    )
                    self.writer.add_histogram(
                        "Gradients/" + name, param.grad, epoch
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
            ) = metric(preds, trues, target_names, dates)

            print(
                "acc:{}, prec:{}, prec_micro:{}, recall:{}, recall_micro{}, F1:{}, F1_micro{}".format(
                    acc, prec, prec_micro, rec, rec_micro, F1, F1_micro
                )
            )
            # Add hyperparameters to SummaryWriter
            # self.writer.add_hparams(vars(self.args), {})
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

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)
        # torch.cuda.empty_cache()
        # gc.collect()
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints/" + setting, "checkpoint.pth")
                )
            )
        preds = []
        trues = []
        # accuracy_over_time = []
        seq_end_lengths_list = []
        dates = test_loader.dataset.get_dates()
        # seq_end_lengths = np.load(
        #     "/home/guilly/iTransformer/dataset/noordoostpolder/test/seq_end_lengths.npy"
        # )
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

                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(
                #     batch_y[:, -self.args.pred_len :, :]
                # ).float()
                # dec_inp = (
                #     torch.cat(
                #         [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                #     )
                #     .float()
                #     .to(self.device)
                # )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
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

                f_dim = -1 if self.args.features == "MS" else 0
                # outputs = outputs[:, -self.args.pred_len :, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                #     self.device
                # )
                outputs = outputs.float()
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)

                # Detach and move to CPU
                probabilities = probabilities.detach().cpu().numpy()

                # Get the predicted class labels by taking the argmax of the probabilities
                # print(probabilities.shape, probabilities)
                predictions = probabilities
                # predictions = np.argmax(probabilities, axis=1)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(
                #         outputs.squeeze(0)
                #     ).reshape(shape)
                #     batch_y = test_data.inverse_transform(
                #         batch_y.squeeze(0)
                #     ).reshape(shape)

                # pred = outputs
                true = batch_y

                if i % 100 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.squeeze(0)
                        ).reshape(shape)
                    gt = true
                    pd = predictions
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
                    # self.writer.add_histogram(
                    #     "Probability Distribution",
                    #     probabilities[0],
                    #     global_step=i,
                    # )
                one_hot_prob = np.eye(len(probabilities[0]))[
                    np.argmax(probabilities[0])
                ]

                preds.append(one_hot_prob)
                trues.append(true[0])
                seq_end_lengths_list.append(seq_end_lengths)

        preds = np.array(preds)  # [Batch, no_classes]
        trues = np.array(trues)

        print("test shape:", preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        target_names = test_data.target

        acc_time, acc_figure = accuracy_over_time(
            preds, trues, seq_end_lengths_list, dates
        )
        print(acc_time)
        acc_figure.savefig(folder_path + "accuracy_over_time.png")
        acc, conf_matrix, prec, prec_micro, rec, rec_micro, F1, F1_micro = (
            metric(preds, trues, target_names, dates)
        )

        print(
            "acc:{}, prec:{}, prec_micro:{}, recall:{}, recall_micro{}, F1:{}, F1_micro{}".format(
                acc, prec, prec_micro, rec, rec_micro, F1, F1_micro
            )
        )

        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("acc:{}, prec:{}, recall:{}, F1:{}".format(acc, prec, rec, F1))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([acc, prec, rec, F1]))
        np.save(folder_path + "confusion_matrix.npy", conf_matrix)
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        conf_matrix.savefig(folder_path + "confusion_matrix.png")
        # plt.savefig(folder_path + "confusion_matrix.pdf")
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

                # decoder input
                # dec_inp = torch.zeros_like(
                #     batch_y[:, -self.args.pred_len :, :]
                # ).float()
                # dec_inp = (
                #     torch.cat(
                #         [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                #     )
                #     .float()
                #     .to(self.device)
                # )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )[0]
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
