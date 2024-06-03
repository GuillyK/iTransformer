import os
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
import sklearn.metrics

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from layers.Elects_loss import EarlyRewardLoss
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast_Elects(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Elects, self).__init__(args)
        log_file_name = "runs/" + args.model_id + ".log"
        self.writer = SummaryWriter(log_dir=log_file_name)
        torch.autograd.set_detect_anomaly(True)
        # self.writer.add_hparams(vars(self.args), {})

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
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
        criterion = EarlyRewardLoss(alpha=0.5, epsilon=10)
        return criterion

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        stats = []
        self.model.eval()
        class_weights = torch.from_numpy(vali_loader.dataset.class_weights)
        # class_weights = class_weights.float().to(self.device)
        criterion = self._select_criterion(class_weights)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                class_weights = vali_loader.dataset.class_weights
                # class_weights = class_weights.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # print(f"{batch_y=}")
                # print(f"{batch_y.shape=}")
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            (
                                log_class_probabilities,
                                probability_stopping,
                                predictions_at_t_stop,
                                t_stop,
                            ) = self.model.predict(
                                batch_x, batch_x_mark, batch_y, batch_y_mark, self.device
                            )
                        else:
                            (
                                log_class_probabilities,
                                probability_stopping,
                                predictions_at_t_stop,
                                t_stop,
                            ) = self.model.predict(
                                batch_x, batch_x_mark, batch_y, batch_y_mark, self.device
                            )
                else:
                    if self.args.output_attention:
                        (
                            log_class_probabilities,
                            probability_stopping,
                            predictions_at_t_stop,
                            t_stop,
                        ) = self.model.predict(
                            batch_x, batch_x_mark, batch_y, batch_y_mark, self.device
                        )
                    else:
                        (
                            log_class_probabilities,
                            probability_stopping,
                            predictions_at_t_stop,
                            t_stop,
                        ) = self.model.predict(
                            batch_x, batch_x_mark, batch_y, batch_y_mark, self.device
                        )

                loss, stat = criterion(
                    log_class_probabilities,
                    probability_stopping,
                    batch_y,
                    return_stats=True,
                )
                stat["loss"] = loss.cpu().detach().numpy()
                stat["probability_stopping"] = (
                    probability_stopping.cpu().detach().numpy()
                )
                stat["class_probabilities"] = (
                    log_class_probabilities.exp().cpu().detach().numpy()
                )
                stat["predictions_at_t_stop"] = (
                    predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
                )
                stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
                stat["targets"] = batch_y.argmax(dim=-1).cpu().detach().numpy()

                stats.append(stat)

                losses.append(loss.cpu().detach().numpy())

        stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}

        self.model.train()
        return np.stack(losses).mean(), stats

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        model_optim = self._select_optimizer()

        class_weights = train_loader.dataset.class_weights
        # class_weights = class_weights.float().to(self.device)
        criterion = self._select_criterion(class_weights)
        scheduler = lr_scheduler.ExponentialLR(model_optim, gamma=0.9)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        preds = []
        trues = []
        self.model.apply(self._weights_init)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                if i == 32:
                    break
                class_weights = train_loader.dataset.class_weights
                # for j in sequences:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder

                if self.args.output_attention:
                    log_class_probabilities, probability_stopping = self.model(
                        batch_x, batch_x_mark, batch_y, batch_y_mark
                    )
                    self.writer.add_graph(
                        self.model,
                        [batch_x, batch_x_mark, batch_y, batch_y_mark],
                    )
                else:
                    log_class_probabilities, probability_stopping = self.model(
                        batch_x, batch_x_mark, batch_y, batch_y_mark
                    )
                    self.writer.add_graph(
                        self.model,
                        [batch_x, batch_x_mark, batch_y, batch_y_mark],
                    )
                    # print(log_class_probabilities, probability_stopping)
                    # print(
                    #     log_class_probabilities.shape,
                    #     probability_stopping.shape,
                    # )
                    loss = criterion(
                        log_class_probabilities, probability_stopping, batch_y
                    )

                    train_loss.append(loss.item())

                    # probabilities = (log_class_probabilities.exp())
                    # probabilities = probabilities.detach().cpu().numpy()
                    # print(f"{probabilities=}")
                    # print(f"{probabilities.shape=}")
                    # one_hot_prob = np.eye(len(probabilities[0]))[
                    #     np.argmax(probabilities[0])
                    # ]
                    # true = batch_y.detach().cpu().numpy()
                    # preds.append(one_hot_prob)
                    # trues.append(true[0])
                    # Detach and move to CPU

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    # fig = plt.figure(figsize=(4, 4))
                    # plt.bar(range(len(probabilities[0])), probabilities[0])
                    # plt.xlabel("Classes")
                    # plt.ylabel("Frequency")
                    # self.writer.add_figure(
                    #     "Probability Distribution", fig, global_step=i
                    # )
                    # plt.close(fig)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                self.writer.add_scalar("Loss/train", loss, epoch)
                for name, param in self.model.named_parameters():
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
            print("start validation")
            vali_loss, stats = self.vali(vali_data, vali_loader, criterion)
            print("start testing")
            test_loss, test_stats = self.vali(
                test_data, test_loader, criterion
            )

            self.writer.add_scalar("Loss/vali", vali_loss, epoch)
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            # log learning rate
            self.writer.add_scalar(
                "Learning Rate", model_optim.param_groups[0]["lr"], epoch
            )
            target_names = train_data.target
            # acc, conf_matrix, prec, rec, F1 = metric(
            #     preds, trues, target_names
            # )
            print("prediction", stats["predictions_at_t_stop"][:, 0])
            print(f"{stats['targets']=}")
            print(f"{stats['predictions_at_t_stop'].shape=}")
            print(f"{stats['targets'].shape=}")
            prec, rec, F1, support = (
                sklearn.metrics.precision_recall_fscore_support(
                    y_pred=stats["predictions_at_t_stop"][:, 0],
                    y_true=stats["targets"],
                    average="macro",
                    zero_division=0,
                )
            )
            acc = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"],
            )
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"]
            )

            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (
                stats["t_stop"].mean() / (self.args.seq_len - 1)
            )

            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"],
            )
            # Add hyperparameters to SummaryWriter
            # self.writer.add_hparams(vars(self.args), {})
            self.writer.add_scalar("Accuracy", acc, epoch)
            self.writer.add_scalar("Precision", prec, epoch)
            self.writer.add_scalar("Recall", rec, epoch)
            self.writer.add_scalar("F1", F1, epoch)
            self.writer.add_scalar("Kappa", kappa, epoch)
            self.writer.add_scalar("Classification Loss", classification_loss, epoch)
            self.writer.add_scalar("Earliness Reward", earliness_reward, epoch)
            self.writer.add_scalar("Earliness", earliness, epoch)
            self.writer.add_figure("Confusion Matrix", stats["confusion_matrix"], epoch)
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
        stats = []
        losses = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        class_weights = test_loader.dataset.class_weights
        criterion = self._select_criterion(class_weights)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                tqdm(test_loader)
            ):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            (
                                log_class_probabilities,
                                probability_stopping,
                                predictions_at_t_stop,
                                t_stop,
                            ) = self.model.predict(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                        else:
                            (
                                log_class_probabilities,
                                probability_stopping,
                                predictions_at_t_stop,
                                t_stop,
                            ) = self.model.predict(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        (
                            log_class_probabilities,
                            probability_stopping,
                            predictions_at_t_stop,
                            t_stop,
                        ) = self.model.predict(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )

                    else:
                        (
                            log_class_probabilities,
                            probability_stopping,
                            predictions_at_t_stop,
                            t_stop,
                        ) = self.model.predict(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )

                loss, stat = criterion(
                    log_class_probabilities,
                    probability_stopping,
                    batch_y,
                    return_stats=True,
                )
                # Apply softmax to get probabilities
                stat["loss"] = loss.cpu().detach().numpy()
                stat["probability_stopping"] = (
                    probability_stopping.cpu().detach().numpy()
                )
                stat["class_probabilities"] = (
                    log_class_probabilities.exp().cpu().detach().numpy()
                )
                stat["predictions_at_t_stop"] = (
                    predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
                )
                stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
                stat["targets"] = y_true.cpu().detach().numpy()

                stats.append(stat)
                losses.append(loss.cpu().detach().numpy())
                probabilities = F.softmax(
                    np.exp(log_class_probabilities), dim=1
                )

                # Detach and move to CPU
                probabilities = probabilities.detach().cpu().numpy()

                # Get the predicted class labels by taking the argmax of the probabilities
                # print(probabilities.shape, probabilities)
                predictions = probabilities
                # predictions = np.argmax(probabilities, axis=1)
                batch_y = batch_y.detach().cpu().numpy()

                # true = batch_y

                if i % 100 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.squeeze(0)
                        ).reshape(shape)
                    gt = batch_y
                    pd = predictions
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
                    self.writer.add_histogram(
                        "Probability Distribution",
                        probabilities[0],
                        global_step=i,
                    )
                one_hot_prob = np.eye(len(probabilities[0]))[
                    np.argmax(probabilities[0])
                ]

                preds.append(one_hot_prob)
                trues.append(true[0])

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
        acc, conf_matrix, prec, rec, F1 = metric(preds, trues, target_names)

        print("acc:{}, prec:{}, recall:{}, F1:{}".format(acc, prec, rec, F1))

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
        # list of dicts to dict of lists
        stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
        precision, recall, fscore, support = (
            sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0],
                y_true=stats["targets"][:, 0],
                average="macro",
                zero_division=0,
            )
        )
        accuracy = sklearn.metrics.accuracy_score(
            y_pred=stats["predictions_at_t_stop"][:, 0],
            y_true=stats["targets"][:, 0],
        )
        kappa = sklearn.metrics.cohen_kappa_score(
            stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0]
        )

        classification_loss = stats["classification_loss"].mean()
        earliness_reward = stats["earliness_reward"].mean()
        earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))

        stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(
            y_pred=stats["predictions_at_t_stop"][:, 0],
            y_true=stats["targets"][:, 0],
        )
        return np.stack(losses).mean(), stats

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
