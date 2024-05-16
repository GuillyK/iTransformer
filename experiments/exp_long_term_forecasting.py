from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = SummaryWriter()  # Create a SummaryWriter instance

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

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
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
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, batch_y, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )
                f_dim = -1 if self.args.features == "MS" else 0
                # outputs = outputs[:, -self.args.pred_len :, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                #     self.device
                # )
                outputs = outputs.float()

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        criterion = self._select_criterion()
        scheduler = lr_scheduler.ExponentialLR(model_optim, gamma=0.8)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        preds = []
        trues = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                # for j in sequences:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # batch_y = torch.zeros_like(
                #     batch_y[:, -self.args.pred_len :, :]
                # ).float()
                # batch_y = (
                #     torch.cat(
                #         [batch_y[:, : self.args.label_len, :], batch_y], dim=1
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
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )


                    f_dim = -1 if self.args.features == "MS" else 0
                    # print(outputs.shape, batch_y.shape)
                    # outputs = outputs[:, -self.args.num_classes :, f_dim:]
                    # batch_y = batch_y.float().to(self.device)
                    outputs = outputs.float()
                    # print(outputs)
                    # exit()
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    probabilities = F.softmax(outputs, dim=1)
                    probabilities = probabilities.detach().cpu().numpy()

                    one_hot_prob = np.eye(len(probabilities[0]))[np.argmax(probabilities[0])]
                    true = batch_y.detach().cpu().numpy()
                    preds.append(one_hot_prob)
                    trues.append(true[0])
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
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                self.writer.add_scalar('Loss/train', loss, epoch)
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram('Weights/' + name, param.data, epoch)
                    self.writer.add_histogram('Gradients/' + name, param.grad, epoch)
            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.writer.add_scalar('Loss/vali', vali_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            # log learning rate
            self.writer.add_scalar('Learning Rate', model_optim.param_groups[0]['lr'], epoch)
            acc, conf_matrix, prec, rec, F1 = metric(preds, trues)
            self.writer.add_scalar('Accuracy', acc, epoch)
            self.writer.add_scalar('Precision', prec, epoch)
            self.writer.add_scalar('Recall', rec, epoch)
            self.writer.add_scalar('F1', F1, epoch)
            self.writer.add_histogram('Confusion Matrix', conf_matrix, epoch)
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
        print("testing modelHEYAAAAA")
        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                tqdm(test_loader)
            ):

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
                            batch_x, batch_x_mark, batch_y, batch_y_mark
                        )

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, batch_y, batch_y_mark
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
                one_hot_prob = np.eye(len(probabilities[0]))[np.argmax(probabilities[0])]

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

        acc, conf_matrix, prec, rec, F1 = metric(preds, trues)

        print("acc:{}, prec:{}, recall:{}, F1:{}".format(acc, prec, rec, F1))

        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("acc:{}, prec:{}, recall:{}, F1:{}".format(acc, prec, rec, F1))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(
            folder_path + "metrics.npy",
            np.array([acc, prec, rec, F1]),
        )
        np.save(folder_path + "confusion_matrix.npy", conf_matrix)
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plt.savefig(folder_path + "confusion_matrix.pdf")
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
