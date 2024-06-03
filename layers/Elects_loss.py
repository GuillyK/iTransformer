import torch
from torch import nn


class EarlyRewardLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=10, weight=None):
        super(EarlyRewardLoss, self).__init__()

        self.negative_log_likelihood = nn.NLLLoss(
            reduction="none", weight=weight
        )
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(
        self,
        log_class_probabilities,
        probability_stopping,
        y_true,
        return_stats=False,
    ):
        N, T, C = log_class_probabilities.shape
        y_reverse_one_hot = y_true.argmax(dim=-1)
        y_max_extended = y_reverse_one_hot.unsqueeze(-1).repeat_interleave(9, dim=-1)

        y_true = y_true.unsqueeze(1).repeat(1, T, 1)

        # equation 3
        Pt = calculate_probability_making_decision(probability_stopping)

        # equation 7 additive smoothing
        Pt = Pt + self.epsilon / T

        # equation 6, right term
        t = torch.ones(
            N, T, device=log_class_probabilities.device
        ) * torch.arange(T).type(torch.FloatTensor).to(
            log_class_probabilities.device
        )
        # print(y_true)
        prob_class = probability_correct_class(log_class_probabilities, y_true)
        earliness_reward = (
            Pt
            * prob_class
            * (1 - t / T)
        )
        earliness_reward = earliness_reward.sum(1).mean(0)
        y_true_indices = y_true.argmax(dim=-1)

        # equation 6 left term
        # print(f"N: {N}, T: {T}, log_class_probabilities size: {log_class_probabilities.size()}, y_true size: {y_max_extended.size()}")
        cross_entropy = self.negative_log_likelihood(
            log_class_probabilities.view(N * T, C), y_max_extended.view(N * T)
        ).view(N, T)
        classification_loss = (cross_entropy * Pt).sum(1).mean(0)

        # equation 6
        loss = (
            self.alpha * classification_loss
            - (1 - self.alpha) * earliness_reward
        )

        if return_stats:
            stats = dict(
                classification_loss=classification_loss.cpu().detach().numpy(),
                earliness_reward=earliness_reward.cpu().detach().numpy(),
                probability_making_decision=Pt.cpu().detach().numpy(),
            )
            return loss, stats
        else:
            return loss


def calculate_probability_making_decision(deltas):
    """
    Equation 3: probability of making a decision

    :param deltas: probability of stopping at each time t
    :return: comulative probability of having stopped
    """
    batchsize, sequencelength = deltas.shape

    pts = list()

    initial_budget = torch.ones(batchsize, device=deltas.device)

    budget = [initial_budget]
    for t in range(1, sequencelength):
        pt = deltas[:, t] * budget[-1]
        budget.append(budget[-1] - pt)
        pts.append(pt)

    # last time
    pt = budget[-1]
    pts.append(pt)

    return torch.stack(pts, dim=-1)


def probability_correct_class(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape

    # eye = (
    #     torch.eye(nclasses).type(torch.ByteTensor).to(logprobabilities.device)
    # )

    # targets_one_hot = eye[targets]
    # todo Maybe change data_y in the dataloader back to a sequence
    # print(f"{targets.shape=}")
    # targets = targets.unsqueeze(1).repeat(1, seqquencelength, 1)
    # print(f"{targets=}")
    # print(targets.bool())
    mask = targets.bool()
    # print(mask)
    # check if every value in the second axis is false
    indices = mask[:, :, :].all(dim=1).nonzero()
    # indices = (is_all_false == False).nonzero(as_tuple=True)
    numbers = indices[:, 0]

    # Create a tensor of all numbers from 0 to 31
    all_numbers = torch.arange(batchsize)

    # Find the missing numbers
    missing_numbers = torch.tensor(
        [num for num in all_numbers if num not in numbers]
    )

    # print(f"{missing_numbers=}")

    for mis_num in missing_numbers:
        mask[mis_num, :, 0] = True
        # logprobabilities[mis_num, :, :] = 0
    # implement the y*\hat{y} part of the loss function
    # print(logprobabilities.shape, targets.shape)
    y_haty = torch.masked_select(logprobabilities, mask)
    # print("sequencelength", seqquencelength, y_haty.shape)
    # print(f"{seqquencelength=}")
    # print(y_haty)
    return y_haty.view(batchsize, seqquencelength).exp()
