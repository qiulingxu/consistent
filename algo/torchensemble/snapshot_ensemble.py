"""
  Snapshot ensemble generates many base estimators by enforcing a base
  estimator to converge to its local minima many times and save the
  model parameters at that point as a snapshot. The final prediction takes
  the average over predictions from all snapshot models.

  Reference:
      G. Huang, Y.-X. Li, G. Pleiss et al., Snapshot Ensemble: Train 1, and
      M for free, ICLR, 2017.
"""


import math
import torch
import logging
import warnings
import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from .utils.logging import get_tb_logger


__all__ = ["SnapshotEnsembleClassifier", "SnapshotEnsembleRegressor"]


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    lr_clip : list or tuple, default=None
        Specify the accepted range of learning rate. When the learning rate
        determined by the scheduler is out of this range, it will be clipped.

        - The first element should be the lower bound of learning rate.
        - The second element should be the upper bound of learning rate.
    epochs : int, default=100
        The number of training epochs.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`DataLoader` container that contains the evaluating data.

        - If ``None``, no validation is conducted after each snapshot
          being generated.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each snapshot being generated.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble with
          ``n_estimators`` base estimators will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _snapshot_ensemble_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different snapshot ensemble
    models.
    """

    def get_doc(item):
        """Return selected item"""
        __doc = {"fit": __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc

def evaluate_uncertainty_part(prev_model, model, dataexpt, learnexpt, train_loader, val_loader, test_loader ):
    """ datamode in train sample or expectation,
        learnmode in empirical learning or expect learner
    """
    model.eval()
    assert dataexpt in [True, False]
    assert learnexpt in [True, False]
    if dataexpt:
        loader = train_loader
    else:
        loader = test_loader
    if learnexpt:
        def pred(ensemble_model, x, y, correct):
            ret = ensemble_model._forward_sep(x)
            ret = T.transpose(T.argmax(ret, dim=2), 0, 1)
            acc = T.eq(ret, T.unsqueeze(y, -1))
            prev_acc = T.unsqueeze(correct, -1)
            consistency = T.sum(acc * prev_acc) / (T.sum(prev_acc) * acc.size(1) +1e-5)
            return consistency
    else:
        def pred(ensemble_model, x, y, correct):
            ret = ensemble_model.forward(x)
            #ret = F.softmax(ret, dim=2)
            ret = T.argmax(ret, dim=1)
            acc = T.eq(ret, y)
            consistency = T.sum(acc * correct) / (T.sum(correct) +1e-5)
            return consistency            
    with T.no_grad():
        cnt = 0 
        tot = 0
        for data in loader:
            if dataexpt:
                x, y, _ = data
            else:
                x, y = data
            x = x.cuda()
            y= y.cuda()        
            ret = prev_model.forward(x)
            ret = T.argmax(ret, dim=1)
            prev_acc = T.eq(ret, y)
            c = pred(model, x, y, prev_acc)
            tot += c * x.size(0)
            cnt += x.size(0)
    return tot / cnt

def evaluate_uncertainty(model, dataloader):
    model.eval()
    cnt =0 
    div = 0
    with T.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            y= y.cuda()
            ret = model._forward_sep(x)
            ret = F.softmax(ret, dim=2)
            m = T.mean(ret,dim=0,keepdim=True)
            div += T.sum(T.mean(T.square(ret - m), dim=0, keepdim=True))
            cnt += ret.size(1)
    return div / cnt


def evaluate_consistency(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    cnt = 0 
    consistent_sample = 0 
    div = 0
    with T.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            y= y.cuda()
            ret1 = model1(x)
            ret2 = model2(x)
            assert len(ret1.size()) == 2
            ret1 = T.argmax(ret1, dim=1)
            ret2 = T.argmax(ret2, dim=1)
            cnt_sample = ret1.eq(y)
            cnt += cnt_sample.float().sum()
            consistent_sample += T.logical_and(cnt_sample, ret1.eq(ret2)).float().sum()

    return consistent_sample * 1.0 / cnt

class _BaseSnapshotEnsemble(BaseModule):
    def __init__(
        self, estimator, n_estimators, estimator_args=None, cuda=True
    ):
        super(BaseModule, self).__init__()

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args

        if estimator_args and not isinstance(estimator, type):
            msg = (
                "The input `estimator_args` will have no effect since"
                " `estimator` is already an object after instantiation."
            )
            warnings.warn(msg, RuntimeWarning)

        self.device = torch.device("cuda" if cuda else "cpu")
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()

    def _validate_parameters(self, lr_clip, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if lr_clip:
            if not (isinstance(lr_clip, list) or isinstance(lr_clip, tuple)):
                msg = "lr_clip should be a list or tuple with two elements."
                self.logger.error(msg)
                raise ValueError(msg)

            if len(lr_clip) != 2:
                msg = (
                    "lr_clip should only have two elements, one for lower"
                    " bound, and another for upper bound."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            if not lr_clip[0] < lr_clip[1]:
                msg = (
                    "The first element = {} should be smaller than the"
                    " second element = {} in lr_clip."
                )
                self.logger.error(msg.format(lr_clip[0], lr_clip[1]))
                raise ValueError(msg.format(lr_clip[0], lr_clip[1]))

        if not epochs > 0:
            msg = (
                "The number of training epochs = {} should be strictly"
                " positive."
            )
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = (
                "The number of batches to wait before printting the"
                " training status should be strictly positive, but got {}"
                " instead."
            )
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

        if not epochs % self.n_estimators == 0:
            msg = (
                "The number of training epochs = {} should be a multiple"
                " of n_estimators = {}."
            )
            self.logger.error(msg.format(epochs, self.n_estimators))
            raise ValueError(msg.format(epochs, self.n_estimators))

    def _forward(self, *x):
        """
        Implementation on the internal data forwarding in snapshot ensemble.
        """
        # Average
        results = [estimator(*x) for estimator in self.estimators_]
        output = op.average(results)

        return output

    def _forward_sep(self, *x):
        results = [estimator(*x) for estimator in self.estimators_]
        return T.stack(results, dim=0)

    def _clip_lr(self, optimizer, lr_clip):
        """Clip the learning rate of the optimizer according to `lr_clip`."""
        if not lr_clip:
            return optimizer

        for param_group in optimizer.param_groups:
            if param_group["lr"] < lr_clip[0]:
                param_group["lr"] = lr_clip[0]
            if param_group["lr"] > lr_clip[1]:
                param_group["lr"] = lr_clip[1]

        return optimizer

    def _set_scheduler(self, optimizer, n_iters):
        """
        Set the learning rate scheduler for snapshot ensemble.
        Please refer to the equation (2) in original paper for details.
        """
        T_M = math.ceil(n_iters / self.n_estimators)
        lr_lambda = lambda iteration: 0.5 * (  # noqa: E731
            torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return scheduler

    def set_scheduler(self, scheduler_name, **kwargs):
        msg = (
            "The learning rate scheduler for Snapshot Ensemble will be"
            " automatically set. Calling this function has no effect on"
            " the training stage of Snapshot Ensemble."
        )
        warnings.warn(msg, RuntimeWarning)


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleClassifier.""", "seq_model"
)
class SnapshotEnsembleClassifier(_BaseSnapshotEnsemble, BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleClassifier.""",  # noqa: E501
        "classifier_forward",
    )
    def forward(self, *x):
        proba = self._forward(*x)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleClassifier.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        lr_clip=None,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
        loss_func = None,
    ):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        estimator = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator, self.optimizer_name, **self.optimizer_args
        )

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        counter = 0  # a counter on generating snapshots
        total_iters = 0
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        estimator.train()
        for epoch in range(epochs):
            for batch_idx, elem in enumerate(train_loader):
                oinputs, otargets, elemorder = elem
                #data, target = io.split_data_target(elem, self.device)
                batch_size = oinputs.size(0)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                
                if loss_func is None:
                    output = estimator(oinputs, otargets)
                    loss = criterion(output, otargets)
                else:
                    oinputs = oinputs.cuda()
                    otargets = otargets.cuda()
                    loss, _, output, _ = loss_func(oinputs, otargets, elemorder, estimator, epoch)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == otargets).sum().item()

                        msg = (
                            "lr: {:.5f} | Epoch: {:03d} | Batch: {:03d} |"
                            " Loss: {:.5f} | Correct: {:d}/{:d}"
                        )
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                                correct,
                                batch_size,
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "snapshot_ensemble/Train_Loss",
                                loss,
                                total_iters,
                            )
                        else:
                            pass
                            #print("None")

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1
                total_iters += 1

            if counter % n_iters_per_estimator == 0:

                # Generate and save the snapshot
                snapshot = self._make_estimator()
                snapshot.load_state_dict(estimator.state_dict())
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader is not None and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                    acc = 100 * correct / total

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "n_estimators: {} | Validation Acc: {:.3f} %"
                        " | Historical Best: {:.3f} %"
                    )
                    self.logger.info(
                        msg.format(len(self.estimators_), acc, best_acc)
                    )
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "snapshot_ensemble/Validation_Acc",
                            acc,
                            len(self.estimators_),
                        )

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleRegressor.""", "seq_model"
)
class SnapshotEnsembleRegressor(_BaseSnapshotEnsemble, BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleRegressor.""",  # noqa: E501
        "regressor_forward",
    )
    def forward(self, *x):
        pred = self._forward(*x)
        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleRegressor.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        lr_clip=None,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
        loss_func = None,
    ):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        estimator = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator, self.optimizer_name, **self.optimizer_args
        )

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")
        counter = 0  # a counter on generating snapshots
        total_iters = 0
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        estimator.train()
        for epoch in range(epochs):
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                assert False, "elem has been changed to contain 3 parts, fix the following code"
                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                if loss_func is None:
                    output = estimator(*data)
                    loss = criterion(output, target)
                else:
                    loss = loss_func(estimator, data[0], data[1], target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = (
                            "lr: {:.5f} | Epoch: {:03d} | Batch: {:03d}"
                            " | Loss: {:.5f}"
                        )
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "snapshot_ensemble/Train_Loss",
                                loss,
                                total_iters,
                            )

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1
                total_iters += 1

            if counter % n_iters_per_estimator == 0:
                # Generate and save the snapshot
                snapshot = self._make_estimator()
                snapshot.load_state_dict(estimator.state_dict())
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    mse = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        mse += criterion(output, target)
                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "n_estimators: {} | Validation MSE: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(
                        msg.format(len(self.estimators_), mse, best_mse)
                    )
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "snapshot_ensemble/Validation_MSE",
                            mse,
                            len(self.estimators_),
                        )

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
