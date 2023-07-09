import torchmetrics


class Metrics:
    def __init__(self, num_classes):

        self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)

        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)

        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, top_k=1)

        self.precision_macro_metric = torchmetrics.Precision(
            task="multiclass", average="macro", num_classes=num_classes, top_k=1
        )

        self.recall_macro_metric = torchmetrics.Recall(
            task="multiclass", average="macro", num_classes=num_classes, top_k=1
        )

        self.precision_micro_metric = torchmetrics.Precision(
            average="micro", task="multiclass", num_classes=num_classes, top_k=1
        )

        self.recall_micro_metric = torchmetrics.Recall(
            average="micro", task="multiclass", num_classes=num_classes, top_k=1
        )
