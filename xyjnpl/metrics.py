from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(object):
    def __init__(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def calculate(self, val_predicta, val_targa):
        val_predict = [int(x) for x in val_predicta]
        val_targ = [int(x) for x in val_targa]
        x = 0
        for i in range(len(val_predict)):
            if val_predict[i] == val_targ[i]:
                x = x + 1
        print(x)
        # None, 'micro', 'macro', 'weighted', 'samples'
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        print('macro -val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        _val_f1 = f1_score(val_targ, val_predict, average="micro")
        _val_recall = recall_score(val_targ, val_predict, average="micro")
        _val_precision = precision_score(val_targ, val_predict, average="micro")
        print('micro -val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
