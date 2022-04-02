from fastai.vision import *
from fastai.tabular import *
from fastai.text import *
from fastai.basics import *
from fastai.callbacks import *
import torch
import logging


__all__ = ['ImageTabularTextLearner', 'collate_mixed', 'image_tabular_text_learner', 'normalize_custom_funcs']

def out_layer(tordata, outfile):
    import numpy as np
    logging.info('beging save result to {0}'.format(outfile))
    logging.info('data shape is {0}'.format(tordata.shape))
    tordata = tordata.detach().numpy()
    out_size = tordata.shape[-1]
    out = tordata.reshape(-1, out_size)
    np.savetxt(outfile, out)


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        # Conv1d with ks=1 is the same as a linear layer, but avoid permutations
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size = 1, bias = False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size = 1, bias = False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)

    def forward(self, x):                      #  x.shape =  [bs x in_channels x seq_len]
        Q = self.conv_Q(x)                                 # [bs x key_channels x seq_len]
        K = self.conv_K(x)                                 # [bs x key_channels x seq_len]
        V = self.conv_V(x)                                 # [bs x out_channels x seq_len]
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)        # [bs x seq_len x seq_len]
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)  # [bs x out_channels x seq_len]
        return x, A


class ImageTabularTextModel(nn.Module):
    def __init__(self, n_cont, encoder, vis_out=512, text_out=128, is_save=False):
        """
        :param n_cont: the dim for tabular features, which is 1399 in this work
        :param encoder: nlp model encoder
        :param vis_out: the output image feature dim, default 512
        :param text_out: the output smiles feature dim, default 128
        :param is_save: save the features as file, default false
        """
        super().__init__()
        self.tab_inf = n_cont
        self.save = is_save
        self.cnn = create_cnn_model(models.resnet50, vis_out)
        self.nlp = SequentialRNN(encoder[0], PoolingLinearClassifier([400 * 3] + [text_out], [.5]))
        self.att = AttentionLayer(1, 1, 96)
        self.fc1 = nn.Sequential(*bn_drop_lin(vis_out + self.tab_inf + text_out, 128, bn=True, p=.5, actn=nn.ReLU()))
        self.fc2 = nn.Sequential(*bn_drop_lin(128, 2, bn=False, p=0.05, actn=nn.Sigmoid()))

    def forward(self, img: Tensor, tab: Tensor, text: Tensor) -> Tensor:
        imgLatent = self.cnn(img)
        # we used the TabularList, just cont values, simply for numpy
        tabLatent = tab[-1]
        textLatent = self.nlp(text)[0]
        # this is for learn.summary(), we only use the cont in TabularList,
        # when summary, it may have error shape for tab features
        if len(tab) == 1:
            tabLatent = torch.rand(1, self.tab_inf)
        cat_feature = torch.cat([F.relu(imgLatent), F.relu(tabLatent), F.relu(textLatent)], dim=1)
        cat_feature = cat_feature.reshape(cat_feature.size(0), 1, -1)
        res, att = self.att(cat_feature)
        res = res.reshape(res.size(0), -1)
        pred = self.fc2(self.fc1(res))
        return pred

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def collate_mixed(samples, pad_idx: int = 0):
    # Find max length of the text from the MixedItemList
    max_len = max([len(s[0].data[2]) for s in samples])

    for s in samples:
        res = np.zeros(max_len + pad_idx, dtype=np.int64)
        res[:len(s[0].data[2])] = s[0].data[2]
        s[0].data[2] = res

    return data_collate(samples)


def split_layers(model: nn.Module) -> List[nn.Module]:
    groups = [[model.cnn]]
    groups += [[model.nlp]]
    groups += [[model.att]]
    groups += [[model.fc1]]
    groups += [[model.fc2]]
    return groups

def _normalize_images_batch(b: Tuple[Tensor, Tensor], mean: FloatTensor, std: FloatTensor) -> Tuple[Tensor, Tensor]:
    x, y = b
    mean, std = mean.to(x[0].device), std.to(x[0].device)
    x[0] = normalize(x[0], mean, std)
    return x, y


def normalize_custom_funcs(mean: FloatTensor, std: FloatTensor, do_x: bool = True, do_y: bool = False) -> Tuple[
    Callable, Callable]:
    mean, std = tensor(mean), tensor(std)
    return (partial(_normalize_images_batch, mean=mean, std=std),
            partial(denormalize, mean=mean, std=std))


class RNNTrainerSimple(LearnerCallback):
    def __init__(self, learn: Learner, alpha: float = 0., beta: float = 0.):
        super().__init__(learn)
        self.not_min += ['raw_out', 'out']
        self.alpha, self.beta = alpha, beta

    def on_epoch_begin(self, **kwargs):
        self.learn.model.reset()


class ImageTabularTextLearner(Learner):
    def __init__(self, data: DataBunch, model: nn.Module, **learn_kwargs):
        super().__init__(data, model, **learn_kwargs)
        alpha: float = 2.
        beta: float = 1.
        self.callbacks.append(RNNTrainerSimple(self, alpha=alpha, beta=beta))
        self.split(split_layers)

def image_tabular_text_learner(data, len_cont_names, nlp_cls, vis_out=512, text_out=128, is_save=False):
    """
    :param data: data for train the model
    :nlp_cls: nlp classfiy file
    :param len_cont_names: feature dim for the tabular, 1399 in this work
    :param vis_out: the output image feature dim, default 512
    :param text_out: the output smiles feature dim, default 128
    :param is_save: save the features as file, default false
    :return:
    """

    l = text_classifier_learner(nlp_cls, AWD_LSTM, drop_mult=0.5)
    l.load_encoder('text_encoder')

    model = ImageTabularTextModel(len_cont_names, l.model, vis_out, text_out, is_save)
    opt_func = partial(optim.Adam, lr=3e-5, betas=(0.9,0.99), weight_decay=0.1, amsgrad=True)
    loss_func = CrossEntropyFlat()
    learn = ImageTabularTextLearner(
        data,
        model,
        opt_func = opt_func,
        loss_func = loss_func,
        metrics=[accuracy]
    )
    callbacks = [
        EarlyStoppingCallback(learn, min_delta=1e-5, patience=4),
        SaveModelCallback(learn)
    ]
    learn.callbacks.extend(callbacks)
    return learn

