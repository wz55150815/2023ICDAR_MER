import torch.nn as nn
import models
import torch
from einops import rearrange
from dataset import tokenizer


class Backbone(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        self.encoder = getattr(models, params['encoder']['net'])(params=self.params)
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):

        cnn_features = self.encoder(images)
        word_probs, struct_probs, words_alphas, struct_alphas, c2p_probs, c2p_alphas = self.decoder(cnn_features,
                                                                                                    labels, images_mask,
                                                                                                    labels_mask,
                                                                                                    is_train=is_train)

        # word_average_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels[:, :, 1].view(-1))
        word_average_loss = self.cross_entropy_loss(rearrange(word_probs, "b l e->(b l) e"),
                                                    rearrange(labels[:, :, 1], "b l->(b l)"))

        struct_probs = torch.sigmoid(struct_probs)
        # 这里使用bce loss而不用交叉熵是因为后面的struct部分是已经固定好的，所以只要预测是0还是1即可
        struct_average_loss = self.bce_loss(struct_probs, labels[:, :, 4:].float())
        if labels_mask is not None:
            # struct_average_loss = (struct_average_loss * labels_mask[:, :, 0][:, :, None]).sum() / (
            #         labels_mask[:, :, 0].sum() + 1e-10)
            struct_average_loss = struct_average_loss[labels_mask[:, :, 0].bool()].sum() / (labels_mask[:, :, 0].sum() + 1e-10)

        if is_train:
            parent_average_loss = self.cross_entropy_loss(c2p_probs.contiguous().view(-1, word_probs.shape[-1]),
                                                          labels[:, :, 3].view(-1))
            # parent_average_loss = self.cross_entropy_loss(rearrange(c2p_probs, "b l e->(b l) e"),
            #                                               rearrange(labels[:, :, 3], "b l->(b l)"))

            # 作者原先将c2p_alphas, words_alphas的传入顺序写反了
            kl_average_loss = self.cal_kl_loss(c2p_alphas, words_alphas, labels,
                                               images_mask[:, :, ::self.ratio, ::self.ratio], labels_mask)

            return (word_probs, struct_probs), (
                word_average_loss, struct_average_loss, parent_average_loss, kl_average_loss)

        return (word_probs, struct_probs), (word_average_loss, struct_average_loss)

    def cal_kl_loss(self, child_alphas, parent_alphas, labels, image_mask, label_mask):
        batch_size, steps, height, width = child_alphas.shape
        device = self.params["device"]
        new_child_alphas = torch.zeros((batch_size, steps, height, width)).to(device)
        # 子节点向父节点解析的注意力得分,切掉最后一层是因为最后为eos_token到上一层，而parent没有eos_token，
        # 下面parent_alphas[:, 1:, :, :]切掉sos_token同理，new_child_alphas存储数据索引的方式和ppt中的parent-hiddens相同
        new_child_alphas[:, 1:, :, :] = child_alphas[:, :-1, :, :].clone()

        new_child_alphas = rearrange(new_child_alphas, " b s h w->(b s) h w")
        parent_ids = labels[:, :, 2] + steps * torch.arange(batch_size)[:, None].to(device)

        # 下面代码的索引取值，parent_ids在pytorch内部会自动进行reshape操作，即parent_dis = parent_ids.reshape(-1)
        new_child_alphas = new_child_alphas[parent_ids]
        new_child_alphas = new_child_alphas[:, 1:, :, :]
        new_parent_alphas = parent_alphas[:, 1:, :, :]

        # KL_alpha = new_child_alphas * (
        #         torch.log(new_child_alphas + 1e-10) - torch.log(new_parent_alphas + 1e-10)) * image_mask
        # KL_loss = (KL_alpha.sum(-1).sum(-1) * label_mask[:, :-1, 0]).sum(-1).sum(-1) / (label_mask.sum() - batch_size)
        KL_alpha = new_parent_alphas * (torch.log(new_parent_alphas + 1e-10) - torch.log(new_child_alphas + 1e-10)) * image_mask
        indices = label_mask[:, :-1, 0].nonzero().t()
        indices_x, indices_y = indices[0], indices[1]
        KL_loss = (KL_alpha.sum(-1).sum(-1)[indices_x, indices_y]).sum() / (label_mask.sum() - batch_size)

        return KL_loss
