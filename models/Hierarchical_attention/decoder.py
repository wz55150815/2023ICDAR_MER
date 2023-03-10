import torch
import torch.nn as nn
from models.Hierarchical_attention.attention import Attention
from dataset import tokenizer


class SAN_decoder(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channels']
        # 加30是给dictionary中没有的字符预留空间
        self.word_num = params["max_token_num"]
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.struct_num = params['struct_num']
        self.words = tokenizer

        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

        self.threshold = params['hybrid_tree']['threshold']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)

        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size, padding_idx=tokenizer.pad_id)

        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        # structure gru
        self.struc_input_gru = nn.GRUCell(self.input_size, self.hidden_size)

        # attention
        self.word_attention = Attention(params)

        # state to word/struct
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_embedding_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.word_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        self.struct_convert = nn.Linear(self.hidden_size // 2, self.struct_num)

        """ child to parent """
        self.c2p_input_gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.c2p_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        self.c2p_attention = Attention(params)

        self.c2p_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_word_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_relation_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.c2p_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        self.dropout = nn.Dropout(params['dropout_ratio']) if params['dropout'] else nn.Identity()

    def forward(self, cnn_features, labels, images_mask, labels_mask, is_train=True):

        batch_size, num_steps, _ = labels.shape
        height, width = cnn_features.shape[2:]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        struct_probs = torch.zeros((batch_size, num_steps, self.struct_num)).to(device=self.device)
        c2p_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
        c2p_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        c2p_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)

        if is_train:
            # parent_hiddens存储见ppt，word_alpha_sums的存储方式与parent_hiddens相同
            parent_hiddens = torch.zeros((batch_size * (num_steps + 1), self.hidden_size)).to(device=self.device)
            parent_hiddens[:batch_size, :] = self.init_hidden(cnn_features, images_mask)
            c2p_hidden = torch.zeros((batch_size, self.hidden_size)).to(device=self.device)
            word_alpha_sums = torch.zeros((batch_size * (num_steps + 1), 1, height, width)).to(device=self.device)

            for i in range(num_steps):
                parent_ids = labels[:, i, 2].clone()  # 每个batch的parent_ids，如[0, 0, ...]
                for item in range(len(parent_ids)):
                    parent_ids[item] = parent_ids[item] * batch_size + item  # 每个时间步的parent_ids
                parent_hidden = parent_hiddens[parent_ids, :]  # 取出上一时间步的parent_hidden
                word_alpha_sum = word_alpha_sums[parent_ids, :, :, :]

                # 给一个batch的parent_words嵌入维度，这里全部是真实值，可以考虑变成概率(随机真实值或者是模型上一次的预测结果)

                word_embedding = self.embedding(labels[:, i, 3])

                # word
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, word_hidden_first,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                if i != num_steps - 1:
                    parent_hiddens[(i + 1) * batch_size:(i + 2) * batch_size, :] = hidden
                    word_alpha_sums[(i + 1) * batch_size:(i + 2) * batch_size, :, :, :] = word_alpha_sum

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                """ child to parent """
                child_embedding = self.embedding(labels[:, -(i + 1), 1])
                relation = labels[:, -(i + 1), 3].clone()  # parent_tokens
                for num in range(relation.shape[0]):
                    if labels[num, -(i + 1), 1] == self.words.struct_id:  # 每个batch的children判断是否为结构字符struct
                        relation[num] = self.words.struct_id

                    # 不是right...等字符且不是<sos>开始符
                    elif relation[num].item() not in tuple(self.words.struct_ids.values()) and relation[num].item() != self.words.sos_id:
                        relation[num] = self.words.right_id

                relation_embedding = self.embedding(relation)

                c2p_hidden_first = self.c2p_input_gru(torch.cat((child_embedding, relation_embedding), dim=1),
                                                      c2p_hidden)
                c2p_context_vec, c2p_alpha, c2p_alpha_sum = self.c2p_attention(cnn_features, c2p_hidden_first,
                                                                               c2p_alpha_sum, images_mask)
                c2p_hidden = self.c2p_out_gru(word_context_vec, word_hidden_first)

                c2p_state = self.c2p_state_weight(c2p_hidden)
                c2p_weighted_word = self.c2p_word_weight(child_embedding)
                c2p_weighted_relation = self.c2p_relation_weight(relation_embedding)
                c2p_context_weighted = self.c2p_context_weight(c2p_context_vec)

                word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                c2p_out_state = self.dropout(c2p_state + c2p_weighted_word + c2p_weighted_relation + c2p_context_weighted)

                word_prob = self.word_convert(word_out_state)  # 父推子prob
                struct_prob = self.struct_convert(word_out_state)  # 父推子struct
                c2p_prob = self.c2p_convert(c2p_out_state)  # 子推父prob

                word_probs[:, i] = word_prob
                struct_probs[:, i] = struct_prob
                c2p_probs[:, -(i + 1)] = c2p_prob
                word_alphas[:, i] = word_alpha
                c2p_alphas[:, -(i + 1)] = c2p_alpha

        else:
            word_embedding = self.embedding(torch.ones(batch_size).long().to(device=self.device))
            word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            struct_list = []
            parent_hidden = self.init_hidden(cnn_features, images_mask)
            for i in range(num_steps):

                # word
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, word_hidden_first,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                word_out_state = current_state + word_weighted_embedding + word_context_weighted
                word_prob = self.word_convert(word_out_state)

                word_probs[0][i, :] = word_prob
                word_alphas[:, i, :, :] = word_alpha

                _, word = word_prob.max(1)

                if word.item() == self.words.struct_id:  # 2

                    struct_prob = self.struct_convert(word_out_state)
                    struct_probs[0][i, :] = struct_prob

                    structs = torch.sigmoid(struct_prob)
                    struct_tuple = tuple(tokenizer.struct_ids.values())
                    for num in range(structs.shape[1] - 1, -1, -1):
                        if structs[0][num].item() > self.threshold:
                            struct_list.append((struct_tuple[num], hidden, word_alpha_sum))

                    if len(struct_list) == 0:
                        break
                    word, parent_hidden, word_alpha_sum = struct_list.pop()
                    word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))

                elif word == self.words.sos_id:
                    if len(struct_list) == 0:
                        break
                    word, parent_hidden, word_alpha_sum = struct_list.pop()
                    word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))

                else:
                    word_embedding = self.embedding(word)
                    parent_hidden = hidden.clone()

        return word_probs, struct_probs, word_alphas, None, c2p_probs, c2p_alphas

    def init_hidden(self, features, feature_mask):

        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)

        return torch.tanh(average)
