import torch
import torch.nn as nn
from infer.attention import Attention
from dataset import tokenizer


class SAN_decoder(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channels']
        self.word_num = params["max_token_num"]
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.struct_num = params['struct_num']
        self.struct_dict = tuple(tokenizer.struct_ids.values())

        ##################################################################################
        self.inside_tokens = [r'\sqrt', r"\textcircled", r"\boxed"]
        self.above_tokens = [r"\xlongequal", r'\xrightarrow']
        self.below_tokens = [r"\overline", r"\widehat", r"\dot", r"\overrightarrow"]
        ##################################################################################

        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet']['conv1_stride']

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

    def forward(self, cnn_features, images_mask):

        height, width = cnn_features.shape[2:]
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_alpha_sum = torch.zeros((1, 1, height, width)).to(device=self.device)

        if False:
            ...

        else:
            word_embedding = self.embedding(torch.ones(1).long().to(device=self.device))
            struct_list = []
            parent_hidden = self.init_hidden(cnn_features, images_mask)

            prediction = ''
            right_brace = 0  # 右括号的数量
            cid, pid = 0, 0
            p_re = 'Start'
            # word = torch.LongTensor([1])
            word = tokenizer.sos_id
            result = [['<s>', 0, -1, 'root']]

            while len(prediction) < 400:

                # word
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, word_hidden_first,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                word_prob = self.word_convert(word_out_state)
                p_word = word
                word = word_prob.max(-1)[-1]
                if word.item() and word.item() != tokenizer.struct_id:
                    cid += 1
                    p_id = cid
                    result.append([tokenizer.words_index_dict[word.item()], cid, pid, p_re])
                    prediction = prediction + tokenizer.words_index_dict[word.item()] + ' '

                # 当预测文字为结构符
                if word.item() == tokenizer.struct_id:

                    struct_prob = self.struct_convert(word_out_state)

                    structs = torch.sigmoid(struct_prob)

                    # 这里倒过来非常的巧妙
                    for num in range(structs.shape[1]-1, -1, -1):  # 6,5,4,3,2,1,0
                        if structs[0][num] > self.threshold:
                            struct_list.append((self.struct_dict[num], hidden, p_word, p_id, word_alpha_sum))
                    if len(struct_list) == 0:
                        break
                    else:
                        # list.pop()除了删除列表的最后一个元素外，还会返回列表的最后一个元素
                        word, parent_hidden, p_word, pid, word_alpha_sum = struct_list.pop()
                        word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))
                    if word == tokenizer.struct_ids["sub"] or (word == tokenizer.struct_ids["below"] and p_word.item() == tokenizer.sum_token):
                        prediction = prediction + '_ { '
                        p_re = 'Sub'  # 下标
                        right_brace += 1
                    elif word == tokenizer.struct_ids["sup"] or (word == tokenizer.struct_ids["above"] and p_word.item() == tokenizer.sum_token):
                        p_re = 'Sup'  # 上标
                        prediction = prediction + '^ { '
                        right_brace += 1

                    ##################################################################################
                    # elif word == tokenizer.struct_ids["above"] and p_word.item() == tokenizer.frac_token:
                    elif word == tokenizer.struct_ids["above"] and p_word.item() in self.above_tokens:
                        ##################################################################################

                        p_re = 'Above'
                        prediction = prediction + '{ '
                        right_brace += 1
                    # elif word == tokenizer.struct_ids["below"] and p_word.item() == tokenizer.frac_token:
                    elif word == tokenizer.struct_ids["below"] and p_word.item() == self.below_tokens:
                        p_re = 'Below'
                        prediction = prediction + '{ '
                        right_brace += 1
                    elif word == tokenizer.struct_ids["L-sup"]:
                        p_re = 'l_sup'
                        prediction = prediction + '[ '
                    elif word == tokenizer.struct_ids["inside"]:
                        p_re = 'Inside'
                        prediction = prediction + '{ '
                        right_brace += 1

                elif word == tokenizer.eos_id:
                    if len(struct_list) == 0:
                        if right_brace != 0:
                            for _ in range(right_brace):
                                prediction = prediction + '} '
                        break  # 表示一个结构，例如sum结构解析完
                    word, parent_hidden, p_word, pid, word_alpha_sum = struct_list.pop()
                    word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))
                    if word == tokenizer.struct_ids["inside"]:
                        prediction = prediction + '] { '
                        right_brace += 1
                        p_re = 'Inside'
                    elif word == tokenizer.struct_ids["sub"] or (word == tokenizer.struct_ids["below"] and p_word.item() == tokenizer.sum_token):
                        p_re = 'Sub'
                        prediction += '} '
                        right_brace -= 1
                        if right_brace != 0:
                            for num in range(right_brace):
                                prediction += '} '
                                right_brace -= 1
                        prediction = prediction + '_ { '
                        right_brace += 1
                    elif word == tokenizer.struct_ids["sup"] or (word == tokenizer.struct_ids["above"] and p_word.item() == tokenizer.sum_token):
                        p_re = 'Sup'
                        prediction += '} '
                        right_brace -= 1
                        if right_brace != 0:
                            for num in range(right_brace):
                                prediction += '} '
                                right_brace -= 1
                        prediction = prediction + '^ { '
                        right_brace += 1

                    ######################################################################################
                    # elif word == tokenizer.struct_ids["above"] and p_word.item() == tokenizer.frac_token:
                    elif word == tokenizer.struct_ids["above"] and p_word.item() in self.above_tokens:
                        ##################################################################################

                        p_re = 'Above'
                        prediction += '} '
                        right_brace -= 1
                        if right_brace != 0:
                            for num in range(right_brace):
                                prediction += '} '
                                right_brace -= 1
                        prediction = prediction + '{ '
                        right_brace += 1

                    #######################################################################################
                    # elif word == tokenizer.struct_ids["below"] and p_word.item() == tokenizer.frac_token:
                    elif word == tokenizer.struct_ids["below"] and p_word.item() == self.below_tokens:
                        ##################################################################################

                        p_re = 'Below'
                        prediction += '} '
                        right_brace -= 1
                        if right_brace != 0:
                            for num in range(right_brace):
                                prediction += '} '
                                right_brace -= 1
                        prediction = prediction + '{ '
                        right_brace += 1
                    elif word == tokenizer.struct_ids["L-sup"]:
                        p_re = 'l_sup'
                        prediction = prediction + '[ '
                    elif word == tokenizer.struct_ids["inside"]:
                        p_re = 'Inside'
                        prediction = prediction + '] { '
                        right_brace += 1
                    elif word == tokenizer.struct_ids["right"]:
                        p_re = 'Right'
                        prediction = prediction + '} '
                        right_brace -= 1
                else:
                    p_re = 'Right'
                    pid = cid
                    word_embedding = self.embedding(word)
                    parent_hidden = hidden.clone()

        return result

    def init_hidden(self, features, feature_mask):

        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)

        return torch.tanh(average)
