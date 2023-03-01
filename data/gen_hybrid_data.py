import os
from pathlib import Path

from tqdm import tqdm


class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none'):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op


# label = '..//data_comer//2016//caption.txt'
label = 'train_set_labels.txt'
out = 'train_set_hyb'

try:
    with open(label, encoding="gb18030") as f:
        lines = f.readlines()
except UnicodeError:
    with open(label, encoding="utf-8") as f:
        lines = f.readlines()

fail_num = 0
for line in tqdm(lines):
    # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
    try:  # 不成功写入的先打印出来，不写入训练集train_set_hyb文件中，之后排查原因
        name, *words = line.split()
        name = name.split('.')[0]
        root = Tree(label='root', parent_label='root', parent_id=-1)

        labels = []
        id = 1
        parents = [Tree(label='<sos>', id=0)]
        parent = Tree(label='<sos>', id=0)

        for i in range(len(words)):
            if words[i] == '\\limits':
                continue
            if i == 0 and words[i] in ['_', '^', '{', '}']:
                print(name)
                fail_num += 1
                break

            # 对于包含"above", "below"两个结构结构符的可以仿照下面的进行处理
            elif words[i] == '{':
                if words[i - 1] == '\\frac':
                    labels.append([id, 'struct', parent.id, parent.label])
                    parents.append(Tree('\\frac', id=parent.id, op='above'))
                    id += 1
                    parent = Tree('above', id=parents[-1].id + 1)
                elif words[i - 1] == '}' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                    parent = Tree('below', id=parents[-1].id + 1)
                    parents[-1].op = 'below'

                #############################################################################################
                # elif words[i - 1] == '\sqrt':
                #     labels.append([id, 'struct', parent.id, '\sqrt'])
                #     parents.append(Tree('\sqrt', id=parent.id))
                #     parent = Tree('inside', id=id)
                #     id += 1

                # 改为：
                elif words[i - 1] in [r'\sqrt', r"\textcircled", r"\boxed"]:
                    labels.append([id, 'struct', parent.id, words[i - 1]])
                    parents.append(Tree(words[i - 1], id=parent.id))
                    parent = Tree('inside', id=id)
                    id += 1

                elif words[i - 1] in [r"\overline", r"\widehat", r"\dot", r"\overrightarrow"]:
                    labels.append([id, 'struct', parent.id, words[i - 1]])
                    parents.append(Tree(words[i - 1], id=parent.id))
                    parent = Tree('below', id=id)
                    id += 1

                # xrightarrow若有下面的内容，会先[]在{}，若直接{，说明就只有above的内容
                elif words[i - 1] in [r"\xlongequal", r'\xrightarrow']:
                    labels.append([id, 'struct', parent.id, words[i - 1]])
                    parents.append(Tree(words[i - 1], id=parent.id))
                    parent = Tree('above', id=id)
                    id += 1
                #############################################################################################

                elif words[i - 1] == ']' and parents[-1].label == '\sqrt':
                    parent = Tree('inside', id=parents[-1].id + 1)

                #############################################################################################
                elif words[i - 1] == ']' and parents[-1].label == r'\xrightarrow':
                    parent = Tree('above', id=parents[-1].id + 1)
                #############################################################################################

                elif words[i - 1] == '^':
                    if words[i - 2] != '}':
                        if words[i - 2] == '\sum':
                            labels.append([id, 'struct', parent.id, parent.label])
                            parents.append(Tree('\sum', id=parent.id))
                            parent = Tree('above', id=id)
                            id += 1

                        else:
                            labels.append([id, 'struct', parent.id, parent.label])
                            parents.append(Tree(words[i - 2], id=parent.id))
                            parent = Tree('sup', id=id)
                            id += 1

                    else:
                        # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                        if parents[-1].label == '\sum':
                            parent = Tree('above', id=parents[-1].id + 1)
                        else:
                            parent = Tree('sup', id=parents[-1].id + 1)
                        # id += 1

                elif words[i - 1] == '_':
                    if words[i - 2] != '}':
                        if words[i - 2] == '\sum':
                            labels.append([id, 'struct', parent.id, parent.label])
                            parents.append(Tree('\sum', id=parent.id))
                            parent = Tree('below', id=id)
                            id += 1

                        else:
                            labels.append([id, 'struct', parent.id, parent.label])
                            parents.append(Tree(words[i - 2], id=parent.id))
                            parent = Tree('sub', id=id)
                            id += 1

                    else:
                        # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                        if parents[-1].label == '\sum':
                            parent = Tree('below', id=parents[-1].id + 1)
                        else:
                            parent = Tree('above', id=parents[-1].id + 1)
                        # id += 1

                else:
                    print('unknown word before {', name, i, words[i])
                    raise ValueError  # 这里直接改成报错方便检查

            elif words[i] == '[' and words[i - 1] == '\sqrt':
                labels.append([id, 'struct', parent.id, '\sqrt'])
                parents.append(Tree('\sqrt', id=parent.id))
                parent = Tree('L-sup', id=id)
                id += 1

            elif words[i] == ']' and parents[-1].label == '\sqrt':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

            ##############################################################
            # ->下面有内容的情况(用[]括住内容)，仿照的是sqrt
            elif words[i] == '[' and words[i - 1] == r'\xrightarrow':
                labels.append([id, 'struct', parent.id, r'\xrightarrow'])
                parents.append(Tree(words[i - 1], id=parent.id))
                parent = Tree('below', id=id)
                id += 1

            elif words[i] == ']' and parents[-1].label == r'\xrightarrow':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1
            ###############################################################

            elif words[i] == '}':
                # 右括号的前面如果不是右括号，则直接加上一个eos表示一个struct的结束
                if words[i - 1] != '}':
                    labels.append([id, '<eos>', parent.id, parent.label])
                    id += 1

                if i + 1 < len(words) and words[i + 1] == '{' and parents[-1].label == '\\frac' and parents[
                    -1].op == 'above':
                    continue
                if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                    continue
                elif i + 1 < len(words) and words[i + 1] != '}':
                    parent = Tree('right', id=parents[-1].id + 1)

                parents.pop()

            else:
                if words[i] in ['^', '_']:
                    continue
                labels.append([id, words[i], parent.id, parent.label])
                parent = Tree(words[i], id=id)
                id += 1

        parent_dict = {0: []}
        for i in range(len(labels)):
            parent_dict[i + 1] = []
            parent_dict[labels[i][2]].append(labels[i][3])

        # with open(f'{out}/{name}.txt', 'a') as f:  # 追加模式
        with open(f'{out}/{name}.txt', 'w') as f:  # 重新写模式
            for line in labels:
                id, label, parent_id, parent_label = line
                if label != 'struct':
                    f.write(f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')
                else:
                    tem = f'{id}\t{label}\t{parent_id}\t{parent_label}'
                    tem = tem + '\tabove' if 'above' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tbelow' if 'below' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tsub' if 'sub' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tsup' if 'sup' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tL-sup' if 'L-sup' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tinside' if 'inside' in parent_dict[id] else tem + '\tNone'
                    tem = tem + '\tright' if 'right' in parent_dict[id] else tem + '\tNone'
                    f.write(tem + '\n')
            if label != '<eos>':
                f.write(f'{id + 1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n')

    except ValueError:
        fail_num += 1


print("total write fail num:", fail_num)