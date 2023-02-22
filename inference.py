import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm
from pathlib import Path
from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import tokenizer
from typing import Union

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--config', default='config.yaml', type=str, help='配置文件路径')
parser.add_argument('--image_path', default=r'.\data_inference\data\2014\img', type=str,
                    help='测试image路径')
parser.add_argument('--label_path', default=r'.\data_inference\data\2014\caption.txt', type=str,
                    help='测试label路径')
args = parser.parse_args()

if not args.config:
    print('请提供config yaml路径！')
    exit(-1)

"""加载config文件"""
params = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

words = tokenizer
params['word_num'] = tokenizer.vocab_size
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)
checkpoints_root_path = Path("checkpoints")
if (checkpoints_root_path / "model.pkl").exists():
    # load_checkpoint(model, None, sorted(checkpoints_root_path.glob('*'))[-1])
    model.load_state_dict(torch.load(checkpoints_root_path / "model.pkl"))

model.eval()
bad_case = {}


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        try:
            return [gtd_list[nodeid][0]]
        except IndexError:
            return
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string


def inference(photo_root_path: Union[Path, str]) -> None:
    """
        Args:
             photo_root_path:
             photo_root_path
               ├── photo1.('.jpg', '.jpeg', '.png', '.bmp')
               ├── photo2.('.jpg', '.jpeg', '.png', '.bmp')
               └── ...
    """
    assert isinstance(photo_root_path, (Path, str))
    photo_root_path = Path(photo_root_path) if isinstance(photo_root_path, str) else photo_root_path
    assert photo_root_path.is_dir(), "文件夹内没有图片"
    for i, file_path in enumerate(photo_root_path.glob('*')):
        if file_path.is_file() and file_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(str(file_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image[None, None, :, :]
            image_mask = torch.ones(image.shape)
            image, image_mask = image.to(device), image_mask.to(device)
            prediction = model(image, image_mask)
            latex_list = convert(1, prediction)
            print(file_path.name, ":", ' '.join(latex_list))
            if i == 30:
                break


def model_eval():
    word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    with open(args.label_path) as f:
        labels = f.readlines()
    with torch.no_grad():
        labels_num = 0
        for item in tqdm(labels):
            name, *label = item.split()
            label = ' '.join(label)
            if name.endswith('.jpg'):
                name = name.split('.')[0]
            img = cv2.imread(os.path.join(args.image_path, name + '.bmp'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image[None, None, :, :]

            image_mask = torch.ones(image.shape)
            image, image_mask = image.to(device), image_mask.to(device)

            prediction = model(image, image_mask)

            latex_list = convert(1, prediction)[:-1]  # 将结束符切掉
            if latex_list is not None:
                labels_num += 1
                latex_string = ' '.join(latex_list)
                if latex_string == label.strip():
                    exp_right += 1
                else:
                    bad_case[name] = {
                        'label': label,
                        'predi': latex_string,
                        'list': prediction
                    }
            else:
                print("错误解析，跳过该条数据的计算")

        print(exp_right / labels_num)


if __name__ == "__main__":
    # model_eval()
    # with open('bad_case.json', 'w') as f:
    #     json.dump(bad_case, f, ensure_ascii=False)
    inference(r".\data\off_image_train")

