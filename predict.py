# coding: UTF-8
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


def predict(config, model, data_iter, test=False):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(config.save_path))
    else:
        model.load_state_dict(torch.load(config.save_path, map_location = torch.device('cpu')))
    model.eval()
    loss_total = 0
    predict_all = None
    # labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            result = torch.max(outputs.data, 1)[1].cpu()
            # print(labels)
            # print(result)
            if predict_all is not None:
                predict_all = torch.cat(tensors=(predict_all, result))
            else:
                predict_all = result

            print(f"predicted {predict_all.shape[0]} records")

    return predict_all.numpy()


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 数据集
    dataset = 'WORDS'

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    _, _, _, pred_data = build_dataset(config)
    pred_iter = build_iterator(pred_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # test
    model = x.Model(config).to(config.device)
    ret = predict(config, model, pred_iter)
    output = pd.DataFrame(ret, columns=["nums"])
    output.to_parquet("ret.parquet")
    # output.to_csv("ret.csv")
