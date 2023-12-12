from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision import transforms
from models import build_model
from config import get_config
from PIL import Image
import argparse
import torch
import time


reso = 224


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='./cfgs/agent_deit_s.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        # default=['MODEL.NAME', 'example',
        #          'MODEL.AGENT.VERSION', 1,
        #          'MODEL.AGENT.NUM', '49-49-49-49',
        #          'MODEL.AGENT.ATTN_TYPE', 'BBBB',
        #          'DATA.IMG_SIZE', reso,
        #          'MODEL.SWIN.WINDOW_SIZE', reso // 4,
        #          ],
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    _, config = parse_option()
    model = build_model(config)
    # from temp_utils import load_pretrained
    # load_pretrained('./max_ema_acc.pth', model)
    info = []

    # ---------- 可视化等 ----------
    # model.load_state_dict(torch.load('./agent_deit_t.pth')['model'])
    # model.eval()
    # image = Image.open('./visualize/img_ori_08563.png')
    # t = transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # image = t(image).reshape(1, 3, 224, 224)
    # with torch.no_grad():
    #     y = model(image)

    # ---------- 前后向传播 ----------
    x = torch.rand(1, 3, reso, reso)
    model.train()
    y = model(x)
    (y.sum()).backward()

    # ---------- 正确性验证 ----------
    model.eval()
    with torch.no_grad():
        y = model(x.repeat(2, 1, 1, 1))
        result = float((y[0] - y[1]).abs().sum())
        info.append('正确，输入相同输出相同' if result == 0.0 else '错误，输入相同输出不同，diff = {}'.format(result))
        x2 = torch.rand(1, 3, reso, reso)
        y2 = model(torch.cat([x, x2], dim=0))
        result = float((y2[0] - y[0]).abs().sum())
        info.append('正确，输入相同输出相同' if result == 0.0 else '错误，输入相同输出不同，diff = {}'.format(result))

    # ---------- 参数量 ----------
    info.append('Params:\t{} M'.format(round(count_parameters(model) / 1e6, 2)))

    # ---------- 计算量 ----------
    x = torch.rand(1, 3, reso, reso)
    flops = FlopCountAnalysis(model, x)
    info.append("FLOPs:\t{} G".format(round(flops.total() / 1e9, 2)))
    # print(flop_count_table(flops, max_depth=10))

    # ---------- 推理速度 ----------
    # n = 10
    # x = torch.rand(n, 3, reso, reso)
    # torch.cuda.synchronize()
    # start = time.time()
    # y = model(x)
    # torch.cuda.synchronize()
    # end = time.time()
    # info.append('Time:\t{} ms/image'.format(round((end - start) * 1000 / n, 2)))

    print('\n')
    for item in info:
        print(item)




