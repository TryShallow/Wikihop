import argparse

from logger import config_logger
from config import Config
from dataset import MyDataset
from trainer import Trainer


if __name__ == '__main__':
    str2bool = lambda x: True if x.upper() in ['TRUE', 'YES', '1'] else False
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test", help='training model using only part data', type=str2bool,
                        default=True)
    parser.add_argument("--use_elmo", help='use elmo feature', type=str2bool, default=True)
    parser.add_argument("--use_glove", help='use glove feature', type=str2bool, default=True)
    parser.add_argument("--use_bert", help='use bert feature', type=str2bool, default=False)
    parser.add_argument("--use_extra", help='add ner and pos feature', type=str2bool, default=True)
    parser.add_argument("--batch_size", help='train batch size', type=int, default=16)
    parser.add_argument("--pretrained_parameters", help='pre-trained model parameter dir',
                        type=str, default=None)
    parser.add_argument("--device", help='training device', type=str, default='cuda:3')
    parser.add_argument("--optim", help='optimization algorithm', type=str, default='sgd')
    parser.add_argument("--learning_rate", help='learning rate', type=float, default=1e-4)
    parser.add_argument("--model_prefix", help='trained model prefix name', type=str,
                        default='wiki_org_gee_')
    args = parser.parse_args()
    config = Config()
    config.use_elmo = args.use_elmo
    config.use_glove = args.use_glove
    config.is_test = args.is_test
    config.batch_size = args.batch_size
    config.pretrained_parameters = args.pretrained_parameters
    config.device = args.device
    config.optim = args.optim
    config.learning_rate = args.learning_rate
    config.use_extra = args.use_extra
    config.model_prefix = args.model_prefix
    config.print()
    logger = config_logger("main")

    trainer = Trainer(logger, config)
    trainer.train()