import os
import gc
import time
import torch
import torch.nn as nn

from torch import optim
from model import Model
from config import Config
from dataset import MyDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, logger, config: Config):
        self.logger = logger
        self.config = config
        self.writer = SummaryWriter(config.log_dir)

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.model_path = config.model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        logger.info('Load preprocessed train and dev file.')
        if config.is_test:
            train_data = MyDataset(config, "", "", is_test=1)
            dev_data = MyDataset(config, "", "", is_test=2)
        else:
            train_data = MyDataset(config, config.train_graph_file,
                                   elmo_file=config.train_elmo_file,
                                   glove_file=config.train_glove_file,
                                   extra_file=config.train_extra_file)
            dev_data = MyDataset(config, config.dev_graph_file,
                                 elmo_file=config.dev_elmo_file,
                                 glove_file=config.dev_glove_file,
                                 extra_file=config.dev_extra_file)
        logger.info("Data has prepared, train: %d, dev: %d." %
                    (len(train_data), len(dev_data)))
        self.train_data_loader = DataLoader(train_data, self.batch_size)
        self.dev_data_loader = DataLoader(dev_data, self.batch_size)
        self.model = Model(config)
        self.load_parameters(self.config.pretrained_parameters)
        self.model.to(config.device)

        if config.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate,
                                       momentum=config.momentum)
        elif config.optim.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def load_parameters(self, model_file=None):
        if model_file is not None:
            self.model.load_state_dict(torch.load(self.model_path + '/' + model_file,
                                                  map_location=self.config.map_location))
            self.logger.info('Load parameters file ' + model_file)

    def train(self):
        self.model.train()
        best_acc = 0.0
        save_model_prefix = os.path.join(self.model_path, self.config.model_prefix)
        for epoch in range(self.num_epochs):
            self.logger.info("Epoch %d/%d" % (epoch + 1, self.num_epochs))
            start_time = time.time()
            for batch in self.train_data_loader:
                output = self.model(MyDataset.to(batch, self.config.device))
                self.model.zero_grad()
                loss = self._calc_loss(output, batch)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("scalar/loss", loss.cpu().item(), epoch)

            time_diff = time.time() - start_time
            self.logger.info("epoch %d time consumed: %dm%ds." %
                             (epoch + 1, time_diff // 60, time_diff % 60))
            # evaluate model
            cur_acc = self.eval_dev(self.dev_data_loader)
            self.model.train()
            self.logger.info("Current accuracy: %.3f" % cur_acc)
            self.writer.add_scalar("scalar/accuracy", cur_acc)
            if cur_acc > best_acc:  # and epoch > 10:
                save_filename = save_model_prefix + str(cur_acc)
                torch.save(self.model.state_dict(), save_filename)
                best_acc = cur_acc

    def eval(self):
        self.model.eval()

    def eval_dev(self, dev_data_loader):
        self.model.eval()
        correct_count = 0
        total_count = 0
        for batch in dev_data_loader:
            output = self.model(MyDataset.to(batch, self.config.device))
            pred = torch.argmax(output, 1)
            correct_count += (pred.cpu().detach().numpy() == batch['answer_index'].numpy()).sum()
            total_count += len(batch['query_length'])
        return float(correct_count) / total_count

    def _calc_loss(self, output, batch):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        return cross_entropy(output, batch['answer_index'].to(self.config.device))