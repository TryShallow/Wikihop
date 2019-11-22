import logging


def config_logger(log_prefix):
    logger_prepared = logging.getLogger()
    logger_prepared.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - '
                                   '%(pathname)s[line:%(lineno)d]: %(message)s')
    # write to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_format)
    logger_prepared.addHandler(stream_handler)
    # write to file
    # rq = time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
    # log_path = os.getcwd() + '/logs/' + log_prefix + "/"
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # log_filename = log_path + rq + '.log'
    # file_handler = logging.FileHandler(log_filename, 'a', encoding='utf-8')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(log_format)
    # logger_prepared.addHandler(file_handler)
    return logger_prepared
