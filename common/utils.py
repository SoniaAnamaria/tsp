import datetime
import time
from collections import defaultdict, deque

import torch


class Value(object):
    def __init__(self, window_size=20, info=None):
        if info is None:
            info = '{global_avg:.2f}'
        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0
        self.info = info

    def __str__(self):
        return self.info.format(
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value)

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]


class Logger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(Value)
        self.delimiter = delimiter

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f'{name}: {str(meter)}')
        return self.delimiter.join(loss_str)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f'"{type(self).__name__}" object has no attribute "{attr}"')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log(self, iterable, print_freq, header, device):
        i = 0
        start_time = time.time()
        end_time = time.time()
        iteration_time = Value(info='{avg:.2f}')
        data_time = Value(info='{avg:.2f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            message = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max_mem: {memory:.2f}GB'
            ])
        else:
            message = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        GB = 1024.0 ** 3
        for obj in iterable:
            data_time.update(time.time() - end_time)
            yield obj
            iteration_time.update(time.time() - end_time)
            if i % print_freq == 0:
                eta_seconds = iteration_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(message.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iteration_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated(device) / GB), flush=True)
                else:
                    print(message.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iteration_time), data=str(data_time)), flush=True)
            i += 1
            end_time = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} total time: {total_time_str}\n')


def compute_accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)
        _, prediction = output.topk(max_k, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.unsqueeze(1))

        res = []
        for k in top_k:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def write_to_file(file, mode, content_to_write):
    with open(file, mode) as f:
        f.write(content_to_write)
