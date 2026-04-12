import os
import json
import sys
from torch.utils.tensorboard import SummaryWriter


class TeeStream:
    """
    Redirect sys.stdout so all print() also goes to a log file.

    Usage in run_experiment():
        _log_file = open(os.path.join(save_dir, "log.txt"), "a")
        _orig_stdout = sys.stdout
        sys.stdout = TeeStream(_log_file, _orig_stdout)
        try:
            ...
        finally:
            sys.stdout = _orig_stdout
            _log_file.close()
    """
    def __init__(self, log_file, terminal):
        self.log_file = log_file
        self.terminal = terminal

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class Logger:
    def __init__(self, save_dir):
        if isinstance(sys.stdout, TeeStream):
            # TeeStream already captures all output -> reuse its log file
            self.log_file = sys.stdout.log_file
            self._owns_file = False
        else:
            self.log_file = open(os.path.join(save_dir, "log.txt"), "a")
            self._owns_file = True
        self.writer = SummaryWriter(log_dir=save_dir)

    def log(self, message):
        if isinstance(sys.stdout, TeeStream):
            # print() goes through TeeStream -> terminal + file simultaneously
            print(message)
        else:
            sys.stdout.write(message + "\n")
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def log_stats(self, stats, step, prefix="Step"):
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def save_config(self, cfg):
        with open(os.path.join(self.writer.log_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=4)

    def close(self):
        if self._owns_file:
            self.log_file.close()
        self.writer.close()
