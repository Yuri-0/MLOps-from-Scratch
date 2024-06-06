import os
import subprocess
import inspect
import platform
import uuid
import time
import pickle
import json

import requests
import randomname
import torch
from typing import OrderedDict, Any

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from torch.nn import Module
from params_parser import fit_params_parser


class ModelManager:
    def __init__(self, endpoint: str, user_id: str, model_name: str | None = None, custom_resource: dict | None = None):
        self.exp_run = False
        endpoint = endpoint + 'endpoint' if endpoint[-1] == '/' else endpoint + '/endpoint'
        self.endpoint = json.loads(requests.get(endpoint).json())
        self.user_id = user_id
        self.model_name = model_name
        self.custom_resource = custom_resource if isinstance(custom_resource, dict) else None

        self.script_path = inspect.stack()[1].filename

    def _warm_up(self) -> None:
        if self.model_name is None:
            self.model_name = randomname.get_name()
        self.exp_run = True
        self.uuid = str(uuid.uuid4())

        exp_init = {
            'experiment_id': self.uuid,
            'user_id': self.user_id,
            'name': self.model_name,
            'status': 'RUNNING',
            'start_time': time.time()
        }
        _ = requests.post(self.endpoint['setup'], data=json.dumps(exp_init))

    def log_params(self, param_name: str, param: str) -> None:
        if not self.exp_run:
            self._warm_up()
        if not isinstance(param, (int, float, str)):
            param = str(param)
        exp_param = {
            'param_name': param_name,
            'value': param,
            'experiment_id': self.uuid
        }
        _ = requests.post(self.endpoint['log_params'], data=json.dumps(exp_param))

    def log_metrics(self, metric_name: str, metric: float | int | str, step: int | None = None) -> None:
        if not self.exp_run:
            self._warm_up()
        exp_metric = {
            'metric_name': metric_name,
            'value': metric,
            'timestamp': time.time(),
            'experiment_id': self.uuid,
            'step': step
        }
        _ = requests.post(self.endpoint['log_metrics'], data=json.dumps(exp_metric))

    def log_data(self, data_name: str, data: Any) -> None:
        if not self.exp_run:
            self._warm_up()
        exp_data = {
            'dataset_name': data_name,
            'dataset_type': str(type(data)),
            'experiment_id': self.uuid,
        }
        exp_file = {
            'request': json.dumps(exp_data),
            'file': pickle.dumps(data)
        }
        response = requests.post(self.endpoint['log_data'], files=exp_file)
        if response.text != 'null':
            print(response.text)

    def log_tags(self, tag_name: str, tag: str) -> None:
        if not self.exp_run:
            self._warm_up()
        exp_tag = {
            'tags': tag_name,
            'value': tag,
            'experiment_id': self.uuid
        }
        requests.post(self.endpoint['log_tags'], data=json.dumps(exp_tag))

    def log_models(self, model: LightningModule | Module, model_name: str | None = None) -> None:
        if model_name is not None:
            self.model_name = model_name
        if not self.exp_run:
            self._warm_up()

        def _is_lazy_weight_tensor(p):
            from torch.nn.parameter import UninitializedParameter

            if isinstance(p, UninitializedParameter):
                return True
            return False

        total_parameters = sum(p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in model.parameters())
        precision_to_bits = {"64": 64, "32": 32, "16": 16, "bf16": 16}
        precision = precision_to_bits.get(model.trainer.precision, 32) if model._trainer else 32
        precision_megabytes = (precision / 8.0) * 1e-6

        model_size = total_parameters * precision_megabytes

        with open(self.script_path, 'r') as f:
            script = f.read()

        env = subprocess.check_output('conda env export', shell=True).decode('utf-8')

        req_txt = subprocess.check_output('pip freeze', shell=True).decode('utf-8')

        python_version = subprocess.check_output('python --version', shell=True).decode()
        self.log_tags('python_version', python_version.split(' ')[-1].replace('\n', ''))

        model_meta = {
            'experiment_id': self.uuid,
            'status': 'DONE',
            'end_time': time.time(),
            'model_size': model_size,
            'cpu': os.cpu_count(),
            'nvidia_gpu': torch.cuda.device_count(),
            'custom_resource': self.custom_resource,
            'system': platform.system(),
            'machine': platform.machine(),
            'script': script,
            'env': env,
            'req_txt': req_txt
        }
        model_file = {
            'request': json.dumps(model_meta),
            'file': pickle.dumps(model.state_dict())
        }
        _ = requests.post(self.endpoint['log_models'], files=model_file)

    def load_models(self, model_name: str, user_id: str | None = None, version: int | None = None)\
            -> OrderedDict | None:
        if user_id is None:
            user_id = self.user_id
        exp_load = {
            'name': model_name,
            'user_id': user_id,
            'version': version
        }
        response = requests.get(self.endpoint['load_models'], data=json.dumps(exp_load))
        if 'text' not in response.headers['Content-Type']:
            print(response.text)
        else:
            model = pickle.loads(response.content)

            return model

    def list_models(self) -> None:
        response = requests.get(self.endpoint['list_models']).json()
        print(response)

    def get_models(self, model_name: str, user_id: str, version: int | None = None) -> None:
        exp_get = {
            'name': model_name,
            'user_id': user_id,
            'version': version
        }
        response = requests.get(self.endpoint['get_models'], data=json.dumps(exp_get))
        response = response.text if not 'json' in response.headers['Content-Type'] else response.json()
        print(response)

    def delete_models(self, model_name: str, user_id: str, version: int) -> None:
        exp_delete = {
            'name': model_name,
            'user_id': user_id,
            'version': version,
            'deleted_time': time.time()
        }
        result = requests.get(self.endpoint['delete_models'], data=json.dumps(exp_delete)).text
        print(result)


class TotalLogger(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        try:
            params = fit_params_parser(self.logger.script_path, trainer)
        except ValueError:
            params = {}
        params.update({x: xx for x, xx in vars(pl_module).items() if not hasattr(xx, '__dict__') and x[0] != '_'})
        for param_name, param in params.items():
            self.logger.log_params(param_name, param)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        metrics, step = trainer.logged_metrics, trainer.current_epoch
        for metric, value in metrics.items():
            metric, value = metric.split('_')[-1], value.item()
            self.logger.log_metrics(metric, value, step)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics, step = trainer.logged_metrics, trainer.current_epoch
        for metric, value in metrics.items():
            if 'val' not in metric:
                metric = 'val_' + metric.split('_')[-1]
            value = value.item()
            self.logger.log_metrics(metric, value, step)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        metrics, step = trainer.logged_metrics, trainer.current_epoch
        for metric, value in metrics.items():
            if 'test' not in metric:
                metric = 'test_' + metric.split('_')[-1]
            value = value.item()
            self.logger.log_metrics(metric, value, step)

    def on_train_end(self, trainer, pl_module):
        self.logger.log_models(pl_module, self.logger.model_name)
