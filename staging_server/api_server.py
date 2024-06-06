from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from typing import Union
import pickle
import json

from model_manager import get_schema, get_version
from model_manager import write_meta, update_meta, write_artifact, get_model, delete_model, load_artifact
from pydantic_models import setup_request, params_request, metrics_request, tags_request, load_request, delete_request
from utils import auto_tagging

app = FastAPI()


@app.get('/endpoint')
async def endpoint() -> object:
    URL = 'http://localhost:8000'
    endpoints = {
        'setup': URL + '/setup',
        'log_params': URL + '/log_params',
        'log_metrics': URL + '/log_metrics',
        'log_data': URL + '/log_data',
        'log_tags': URL + '/log_tags',
        'log_models': URL + '/log_models',
        'list_models': URL + '/list_models',
        'get_models': URL + '/get_models',
        'delete_models': URL + '/delete_models',
        'load_models': URL + '/load_models'
    }
    return json.dumps(endpoints)


@app.post('/setup')
async def setup(request: setup_request) -> None:
    request = request.dict()
    write_meta('experiment', request)


@app.post('/log_params')
async def log_params(request: params_request) -> None:
    request = request.dict()
    write_meta('params', request)


@app.post('/log_metrics')
async def log_metrics(request: metrics_request) -> None:
    request = request.dict()
    write_meta('metric', request)


@app.post('/log_data')
async def log_data(request: UploadFile, file: UploadFile) -> Union[str, None]:
    request, dataset = json.loads(request.file.read()), pickle.loads(file.file.read())
    version = get_version(request['experiment_id'])

    request_keys, request_values = list(request.keys()), list(request.values())
    request_keys = request_keys[:1] + ['dataset_uri'] + request_keys[1:]
    request_values = request_values[:1] + [None] + request_values[1:]
    request = {x: xx for x, xx in zip(request_keys, request_values)}
    write_meta('dataset', request)

    result = write_artifact(request['experiment_id'], 'dataset', version, dataset)

    return result


@app.post('/log_tags')
async def log_tags(request: tags_request) -> None:
    request = request.dict()
    write_meta('tags', request)


@app.post('/log_models')
async def log_models(request: UploadFile, file: UploadFile) -> None:
    request = json.loads(request.file.read())
    request['model_binary'] = file.file.read()

    version = get_version(request['experiment_id'])

    req_parser = {}
    for table_name, schema in get_schema(['experiment', 'resource']).items():
        req_parser[table_name] = {k: v for k, v in request.items() if k in schema}

    for table_name, meta_data in req_parser.items():
        if table_name == 'experiment':
            update_meta(table_name, meta_data)
        else:
            write_meta(table_name, meta_data)

    tags = auto_tagging(request['req_txt'])
    if tags:
        for tag_name, value in tags.items():
            write_meta('tags', {'tags': tag_name, 'value': value, 'experiment_id': request['experiment_id']})

    for kind in ['model_binary', 'script', 'env', 'req_txt']:
        write_artifact(request['experiment_id'], kind, version, request[kind])


@app.get('/list_models')
async def list_models() -> list:
    return get_model()


@app.get('/get_models')
async def get_models(request: load_request) -> Union[dict, str]:
    request = request.dict()
    return get_model(**request)


@app.get('/delete_models')
async def delete_models(request: delete_request) -> str:
    request = request.dict()
    return delete_model(**request)


@app.get('/load_models')
async def load_models(request: load_request) -> str:
    request = request.dict()
    status, model_path = load_artifact(**request)

    return FileResponse(model_path) if status == 200 else model_path
