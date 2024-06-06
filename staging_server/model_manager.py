import os
import shutil
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

from utils import install

root_path=os.getcwd()

if not os.path.exists(os.path.join(root_path, 'model_registry', 'repository')):
    install(root_path)


def get_schema(tables):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()

        schema = {}
        for table in tables:
            cursor.execute('SELECT * FROM {}'.format(table))
            schema[table] = list(map(lambda x: x[0], cursor.description))

    return schema


def get_version(experiment_id):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()

        cursor.execute('SELECT experiment_id, tags, value FROM tags WHERE experiment_id=? AND tags="version"',
                       (experiment_id,))
        version = cursor.fetchone()
        if version is None:
            cursor.execute('SELECT experiment_id, name FROM experiment WHERE experiment_id=? AND deleted_time is null',
                           (experiment_id,))
            name = cursor.fetchone()[-1]

            cursor.execute('SELECT experiment_id, name FROM experiment WHERE name=? AND deleted_time is null',
                           (name,))
            exps = ', '.join([f'"{x[0]}"' for x in cursor.fetchall()])

            cursor.execute('SELECT experiment_id, tags, value FROM tags WHERE experiment_id in ({}) AND tags="version"'
                           .format(exps))
            versions = cursor.fetchall()
            version = max([x[-1] if x[-1] else 0 for x in versions]) + 1 if versions else 0

            cursor.execute('INSERT INTO tags VALUES ("version", ?, ?)', (version, experiment_id))
        else:
            version = version[-1]

        return version


def get_experiment(name, user_id=None, version=None):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()

        if user_id is None:
            cursor.execute('SELECT experiment_id, name FROM experiment WHERE name=? AND deleted_time is null', (name,))
        else:
            cursor.execute('SELECT experiment_id, name, user_id FROM experiment '
                           'WHERE name=? AND user_id=? AND deleted_time is null', (name, user_id))
        exps = ', '.join([f'"{x[0]}"' for x in cursor.fetchall()])

        if version is None:
            cursor.execute('SELECT experiment_id, tags, value FROM tags WHERE experiment_id in ({}) AND tags="version" '
                           'ORDER BY value DESC'.format(exps))
        else:
            cursor.execute('SELECT experiment_id, tags, value FROM tags WHERE experiment_id in ({}) AND tags="version" '
                           'AND value={}'.format(exps, version))
        last_exp = cursor.fetchone()
        last_exp = (last_exp[0], last_exp[-1])

    return last_exp


def write_meta(table_name, meta_data):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO {}({}) VALUES({})'.format(
            table_name,
            ', '.join(meta_data.keys()),
            ', '.join(['?' for _ in range(len(meta_data))])
        ), tuple(meta_data.values()))


def update_meta(table_name, meta_data):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE {} SET {} WHERE experiment_id="{}"'.format(
            table_name,
            ', '.join([f'{k}={v}' if isinstance(v, (int, float, complex)) else f'{k}="{v}"'
                       for k, v in meta_data.items() if k != 'experiment_id']),
            meta_data['experiment_id']
        ))


def get_model(name='all', user_id=None, version=None):
    if name == 'all':
        with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT experiment_id, user_id, name, end_time, model_size '
                           'FROM experiment WHERE end_time is not null AND deleted_time is null')
            all_models = cursor.fetchall()
            exps = ', '.join([f'"{x[0]}"' for x in all_models])

            cursor.execute('SELECT tags, value, experiment_id FROM tags WHERE experiment_id in ({}) AND tags="version"'
                           .format(exps))
            all_versions = {x[2]: x[1] for x in cursor.fetchall()}

        model_list = []
        for meta_data in all_models:
            simple_info = {
                'Model Name': meta_data[2],
                'Created By': meta_data[1],
                'Model Size (MB)': meta_data[4],
                'Creation Time': datetime.fromtimestamp(meta_data[3]).strftime('%Y-%m-%d %H:%M:%S'),
                'Version': all_versions[meta_data[0]]
            }
            model_list.append(simple_info)

        return model_list
    else:
        if user_id is None:
            raise TypeError

        try:
            experiment_id, _ = get_experiment(name, user_id, version)
        except TypeError:
            return 'Model not found'

        with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT experiment_id, name, user_id, model_size, end_time FROM experiment '
                           'WHERE experiment_id=?', (experiment_id,))
            result = cursor.fetchone()
            detail_info = {k: v for k, v in zip(['model_name', 'creator', 'model_size', 'creation_time'], result[1:])}
            detail_info['creation_time'] = datetime.fromtimestamp(detail_info['creation_time'])\
                .strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('SELECT dataset_name, dataset_type, experiment_id FROM dataset WHERE experiment_id=?',
                           (experiment_id,))
            result = cursor.fetchone()
            if result:
                detail_info.update({k: v for k, v in zip(['dataset_name', 'dataset_type'], result[:-1])})
            else:
                detail_info.update({x: None for x in ['dataset_name', 'dataset_type']})

            cursor.execute('SELECT tags, value, experiment_id FROM tags WHERE experiment_id=?', (experiment_id,))
            result = cursor.fetchall()
            detail_info.update({'tags': [{x[0]: x[1]} for x in result]})

            cursor.execute('SELECT system, machine, experiment_id FROM resource WHERE experiment_id=?', (experiment_id,))
            result = cursor.fetchone()
            detail_info.update({k: v for k, v in zip(['system', 'machine'], result[:-1])})

            schema = ['model_name', 'creator', 'model_size', 'dataset_name', 'dataset_type',
                      'tags', 'system', 'machine', 'creation_time']

        return {x: detail_info[x] for x in schema}


def delete_model(name, user_id, version, deleted_time):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()

        try:
            experiment_id, _ = get_experiment(name, user_id, version)
        except TypeError:
            return 'Model not found'

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [x[0] for x in cursor.fetchall() if x[0] != 'experiment']

        for table in tables:
            cursor.execute('DELETE FROM {} WHERE experiment_id=?'.format(table), (experiment_id,))

        cursor.execute('UPDATE experiment SET deleted_time=? WHERE experiment_id=?',
                       (deleted_time, experiment_id,))

        shutil.rmtree(os.path.join(root_path, 'model_registry', 'repository', user_id, name, str(version)))

        return f'Model of (model name: {name} / creator: {user_id} / verison: {version}) is deleted successfully'


def write_artifact(experiment_id, kind, version, data):
    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT experiment_id, user_id, name FROM experiment WHERE experiment_id=? AND deleted_time is null',
                       (experiment_id,))
        user_id, model_name = cursor.fetchone()[1:]

    save_dir = os.path.join(root_path, 'model_registry', 'repository', user_id, model_name, str(version))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if kind == 'model_binary':
        with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
            f.write(data)
    elif kind == 'dataset':
        if isinstance(data, pd.DataFrame):
            data.to_csv(os.path.join(save_dir, 'dataset.csv'))
        elif isinstance(data, np.ndarray):
            np.save(os.path.join(save_dir, 'dataset.npy'), data)
        elif isinstance(data, bytes):
            with open(os.path.join(save_dir, 'dataset.pkl'), 'wb') as f:
                f.write(data)
        else:
            return 'Not capable dataset type'
        with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE dataset SET dataset_uri=? WHERE experiment_id=?', (save_dir, experiment_id))
        return
    else:
        artifact_book = {'script': 'script.py', 'env': 'environment.yaml', 'req_txt': 'requirements.txt'}
        if any(kind == x for x in artifact_book.keys()):
            with open(os.path.join(save_dir, artifact_book[kind]), 'w') as f:
                f.write(data)

    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE experiment SET artifact_uri=? WHERE experiment_id=?', (save_dir, experiment_id))


def load_artifact(name, user_id, version):
    try:
        if version is None:
            _, version = get_experiment(name)

        model_file = os.path.join(root_path, 'model_registry', 'repository', user_id, name, str(version), 'model.pkl')
        if not os.path.exists(model_file):
            raise TypeError

    except (TypeError, FileNotFoundError):
        return 404, 'Model not found'

    return 200, model_file
