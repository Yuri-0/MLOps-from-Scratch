import re
import inspect


def fit_params_parser(script_path, trainer):
    with open(script_path, 'r') as f:
        codes = '\n'.join([x.rstrip() for x in f.readlines()]) + '\n'

    params = re.search(r'\.fit\((.*?)\)\n', codes).group(1)

    pattern = r"(?:\[(.*?)\]|\((.*?)\)|{(.*?)})"

    matches = re.findall(pattern, params)
    matches = [match for group in matches for match in group if match]
    for match in matches:
        params = params.replace(match, '{}')

    params = params.split(',')
    params = [x.format(matches.pop(0)).strip() if '{}' in x else x.strip() for x in params]

    fit_params = {}
    defaults = [x.name for x in inspect.signature(trainer.fit).parameters.values()]
    for idx, param in enumerate(params):
        if '=' in param:
            kv_pair = param.split('=')
            fit_params[kv_pair[0]] = kv_pair[1]
        else:
            fit_params[defaults[idx]] = param

    return fit_params
