import yaml


def merge_cfg(cfgs):
    result = dict()
    for cfg in cfgs:
        result.update(yaml.safe_load(open(cfg)))
    return result
