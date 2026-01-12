# 自主复现

import os
import pathlib
import sys
import elements
import embodied.jax
import ruamel.yaml as yaml
import portal


# ================================================================ #
# ================================================================ #
# ================================================================ #

# agent.py start

# ================================================================ #
# ================================================================ #
# ================================================================ #

class Agent(embodied.jax.Agent):
    banner = [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
    ]

    def __init__(self, obsSpace, actSpace, config):
        pass





# ================================================================ #
# ================================================================ #
# ================================================================ #

# agent.py end

# ================================================================ #
# ================================================================ #
# ================================================================ #







# ================================================================ #
# ================================================================ #
# ================================================================ #

# main.py start

# ================================================================ #
# ================================================================ #
# ================================================================ #

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name



def main(argv=None):
    [elements.print(line) for line in Agent.banner]

    configs = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())))

    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))

    logdir = elements.Path(config.logdir)

    if not config.script.endswith(('_env', '_replay')):
        logdir.mkdir()
        config.save(logdir / 'config.yaml')

    def init():
        elements.timer.global_timer.enabled = config.logger.timer





    print(config)



# ================================================================ #
# ================================================================ #
# ================================================================ #

# main.py end

# ================================================================ #
# ================================================================ #
# ================================================================ #


if __name__ == '__main__':
    main()






