# 自主复现

import os
import pathlib
import sys
import elements
import ruamel.yaml as yaml
import portal
import memory_maze


# ================================================================ #
# ================================================================ #
# ================================================================ #

# agent.py start

# ================================================================ #
# ================================================================ #
# ================================================================ #

class Agent:
    banner = [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
    ]





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

# 获取当前脚本文件所在的目录路径
folder = pathlib.Path(__file__).parent

# 将父目录插入到系统路径的开头，使Python可以导入父目录中的模块
sys.path.insert(0, str(folder.parent))
# 将祖父目录插入到系统路径的开头，使Python可以导入祖父目录中的模块
sys.path.insert(1, str(folder.parent.parent))
# 将当前目录名称设置为包名，用于相对导入
__package__ = folder.name



def main(argv=None):
    # 如果Agent模块有banner属性，则打印banner的每一行
    [elements.print(line) for line in Agent.banner]
    # 读取配置文件，配置文件位于当前目录下的configs.yaml
    configs = elements.Path(folder / 'configs.yaml').read()
    # 使用yaml的安全模式加载配置文件内容
    configs = yaml.YAML(typ='safe').load(configs)
    # 解析命令行参数，已知的配置参数为['defaults'],其他未知参数保留在other中
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    # 使用默认配置创建配置对象
    config = elements.Config(configs['defaults'])
    # 遍历所有指定的配置名称，按顺序更新配置(后面的配置会覆盖前面的)
    for name in parsed.configs:
        config = config.update(configs[name])
    # 解析其他命令行参数并更新配置
    config = elements.Flags(config).parse(other)
    # 更新日志目录配置，使用时间戳格式化字符串
    config = config.update(logdir=(
        config.logdir.format(timestamp=elements.timestamp())))

    # 检查是否存在环境变量JOB_COMPLETION_INDEX(通常在集群作业中使用)
    if 'JOB_COMPLETION_INDEX' in os.environ:
        config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))

    # 创建日志目录路径对象
    logdir = elements.Path(config.logdir)

    # 如果脚本名不是以'_env'或者'_replay'结尾，则创建日志目录并保存配置
    if not config.script.endswith(('_env', '_replay')):
        logdir.mkdir()  #
        config.save(logdir / 'config.yaml')

    def init():
        elements.timer.global_timer.enabled = config.logger.timer




    portal.setup(
        errfile=config.errfile and logdir / 'error',
        clientkw=dict(logging_color='cyan'),
        serverkw=dict(logging_color='cyan'),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context
    )

    print(args)

    if config.script == 'train':
        main(config)




def make_agent(config):
    make_env(config, 0)


def make_env(config, index, **overrides):
    suite, task = config.task.split('_', 1)
    config.env.get(suite, {})





# ================================================================ #
# ================================================================ #
# ================================================================ #

# main.py end

# ================================================================ #
# ================================================================ #
# ================================================================ #


if __name__ == '__main__':
    main()






