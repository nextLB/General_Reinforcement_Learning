# 通用式强化学习


关于官方eamerV3运行的备选命令示例



    python -u main.py \
      --configs size1m \
      --batch_size 4 \
      --batch_length 32 \
      --run.envs 4 \
      --jax.compute_dtype float32 \
      --logdir ~/logdir/dreamer/test_small \
      --replay.size 1e5 \
      --agent.dyn.rssm.deter 512 \
      --agent.dyn.rssm.hidden 64 \
      --agent.dyn.rssm.classes 4 \
      --agent.dyn.rssm.blocks 4




    python -u main.py \
      --logdir ~/logdir/dreamer/{timestamp} \
      --configs crafter \
      --batch_size 4 \
      --batch_length 32 \
      --run.train_ratio 32








## 关于显存使用的查看

    watch -n 2 nvidia-smi



