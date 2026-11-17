from pathlib import Path
import json
import tempfile
import numpy as np
from popgym_arcade.experiments import memory_bias as experiment

experiment.evaluate = lambda *args, **kwargs: np.array([0.25, 0.75], dtype=np.float32)
with tempfile.TemporaryDirectory(prefix='ppo-checkpoint-smoke-') as temporary:
    root=Path(temporary)
    args=experiment.build_parser().parse_args(['--variants','ib_reverse_prior20','--hidden','6','--timesteps','16','--checkpoint-interval','8','--gradient-log-interval','4','--eval-episodes','2','--output',str(root/'original')])
    config=experiment.upstream_config() | {'ENV_NAME':'CartPoleHard','PARTIAL':False,'TOTAL_TIMESTEPS':16,'NUM_ENVS':2,'NUM_STEPS':4,'NUM_MINIBATCHES':2,'UPDATE_EPOCHS':1}
    experiment.run_one(args,'ib_reverse_prior20',0,config)
    original=root/'original/ib_reverse_prior20_seed0'
    assert len(list((original/'checkpoints').glob('step_*/checkpoint.json')))==2
    extended=experiment.build_parser().parse_args(['--variants','ib_reverse_prior20','--hidden','6','--timesteps','24','--checkpoint-interval','8','--gradient-log-interval','4','--eval-episodes','2','--resume-from',str(original),'--extend-training','--output',str(root/'extended')])
    experiment.run_one(extended,'ib_reverse_prior20',0,config|{'TOTAL_TIMESTEPS':24})
    directory=root/'extended/ib_reverse_prior20_seed0'
    rows=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    gradients=[json.loads(line) for line in (directory/'gradient_norms.jsonl').read_text().splitlines()]
    assert [row['steps'] for row in rows]==[8,16,24]
    assert len(gradients)==3
    assert all(row['gradient_norm_post_clip_mean'] <= 0.500001 for row in rows)
    assert (original/'result.json').exists()
    print('PASS: real PPO training checkpoints, complex optimizer state, frozen calibration reuse, gradient logging, and explicit budget extension into fresh output.')
