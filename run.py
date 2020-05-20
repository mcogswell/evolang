#!/usr/bin/env python3
import subprocess
from subprocess import Popen
import time
import tempfile
import socket
import argparse
import sys
import os
import os.path as pth
import glob
import shlex
import random

parser = argparse.ArgumentParser(description='Generic slurm run script')
parser.add_argument('EXP', type=str, nargs='?', help='experiment to run',
                    default=None)
parser.add_argument('-w', '--node', help='name of the machine to run this on')
parser.add_argument('-x', '--exclude', help='do not run on this machine')
parser.add_argument('-g', '--ngpus', type=int, default=1,
                    help='number of gpus to use')
parser.add_argument('-d', '--delay', type=int, default=0,
                    help='number of hours to delay job start')
parser.add_argument('-p', default='debug_noslurm',
                    choices=['debug', 'short', 'long', 'batch', 'debug_noslurm'])
parser.add_argument('-f', '--profile', type=int, default=0)
parser.add_argument('-m', '--mode',
                        help='which command to run',
                        choices=['train', 'train2', 'evaluate', 'evaluate2'])
parser.add_argument('-q', '--qos', help='slurm quality of service',
                        choices=['overcap'])

args = parser.parse_args()

Popen('mkdir -p data/cmds/', shell=True).wait()
Popen('mkdir -p data/logs/', shell=True).wait()
Popen('mkdir -p data/experiments/', shell=True).wait()
python = 'python -m torch.utils.bottleneck ' if args.profile == 1 else 'python'

jobid = 0
def runcmd(cmd, external_log=None):
    '''
    Run cmd, a string containing a command, in a bash shell using gpus
    allocated by slurm. Frequently changed slurm settings are configured
    here. 
    '''
    global jobid
    log_fname = 'data/logs/job_{}_{:0>3d}_{}.log'.format(
                    int(time.time()), random.randint(0, 999), jobid)
    if external_log:
        if pth.lexists(external_log):
            os.unlink(external_log)
        link_name = pth.relpath(log_fname, pth.dirname(external_log))
        os.symlink(link_name, external_log)
    jobid += 1
    # write SLURM job id then run the command
    write_slurm_id = True
    if write_slurm_id:
        script_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                             dir='./data/cmds/', prefix='.', suffix='.slurm.sh')
        script_file.write('echo "slurm job id: $SLURM_JOB_ID"\n')
        script_file.write('echo ' + cmd + '\n')
        script_file.write('echo "host: $HOSTNAME"\n')
        script_file.write('echo "cuda: $CUDA_VISIBLE_DEVICES"\n')
        #script_file.write('nvidia-smi -i $CUDA_VISIBLE_DEVICES\n')
        script_file.write(cmd)
        script_file.close()
        # use this to restrict runs to current host
        #hostname = socket.gethostname()
        #cmd = ' -w {} bash '.format(hostname) + script_file.name
        cmd = 'bash ' + script_file.name

    srun_prefix = 'srun --gres gpu:{} '.format(args.ngpus)
    if args.node:
        srun_prefix += '-w {} '.format(args.node)
    if args.exclude:
        srun_prefix += '-x {} '.format(args.exclude)
    if args.delay:
        srun_prefix += '--begin=now+{}hours '.format(args.delay)
    if args.qos:
        srun_prefix += '--qos {} '.format(args.qos)

    ############################################################################
    # uncomment the appropriate line to configure how the command is run
    #print(cmd)
    if args.p == 'debug':
        # debug to terminal
        Popen(srun_prefix + '-p debug --pty ' + cmd, shell=True).wait()
        # debug to log file
        #Popen(srun_prefix + '-p debug -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
        # debug on current machine without slurm (manually set CUDA_VISIBLE_DEVICES)
        #Popen(cmd, shell=True).wait()

        #logfile = open(log_fname, 'w', buffering=1)
        #proc = Popen(cmd, shell=True, stdout=logfile, stderr=logfile, bufsize=0, universal_newlines=True)
        #proc.wait()
        #for line in proc.stdout:
        #    print(line, end='')
        #    logfile.write(line)
        #proc.wait()
        return None
    elif args.p == 'debug_noslurm':
        # debug on current machine without slurm (manually set CUDA_VISIBLE_DEVICES)
        Popen(cmd, shell=True).wait()
        return None
    elif args.p == 'short':
        # log file
        Popen(srun_prefix + '-p short -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
        return log_fname
    elif args.p == 'long':
        # log file
        Popen(srun_prefix + '-p long -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
        return log_fname
    elif args.p == 'batch':
        # This is not a partition but is for use with sbatch, which itself
        # specifies a partition. The script which calls `python run.py` should
        # be run with sbatch and have the appropriate partition specified.
        # Furthermore, it is impotant to Popen.wait() for sbatch calls to srun.
        # log file
        Popen(srun_prefix + '-o {} --open-mode=append '.format(log_fname) + cmd, shell=True).wait()
        return log_fname


#######################################
# Config

def config(exp):
    if exp is None:
        import warnings
        warnings.warn('Missing experiment code')
        return locals()
    experiment = exp # put it in locals()
    assert exp.startswith('exp')
    exp_dir = pth.join('data/experiments', exp)
    model_dir = pth.join(exp_dir, 'models')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_fname = pth.join(exp_dir, 'log.txt')
    test_log_fname = pth.join(exp_dir, 'evaluate_log.txt')
    vers = list(map(int, exp[3:].split('.')))
    vers += [0] * (100 - len(vers))
    data_path = 'data/synthetic_dataset.json'
    remember = ''
    nabots = nqbots = 1
    epochs = ''
    kill_epoch = 0
    kill_policy = 'random'
    kill_type_policy = 'both'
    seed = ''
    symmetric = 0
    reward = ''
    task_reward_freq = ''
    teaching_freq = ''
    save_epoch = '1000'
    batch_size = ''
    lr = ''

    if vers[0] == 0:
        aOutVocab = 4
        qOutVocab = 3
        remember = '--memoryless-abot'
        if vers[1] == 1:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''
        elif vers[1] == 2:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''

    elif vers[0] in [1, 2]:
        if vers[0] == 2:
            data_path = 'datasets/adapted_toy64_split_0.8.json'
        aOutVocab = 4
        qOutVocab = 3
        remember = '--memoryless-abot'
        if vers[1] == 1:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''
        elif vers[1] == 2:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        if vers[2] in [0, 1]:
            nabots = nqbots = 1
        else:
            assert vers[2] > 1
            nabots = nqbots = vers[2]
        if vers[3] == 1:
            kill_epoch = 300
        elif vers[3] == 2:
            kill_epoch = 600
        elif vers[3] == 3:
            kill_epoch = 1000
        elif vers[3] == 4:
            kill_epoch = 5000
        elif vers[3] == 5:
            kill_epoch = 10000

    elif vers[0] == 3:
        # baseline or overcomplete model
        aOutVocab = 4
        qOutVocab = 3
        remember = '--memoryless-abot'
        if vers[1] == 2:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        elif vers[1] == 3:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
            symmetric = 1

        # normal(0), multiagent(1), or evolve(2/3)
        if vers[2] > 0:
            nabots = nqbots = 5
        if vers[2] == 1:
            kill_epoch = 10000
        elif vers[2] == 2:
            kill_epoch = 5000
        elif vers[2] == 3:
            kill_epoch = 0

        # emnlp or random
        if vers[3] == 0:
            # assume emnlp TODO 
            data_path = 'datasets/emnlp_toy64_te0.2_folds4_'
            # different folds
            data_path += 'cv{}.json'.format(vers[4])
        elif vers[3] == 1:
            data_path = 'datasets/adapted_toy64_split_0.8.json'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    elif vers[0] == 4:
        # overcomplete model
        if vers[1] == 0:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        elif vers[1] == 1:
            aOutVocab = 32
            qOutVocab = 32
            remember = ''
            symmetric = 1
        elif vers[1] == 2:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
            symmetric = 1

        # normal(0), multiagent(1), or evolve(2/3)
        nabots = nqbots = 5
        kill_epoch = 0
        if vers[2] == 1:
            kill_epoch = 5000
        elif vers[2] == 2:
            kill_epoch = 5000
            kill_policy = 'simple_meritocratic'
        elif vers[2] == 3:
            kill_epoch = 5000
            kill_policy = 'eps_greedy0.2'
        elif vers[2] == 4:
            kill_epoch = 5000
            kill_policy = 'eps_greedy0.0'
        elif vers[2] == 5:
            kill_epoch = 5000
            nabots = nqbots = 20
        elif vers[2] == 6:
            kill_epoch = 5000
            nabots = nqbots = 40

        # emnlp or random
        if vers[3] == 0:
            # assume emnlp TODO 
            data_path = 'datasets/emnlp_toy64_te0.2_folds4_'
            # different folds
            data_path += 'cv{}.json'.format(vers[4])
        elif vers[3] == 1:
            data_path = 'datasets/adapted_toy64_split_0.8.json'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    # first teaching experiment
    elif vers[0] == 5:
        # overcomplete model
        aOutVocab = 64
        qOutVocab = 64
        remember = ''

        # eps greedy selection
        nabots = nqbots = 5
        kill_epoch = 5000
        kill_policy = 'eps_greedy0.2'

        # teaching / task reward tradeoff
        if vers[1] == 0:
            task_reward_freq = '--task-reward-freq 1'
            teaching_freq = '--teaching-freq 1'
        elif vers[1] == 1:
            task_reward_freq = '--task-reward-freq 1'
            teaching_freq = '--teaching-freq 4'
        elif vers[1] == 2:
            task_reward_freq = '--task-reward-freq 4'
            teaching_freq = '--teaching-freq 1'
        elif vers[1] == 3:
            task_reward_freq = '--task-reward-freq 1'
            teaching_freq = '--teaching-freq 40'

        # assume emnlp crossval
        data_path = 'datasets/emnlp_toy64_te0.2_folds4_'
        # different folds
        fold = vers[4]
        data_path += 'cv{}.json'.format(fold)

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    elif vers[0] in [7, 8]:
        # baseline or overcomplete model
        if vers[1] == 0:
            aOutVocab = 4
            qOutVocab = 3
            remember = '--memoryless-abot'
        elif vers[1] == 1:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        elif vers[1] == 2:
            aOutVocab = 4
            qOutVocab = 3
            remember = ''
        elif vers[1] == 3:
            aOutVocab = 64
            qOutVocab = 64
            remember = '--memoryless-abot'
        elif vers[1] == 4:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''

        # baseline
        if vers[2] == 0:
            kill_epoch = 0
        # multiagent
        elif vers[2] == 1:
            kill_epoch = 0
            nabots = nqbots = 5
        # random killing
        elif vers[2] == 2:
            nabots = nqbots = 5
            kill_epoch = 5000
        # eps greedy
        elif vers[2] == 3:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'eps_greedy0.2'
        # kill oldest bots
        elif vers[2] == 4:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'oldest'
        # single agent, randomly kill qbot or abot
        elif vers[2] == 5:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'random'
        # single agent, alternate between killing qbot and abot
        elif vers[2] == 6:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'alternate'
        # single agent, kill all (random restart baseline)
        elif vers[2] == 7:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # multi agent, kill all (random restart baseline)
        elif vers[2] == 8:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi All (longer)
        elif vers[2] == 9:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi Oldest (longer)
        elif vers[2] == 10:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'
        # Multi Uniform Random (longer)
        elif vers[2] == 11:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
        # Multi Epsilon Greedy (longer)
        elif vers[2] == 12:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'eps_greedy0.2'
        # Multi Oldest (longer) (n=10)... don't use this one because it cuts the number of iterations in half (need kill_epoch=50000)
        elif vers[2] == 13:
            # TODO: increase kill_epoch here
            nabots = nqbots = 10
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'

        # comp cross-val
        if vers[3] == 0:
            # assume emnlp TODO 
            data_path = 'datasets/comp_toy64_te0.2_folds4_'
            # different folds
            data_path += 'cv{}.json'.format(vers[4])
        # comp stop on train
        elif vers[3] == 1:
            data_path = 'datasets/comp_toy64_split_0.8.json'
        # emnlp cross-val
        elif vers[3] == 2:
            # assume emnlp TODO 
            data_path = 'datasets/emnlp_toy64_te0.2_folds4_'
            # different folds
            data_path += 'cv{}.json'.format(vers[4])
        # emnlp stop on train
        elif vers[3] == 3:
            data_path = 'datasets/adapted_toy64_split_0.8.json'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

        # just saves initialization
        if vers[0] == 8:
            epochs = '--num-epochs 0'

    elif vers[0] == 9:
        # see exp9.py
        pass

    elif vers[0] == 10:
        # vers[1,2,4,5] follow exp7 and vers[3] uses different rewards
        # baseline or overcomplete model
        if vers[1] == 0:
            aOutVocab = 4
            qOutVocab = 3
            remember = '--memoryless-abot'
        elif vers[1] == 1:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        elif vers[1] == 2:
            aOutVocab = 4
            qOutVocab = 3
            remember = ''
        elif vers[1] == 3:
            aOutVocab = 64
            qOutVocab = 64
            remember = '--memoryless-abot'
        elif vers[1] == 4:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''

        # baseline
        if vers[2] == 0:
            kill_epoch = 0
        # multiagent
        elif vers[2] == 1:
            kill_epoch = 0
            nabots = nqbots = 5
        # random killing
        elif vers[2] == 2:
            nabots = nqbots = 5
            kill_epoch = 5000
        # eps greedy
        elif vers[2] == 3:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'eps_greedy0.2'
        # kill oldest bots
        elif vers[2] == 4:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'oldest'
        # single agent, randomly kill qbot or abot
        elif vers[2] == 5:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'random'
        # single agent, alternate between killing qbot and abot
        elif vers[2] == 6:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'alternate'
        # single agent, kill all (random restart baseline)
        elif vers[2] == 7:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # multi agent, kill all (random restart baseline)
        elif vers[2] == 8:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi All (longer)
        elif vers[2] == 9:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi Oldest (longer)
        elif vers[2] == 10:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'
        # Multi Uniform Random (longer)
        elif vers[2] == 11:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
        # Multi Epsilon Greedy (longer)
        elif vers[2] == 12:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'eps_greedy0.2'
        # Multi Oldest (longer) (n=10)
        elif vers[2] == 13:
            nabots = nqbots = 10
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'

        # different reward strategies
        if vers[3] == 0:
            reward = '--reward both:1'
        elif vers[3] == 1:
            reward = '--reward one:0.5'
        elif vers[3] == 2:
            reward = '--reward both:1_vocab:1'
        elif vers[3] == 3:
            reward = '--reward one:0.5_vocab:1'
        elif vers[3] == 4:
            # in effect both is 1.0 and one is 0.25
            reward = '--reward one:0.25_both:1'
        elif vers[3] == 5:
            reward = '--reward both:1_entropy:0.1'
        elif vers[3] == 6:
            reward = '--reward both:1_entropy:0.01'
        elif vers[3] == 7:
            reward = '--reward both:1_entropy:0.001'
        elif vers[3] == 8:
            reward = '--reward both:1_entropy:0.0001'
        elif vers[3] == 9:
            reward = '--reward curr1:1'
        elif vers[3] == 10:
            reward = '--reward curr2:1'
        elif vers[3] == 11:
            reward = '--reward both:1_entropy:0.0000000001'
        elif vers[3] == 12:
            reward = '--reward both:1'
        elif vers[3] == 13:
            reward = '--reward both:1_entropy:0.0'

        data_path = 'datasets/comp_toy64_te0.2_folds4_'
        # different folds
        data_path += 'cv{}.json'.format(vers[4])
        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    elif vers[0] in [11]:
        save_epoch = '10000'
        reward = '--reward one:0.5'
        if vers[6] == 1:
            reward = '--reward curr2:1'
        elif vers[6] == 2:
            # in effect both is 1.0 and one is 0.25
            reward = '--reward one:0.25_both:1'
        elif vers[6] == 3:
            # in effect both is 1.0 and one is 0.25
            reward = '--reward one:0.25_both:1_vocab:1'
        elif vers[6] == 4:
            reward = '--reward both:1'
        elif vers[6] == 5:
            reward = '--reward one:0.25_both:1_base:0'

        # baseline or overcomplete model
        if vers[1] == 0:
            aOutVocab = 4
            qOutVocab = 3
            remember = '--memoryless-abot'
        elif vers[1] == 1:
            aOutVocab = 64
            qOutVocab = 64
            remember = ''
        elif vers[1] == 2:
            aOutVocab = 4
            qOutVocab = 3
            remember = ''
        elif vers[1] == 3:
            aOutVocab = 64
            qOutVocab = 64
            remember = '--memoryless-abot'
        elif vers[1] == 4:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''

        # baseline
        if vers[2] == 0:
            kill_epoch = 0
        # multiagent
        elif vers[2] == 1:
            kill_epoch = 0
            nabots = nqbots = 5
        # random killing
        elif vers[2] == 2:
            nabots = nqbots = 5
            kill_epoch = 5000
        # eps greedy
        elif vers[2] == 3:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'eps_greedy0.2'
        # kill oldest bots
        elif vers[2] == 4:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'oldest'
        # single agent, randomly kill qbot or abot
        elif vers[2] == 5:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'random'
        # single agent, alternate between killing qbot and abot
        elif vers[2] == 6:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'random'
            kill_type_policy = 'alternate'
        # single agent, kill all (random restart baseline)
        elif vers[2] == 7:
            nabots = nqbots = 1
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # multi agent, kill all (random restart baseline)
        elif vers[2] == 8:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi All (longer)
        elif vers[2] == 9:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'all'
            kill_type_policy = 'all'
        # Multi Oldest (longer)
        elif vers[2] == 10:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'
        # Multi Uniform Random (longer)
        elif vers[2] == 11:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
        # Multi Epsilon Greedy (longer)
        elif vers[2] == 12:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'eps_greedy0.2'
        # Multi Oldest (longer) (n=10)... don't use this one because it cuts the number of iterations in half (need kill_epoch=50000)
        elif vers[2] == 13:
            nabots = nqbots = 10
            epochs = '--num-epochs 5000000'
            kill_epoch = 25000
            kill_policy = 'oldest'
        elif vers[2] == 14:
            nabots = nqbots = 5
            epochs = '--num-epochs 5000000'
            kill_epoch = 100000
            kill_policy = 'oldest'
        elif vers[2] == 15:
            nabots = nqbots = 5
            epochs = '--num-epochs 3000000'
            kill_epoch = 20000
            kill_policy = 'oldest'

        if vers[3] == 0:
            data_path = 'datasets/karan_large.json'
            # different folds
            #data_path += 'cv{}.json'.format(vers[4])
        elif vers[3] == 1:
            data_path = 'datasets/one_task_large.json'
        elif vers[3] == 2:
            data_path = 'datasets/karan_large.json'
            batch_size = '--batch-size 100'
        elif vers[3] == 3:
            data_path = 'datasets/karan_large.json'
            batch_size = '--batch-size 256'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    elif vers[0] in [12]:
        save_epoch = '10000'
        lr = '--learning-rate 0.001'
        epochs = '--num-epochs 500000'

        reward = '--reward one:0.5_both:2 --rl-scale 1.0'
        if vers[2] == 1:
            reward = '--reward one:0.5_both:2_base:-1 --rl-scale 1.0'
        elif vers[2] == 2:
            reward = '--reward one:0.5_both:2_entropy:0.1 --rl-scale 1.0'
        elif vers[2] == 3:
            reward = '--reward one:0.5_oneThresh:10000_both:2_entropy:0.1 --rl-scale 1.0'
        elif vers[2] == 4:
            reward = '--reward one:1 --rl-scale 1.0'
        elif vers[2] == 5:
            reward = '--reward both:2 --rl-scale 1.0'
        elif vers[2] == 6:
            lr = '--learning-rate 0.1'
            reward = '--reward one:0.5_both:1_baseline:0.999 --rl-scale 1.0'
        elif vers[2] == 7:
            lr = '--learning-rate 0.1'
            reward = '--reward one:0.5_both:1_baseline:0.999_entropy:1 --rl-scale 1.0'
        elif vers[2] == 8:
            lr = '--learning-rate 0.1'
            reward = '--reward one:0.5_both:1_baseline:0.999_vocab:1 --rl-scale 1.0'
        elif vers[2] == 9:
            reward = '--reward one:0.5_both:2 --rl-scale 1.0 --init sorted'
        elif vers[2] == 10:
            lr = '--learning-rate 0.1'
            reward = '--reward one:0.5_both:1_baseline:0.999 --rl-scale 1.0 --init sorted'
        elif vers[2] == 11:
            lr = '--learning-rate 0.001'
            reward = '--reward both:1_entropy:0.0_base:-1 --rl-scale 1.0 --critic value --init sorted'
        elif vers[2] == 12:
            lr = '--learning-rate 0.01'
            reward = '--reward both:1_entropy:0.0_base:0 --rl-scale 1.0 --critic value'

        # large dataset with batches
        data_path = 'datasets/karan_large.json'
        if vers[3] == 1:
            data_path = 'datasets/adapted_toy64_split_0.8.json'
        batch_size = '--batch-size 100'

        # overcomplete
        aOutVocab = 64
        qOutVocab = 64
        remember = ''

        # single agent
        if vers[1] == 0:
            kill_epoch = 0
        # multiagent
        elif vers[1] == 1:
            kill_epoch = 0
            nabots = nqbots = 5
        # multiagent with killing
        elif vers[1] == 2:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'oldest'
        # multiagent with killing
        elif vers[1] == 3:
            nabots = nqbots = 5
            kill_epoch = 20000
            kill_policy = 'oldest'

    elif vers[0] in [13]:
        save_epoch = '2000'
        lr = '--learning-rate 0.01'
        epochs = '--num-epochs 500000'

        # overcomplete
        aOutVocab = 64
        qOutVocab = 64
        remember = ''

        # single agent
        if vers[1] == 0:
            kill_epoch = 0
        # multiagent
        elif vers[1] == 1:
            kill_epoch = 0
            nabots = nqbots = 5
        # multiagent with killing
        elif vers[1] == 2:
            nabots = nqbots = 5
            kill_epoch = 20000
            kill_policy = 'oldest'
        elif vers[1] == 3:
            nabots = nqbots = 5
            kill_epoch = 5000
            kill_policy = 'oldest'

        # loss
        reward = '--reward both:1_base:0 --rl-scale 1.0 --critic value'
        if vers[2] == 1:
            reward = '--reward both:1_base:0 --rl-scale 1.0 --critic value --init sorted'
        elif vers[2] == 2:
            reward = '--reward both:1_base:0_entropy:0.01 --rl-scale 1.0 --critic value'

        # dataset
        data_path = 'datasets/karan_large.json'
        batch_size = '--batch-size 100'
        if vers[3] == 1:
            data_path = 'datasets/adapted_toy64_split_0.8.json'
        elif vers[3] == 2:
            data_path = 'datasets/adapted_toy64_split_0.8.json'
            batch_size = '--batch-size 1000'

        if vers[4] == 1:
            # used to indicate that this experiment samples bots without replacement
            pass

    elif vers[0] in [14]:
        save_epoch = '1000'
        lr = '--learning-rate 0.01'
        epochs = '--num-epochs 300000'

        # overcomplete
        aOutVocab = 64
        qOutVocab = 64
        remember = ''
        # small vocab
        if vers[2] == 1:
            aOutVocab = 12
            qOutVocab = 3
            remember = ''
        # small vocab + memoryless
        elif vers[2] == 2:
            aOutVocab = 4
            qOutVocab = 3
            remember = '--memoryless-abot'

        # loss
        reward = '--reward both:1_base:0 --rl-scale 1.0 --critic value'

        # Single Agent
        if vers[1] == 0:
            kill_epoch = 0
        # Single Agent Random
        elif vers[1] == 1:
            nabots = nqbots = 1
            kill_epoch = 10000
            kill_policy = 'random'
            kill_type_policy = 'random'
        # Single Agent Alternate
        elif vers[1] == 2:
            nabots = nqbots = 1
            kill_epoch = 10000
            kill_policy = 'random'
            kill_type_policy = 'alternate'
        # Multi Agent No Replacement
        elif vers[1] == 3:
            kill_epoch = 0
            nabots = nqbots = 5
        # Multi Uniform Random
        elif vers[1] == 4:
            nabots = nqbots = 5
            kill_epoch = 10000
        # Multi Epsilon Greedy
        elif vers[1] == 5:
            nabots = nqbots = 5
            kill_epoch = 10000
            kill_policy = 'eps_greedy0.2'
        # Multi Oldest
        elif vers[1] == 6:
            nabots = nqbots = 5
            kill_epoch = 10000
            kill_policy = 'oldest'

        # dataset
        batch_size = '--batch-size 1000'
        # comp dataset w/ cross-val
        data_path = 'datasets/comp_toy64_te0.2_folds4_'
        # different folds
        data_path += 'cv{}.json'.format(vers[4])
        # emnlp dataset
        if vers[3] == 1:
            data_path = 'datasets/adapted_toy64_split_0.8.json'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])

    elif vers[0] in [15]:
        save_epoch = '1000'
        epochs = '--num-epochs 300000'

        aOutVocab = 4
        qOutVocab = 3
        remember = '--memoryless-abot'

        # loss
        lr = '--learning-rate 0.001'
        reward = '--reward both:1_base:0 --rl-scale 1.0 --critic value'

        #lr = '--learning-rate 0.0001'
        #reward = '--reward both:1_base:-10 --rl-scale 100.0 --critic none'

        # Single Agent
        if vers[1] == 0:
            kill_epoch = 0
        # Single Agent Random
        elif vers[1] == 1:
            nabots = nqbots = 1
            kill_epoch = 10000
            kill_policy = 'random'
            kill_type_policy = 'random'
        # Single Agent Alternate
        elif vers[1] == 2:
            nabots = nqbots = 1
            kill_epoch = 10000
            kill_policy = 'random'
            kill_type_policy = 'alternate'

        # dataset
        batch_size = '--batch-size 1000'
        data_path = 'datasets/adapted_toy64_split_0.8.json'

        # used for different versions of the same run
        if vers[5] != 0:
            seed = '--seed {}'.format(vers[5])


    if args.profile:
        epochs = '--num-epochs 100'
        workers = '--workers 0'

    return locals()

locals().update(config(args.EXP))

#######################################
# Pre-processing

#######################################
# Train

if args.mode == 'train':
    runcmd('{python} train.py --use-gpu \
            --data-path {data_path} \
            {remember} \
            --save-path {model_dir} \
            {batch_size} {epochs} \
            --save-epoch {save_epoch} \
            {seed} \
            --num-qbots {nqbots} \
            --num-abots {nabots} \
            --symmetric {symmetric} \
            --kill-epoch {kill_epoch} \
            --kill-policy {kill_policy} \
            --kill-type-policy {kill_type_policy} \
            {reward} \
            {task_reward_freq} \
            {teaching_freq} \
            --q-out-vocab {qOutVocab} \
            --a-out-vocab {aOutVocab}'.format(**locals()),
            external_log=log_fname)

#######################################
# Evaluate

#        --include-speed \
# if not included, evaluate will only load the best model
#        --many-epochs \

elif args.mode == 'evaluate':
    ## fixed epoch analysis (single agent)
    #runcmd('{python} evaluate.py \
    #        --epoch 39000 \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)

    ## fixed epoch analysis (multi agent)
    #runcmd('{python} evaluate.py \
    #        --epoch 199000 \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)

    ## fixed epoch language analysis (just for multi agent)
    #runcmd('{python} evaluate.py \
    #        --epoch 199000 \
    #        --analyze-lang \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)

    ## generate custom visualization data (specify manually in evaluate.py)
    #runcmd('{python} evaluate.py \
    #        --custom \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)

    ## language analysis for exp8
    #runcmd('{python} evaluate.py \
    #        --analyze-lang --use-initial-epoch \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)

    ## language analysis for exp9
    #runcmd('{python} evaluate.py \
    #        --epoch 39000 \
    #        --analyze-lang \
    #        --load-path {model_dir}'.format(**locals()),
    #        external_log=test_log_fname)
    pass

