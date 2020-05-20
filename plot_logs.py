#!/usr/bin/env python
import os.path as pth
import glob
import re
import json
import numpy as np
import visdom
from math import isnan


float_exp = r'[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|(?:inf)|(?:nan))'
epoch_re = re.compile(r'.*\[Epoch: ({0})\]\[Reward: ({0})\].*'.format(float_exp))
smooth_re = re.compile(r'.*\[Smooth \(tr,val\) \(({0}),({0})\)\].*'.format(float_exp))
metric_res = {
    'both': re.compile(r'.*\[YY \(tr,val\) \(({0}),({0})\)\]\[YO \(tr,val\) \(({0}),({0})\)\]\[OY \(tr,val\) \(({0}),({0})\)\]\[OO \(tr,val\) \(({0}),({0})\)\](?:\[<-both\]|$).*'.format(float_exp)),
    'first': re.compile(r'.*\[YY \(tr,val\) \(({0}),({0})\)\]\[YO \(tr,val\) \(({0}),({0})\)\]\[OY \(tr,val\) \(({0}),({0})\)\]\[OO \(tr,val\) \(({0}),({0})\)\]\[<-first\].*'.format(float_exp)),
    'second': re.compile(r'.*\[YY \(tr,val\) \(({0}),({0})\)\]\[YO \(tr,val\) \(({0}),({0})\)\]\[OY \(tr,val\) \(({0}),({0})\)\]\[OO \(tr,val\) \(({0}),({0})\)\]\[<-second\].*'.format(float_exp)),
    'atleast': re.compile(r'.*\[YY \(tr,val\) \(({0}),({0})\)\]\[YO \(tr,val\) \(({0}),({0})\)\]\[OY \(tr,val\) \(({0}),({0})\)\]\[OO \(tr,val\) \(({0}),({0})\)\]\[<-atleast\].*'.format(float_exp)),
}

def extract_lines(log_path, metric):
    with open(log_path, 'r') as f:
        lines = list(f)
    epochs = []
    rewards = []
    train_accs = {'YY': [], 'YO': [], 'OY': [], 'OO': [], 'Smooth': []}
    eval_accs = {'YY': [], 'YO': [], 'OY': [], 'OO': [], 'Smooth': []}
    for line in lines:
        epoch_match = epoch_re.match(line)
        if epoch_match:
            epoch = float(epoch_match.group(1))
            reward = float(epoch_match.group(2))
        smooth_match = smooth_re.match(line)
        if smooth_match:
            tr_acc = float(smooth_match.group(1))
            te_acc = float(smooth_match.group(2))
            train_accs['Smooth'].append(tr_acc)
            eval_accs['Smooth'].append(te_acc)
        metric_match = metric_res[metric].match(line)
        if metric_match:
            for i, key in enumerate(['YY', 'YO', 'OY', 'OO']):
                tr_acc = float(metric_match.group(2*i+1))
                if isnan(tr_acc):
                    tr_acc = train_accs[key][-1] if len(train_accs[key]) else 0
                te_acc = float(metric_match.group(2*i+2))
                if isnan(te_acc):
                    te_acc = eval_accs[key][-1] if len(eval_accs[key]) else 0
                train_accs[key].append(tr_acc)
                eval_accs[key].append(te_acc)
            epochs.append(epoch)
            rewards.append(reward)
    return epochs, rewards, train_accs, eval_accs


def main(args):
    '''
    e.g. argument "^exp0.[23456789]$"

    Usage:
        plot_logs.py <exp_regex> [-d] [-p] [-m metric]

    Options:
        -d  Delete matching visdom environments and exit
        -p  Just print matching experiments
        -m metric  The metric to plot (both|first|second|atleast) [default: both]
    '''
    config_file = '.visdom_config.json'
    metric = args['-m']

    # find visdom server
    with open(config_file, 'r') as f:
        config = json.load(f)
    if 'server' in config:
        server = config['server']
    if 'port' in config:
        port = int(config['port'])

    # enumerate experiment directories
    exp_dirs = glob.glob(r'data/experiments/*/')
    exp_re = re.compile(args['<exp_regex>'])
    exps = []
    envs = []
    for exp_dir in exp_dirs:
        # figure out which environments to plot
        if not pth.isdir(exp_dir):
            print('skipping {}'.format(exp_dir))
            continue
        exp_name = pth.basename(pth.dirname(exp_dir))
        if exp_re.match(exp_name) is None:
            continue
        if args['-p']:
            print(exp_name)
            continue

        # delete any existing env data
        env_name = 'emerge_' + exp_name
        envs.append(env_name)
        viz = visdom.Visdom(
            port=port,
            env=env_name,
            server=server,
        )
        viz.delete_env(env_name)
        if args['-d']:
            continue

        # parse data
        log_path = pth.join(exp_dir, 'log.txt')
        try:
            epochs, rewards, train_accs, eval_accs = extract_lines(log_path, metric)
        except FileNotFoundError:
            print('skipping {} ({} not found)'.format(exp_name, log_path))
            continue
        if len(epochs) == 0:
            print('skipping {} ({} epochs)'.format(exp_name, len(epochs)))
            continue
        print('plotting {} ({} epochs)'.format(exp_name, len(epochs)))

        # plot data
        X = np.array(epochs)
        Y = np.array([rewards]).T
        viz.line(Y=Y, X=X, opts={
                    'legend': ['rewards'],
                    'title': 'Rewards',
                    'xlabel': 'epoch',
                    'ylabel': 'avg reward',
                })
        for key in ['YY', 'YO', 'OY', 'OO', 'Smooth']:
            if key == 'Smooth':
                pass
            Y = np.array([eval_accs[key]]).T
            viz.line(Y=Y, X=X, opts={
                        'legend': ['val acc'],
                        'title': key + ' Val acc',
                        'xlabel': 'epoch',
                        'ylabel': 'acc',
                    })
            Y = np.array([train_accs[key]]).T
            viz.line(Y=Y, X=X, opts={
                        'legend': ['train acc'],
                        'title': key + ' Train acc',
                        'xlabel': 'epoch',
                        'ylabel': 'acc',
                    })

    if envs:
        pass #viz.save(envs)


if __name__ == '__main__':
    import docopt, textwrap
    args = docopt.docopt(textwrap.dedent(main.__doc__))
    main(args)
