#!/usr/bin/env python3
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch import optim

from parlai.core.params import ParlaiParser

import options
from bots import Questioner, Answerer, QABot
from dataloader import ShapesQADataset
from world import QAWorld

from evaluate import evaluate_world_fixed
from utils import batch_entropy


def decide_who_teaches(results, opt, qbot_i):
    val_accs = torch.tensor(results['val']['both'])
    qbot_merit_score = val_accs.mean(dim=1)
    qbot_merit_score[qbot_i] = -np.inf
    # choose another qbot to teach qbot_i
    if opt['teacher_policy'] == 'simple_meritocratic':
        temp = 0.5
        # sample agents with high val acc more frequently
        qbot_merit_dist = torch.softmax(qbot_merit_score / temp, dim=0)
        qbot_i = qbot_merit_dist.multinomial(1).item()
        return qbot_i
    else:
        raise Exception('unkown teacher policy {}'.format(opt['teacher_policy']))


def decide_who_dies(results, opt, world):
    if opt['kill_policy'] == 'all':
        abots = list(range(results['abots']))
        qbots = list(range(results['qbots']))
        return qbots, abots
    elif opt['kill_policy'] == 'random':
        if opt['symmetric']:
            bot_i = random.randint(0, results['qbots']-1)
            return bot_i
        else:
            # NOTE: this order is important for reproducibility
            abot_i = random.randint(0, results['abots']-1)
            qbot_i = random.randint(0, results['qbots']-1)
            return qbot_i, abot_i
    elif opt['kill_policy'] == 'oldest':
        assert not opt['symmetric']
        max_qbot_age = max(b.age for b in world.qbots)
        oldest_qbots = [i for i, b in enumerate(world.qbots) if b.age == max_qbot_age]
        qbot_i = random.choice(oldest_qbots)
        max_abot_age = max(b.age for b in world.abots)
        oldest_abots = [i for i, b in enumerate(world.abots) if b.age == max_abot_age]
        abot_i = random.choice(oldest_abots)
        return qbot_i, abot_i
    val_accs = torch.tensor(results['val']['both'])
    qbot_merit_score = val_accs.mean(dim=1)
    abot_merit_score = val_accs.mean(dim=0)
    if opt['symmetric']:
        merit_score = (qbot_merit_score + abot_merit_score) / 2
        qbot_merit_score = abot_merit_score = merit_score
    if opt['kill_policy'] == 'simple_meritocratic':
        temp = 0.5
        # sample agents with low val acc more frequently
        qbot_merit_dist = torch.softmax(-qbot_merit_score / temp, dim=0)
        qbot_i = qbot_merit_dist.multinomial(1).item()
        if opt['symmetric']:
            return qbot_i
        abot_merit_dist = torch.softmax(-abot_merit_score / temp, dim=0)
        abot_i = abot_merit_dist.multinomial(1).item()
        return qbot_i, abot_i
    elif opt['kill_policy'].startswith('eps_greedy'):
        eps = float(opt['kill_policy'][10:])
        assert 0 <= eps and eps <= 1
        kill_worstq = torch.tensor([eps, 1 - eps]).multinomial(1).item()
        if kill_worstq:
            qbot_i = qbot_merit_score.min(dim=0)[1].item()
        else:
            qbot_i = random.randint(0, results['qbots']-1)
        if opt['symmetric']:
            return qbot_i
        kill_worsta = torch.tensor([eps, 1 - eps]).multinomial(1).item()
        if kill_worsta:
            abot_i = abot_merit_score.min(dim=0)[1].item()
        else:
            abot_i = random.randint(0, results['abots']-1)
        return qbot_i, abot_i
    else:
        raise Exception('unknown kill policy {}'.format(opt['kill_policy']))


def train(world, dataset, verbose=True):
    OPT = world.opt
    q_optimizers = [optim.Adam(qbot.parameters(),
                              lr=OPT['learning_rate']) for qbot in world.qbots]
    if OPT['symmetric']:
        a_optimizers = q_optimizers
    else:
        a_optimizers = [optim.Adam(abot.parameters(),
                                  lr=OPT['learning_rate']) for abot in world.abots]
    def replace_abot(i, epoch):
        if verbose:
            print('replacing abot {}'.format(i))
        abot = Answerer(OPT, generation=epoch // OPT['kill_epoch'])
        world.abots[i] = abot.cuda() if OPT['use_gpu'] else abot 
        a_optimizers[i] = optim.Adam(abot.parameters(), lr=OPT['learning_rate'])
    def replace_qbot(i, epoch):
        if verbose:
            print('replacing qbot {}'.format(i))
        qbot = Questioner(OPT, generation=epoch // OPT['kill_epoch'])
        world.qbots[i] = qbot.cuda() if OPT['use_gpu'] else qbot
        q_optimizers[i] = optim.Adam(qbot.parameters(), lr=OPT['learning_rate'])
    def replace_bot(i, epoch):
        if verbose:
            print('replacing bot {}'.format(i))
        bot = QABot(OPT, generation=epoch // OPT['kill_epoch'])
        bot = bot.cuda() if OPT['use_gpu'] else bot
        world.qbots[i] = bot
        world.abots[i] = bot
        q_optimizers[i] = optim.Adam(bot.parameters(), lr=OPT['learning_rate'])

    NUM_ITER_PER_EPOCH = max(0, int(np.ceil(len(dataset) / OPT['batch_size'])))

    matches = {'train': None, 'val': None}
    accuracy = {'train': None, 'val': None}
    train_checkpoints = []
    best_val_acc = 0
    best_val_epoch = 0
    best_train_acc = 0
    best_train_epoch = 0
    log_data = []
    eval_results = None

    # save the intiailization in a checkpoint
    save_path = os.path.join(OPT['save_path'], 'initial_world.pth')
    world.save_agents(save_path)

    # this reward tensor is re-used every iteration
    reward = torch.Tensor(OPT['batch_size']).fill_(-10 * OPT['rl_scale'])
    cumulative_reward = None
    if OPT.get('use_gpu'):
        reward = reward.cuda()
    reward_weights = [rew.split(':') for rew in OPT['reward'].split('_')]
    reward_weights = {k: float(v) for k, v in reward_weights}
    # vocab prior initialization
    alpha = 10.0
    q_pcount = alpha / OPT['q_out_vocab']
    a_pcount = alpha / OPT['a_out_vocab']
    word_prior = {
        # initialize with pseudo counts
        'q': q_pcount * torch.ones(OPT['q_out_vocab']).to(reward),
        'a': a_pcount * torch.ones(OPT['a_out_vocab']).to(reward),
    }

    for epoch_id in range(OPT['num_epochs']):
        for iter_id in range(NUM_ITER_PER_EPOCH):
            qbot_i, abot_i = world.sample_bots()
            do_teaching = (OPT['teaching_freq'] and eval_results and
                           epoch_id % OPT['teaching_freq'] == 0)
            do_task_reward = (OPT['task_reward_freq'] and
                              epoch_id % OPT['task_reward_freq'] == 0)
            if matches.get('train') is not None:
                batch = dataset.random_batch('train', matches['train'])
            else:
                batch = dataset.random_batch('train')

            # figure out what teacher qbot says
            if do_teaching:
                qteacher_i = decide_who_teaches(eval_results, OPT, qbot_i)
                world.set_qbot(qteacher_i)
                world.qbot.observe({'batch': batch, 'episode_done': True})
                for _ in range(OPT['num_rounds']):
                    world.parley()
                teacher_dist = world.qbot.action_distributions
                world.set_qbot(qbot_i)

            # dialog
            world.qbot.observe({'batch': batch, 'episode_done': True})
            for _ in range(OPT['num_rounds']):
                world.parley()
            # solve the task
            guess_token, guess_distr = world.qbot.predict(batch['task'], 2)

            # compute the loss
            q_obs = {'episode_done': True}
            a_obs = {'episode_done': True}
            # reward for Qbot task completion
            if do_task_reward:
                base = reward_weights.get('base', -10)
                # reward formulation and reinforcement
                reward.fill_(base * OPT['rl_scale'])
                # both attributes need to match
                first_match = (guess_token[0] == batch['labels'][:, 0])
                second_match = (guess_token[1] == batch['labels'][:, 1])
                one_weight = reward_weights.get('one', 0.0)
                both_weight = reward_weights.get('both', 0.0)
                if reward_weights.get('curr1', 0.0):
                    # allow positive reward for one correct attribute initially
                    if epoch_id > 25000:
                        both_weight = 1.0
                    else:
                        one_weight = 0.5
                if reward_weights.get('curr2', 0.0):
                    # allow positive reward for one correct attribute at the beginning of every generation
                    NQ = len(world.qbots)
                    NA = len(world.abots)
                    # TODO: this should depend on what kill_epoch is set to in
                    # the killing experiments this one is compared to (20000 so
                    # far, but it could change)
                    if OPT['kill_epoch'] == 0:
                        if epoch_id < 20000:
                            one_weight = 0.5
                        else:
                            both_weight = 1.0
                    elif (epoch_id % OPT['kill_epoch']) < (OPT['kill_epoch'] // max(NQ, NA)):
                        one_weight = 0.5
                    else:
                        both_weight = 1.0
                if one_weight:
                    r = one_weight * OPT['rl_scale']
                    reward[first_match ^ second_match] = r
                    reward[first_match & second_match] = 2 * r
                if both_weight:
                    r = both_weight * OPT['rl_scale']
                    # NOTE: this is meant to overwrite the 2*r from 'one'
                    reward[first_match & second_match] = r
                q_obs['reward'] = reward if world.qbot.training else None
                a_obs['reward'] = reward if world.abot.training else None
                if reward_weights.get('vocab', 0.0):
                    # TODO: maybe try different counts for q0/a0 and q1/a1
                    # NOTE: bincount could be a source of indeterminism
                    word_prior['q'] += world.qbot.actions[0].bincount(minlength=OPT['q_out_vocab']).to(torch.float)
                    word_prior['q'] += world.qbot.actions[1].bincount(minlength=OPT['q_out_vocab']).to(torch.float)
                    q_denom = (word_prior['q'].sum() + alpha - 1)
                    q_word_logprobs = torch.log(word_prior['q'] / q_denom)
                    q_ll_q0 = q_word_logprobs[world.qbot.actions[0]]
                    q_ll_q1 = q_word_logprobs[world.qbot.actions[1]]
                    r = 0.5 * q_ll_q0 + 0.5 * q_ll_q1
                    if q_obs['reward'] is not None:
                        q_obs['reward'] += r * OPT['rl_scale'] * reward_weights['vocab']

                    word_prior['a'] += world.abot.actions[0].bincount(minlength=OPT['a_out_vocab']).to(torch.float)
                    word_prior['a'] += world.abot.actions[1].bincount(minlength=OPT['a_out_vocab']).to(torch.float)
                    a_denom = (word_prior['a'].sum() + alpha - 1)
                    a_word_logprobs = torch.log(word_prior['a'] / a_denom)
                    a_ll_a0 = a_word_logprobs[world.abot.actions[0]]
                    a_ll_a1 = a_word_logprobs[world.abot.actions[1]]
                    r = 0.5 * a_ll_a0 + 0.5 * a_ll_a1
                    if a_obs['reward'] is not None:
                        a_obs['reward'] += r * OPT['rl_scale'] * reward_weights['vocab']
                if reward_weights.get('entropy', 0.0):
                    entropy_loss = 0
                    q0p = world.qbot.action_distributions[0].probs
                    q1p = world.qbot.action_distributions[1].probs
                    entropy_loss += batch_entropy(q0p).mean()
                    entropy_loss += batch_entropy(q1p).mean()
                    q_obs['extra_loss'] = -entropy_loss * reward_weights['entropy']
                    entropy_loss = 0
                    a0p = world.abot.action_distributions[0].probs
                    a1p = world.abot.action_distributions[1].probs
                    entropy_loss += batch_entropy(a0p).mean()
                    entropy_loss += batch_entropy(a1p).mean()
                    a_obs['extra_loss'] = -entropy_loss * reward_weights['entropy']
            # loss for saying things like the teacher Qbot
            if do_teaching:
                q_obs['teacher_dist'] = teacher_dist

            # observe loss and backprop
            q_optimizer = q_optimizers[qbot_i]
            q_optimizer.zero_grad()
            a_optimizer = a_optimizers[abot_i]
            a_optimizer.zero_grad()
            world.qbot.observe(q_obs)
            world.abot.observe(a_obs)
            # gradient step
            if world.qbot.training:
                q_optimizer.step()
            if world.abot.training:
                a_optimizer.step()
            world.age_bots()

            # logging
            # cumulative reward
            batch_reward = torch.mean(reward) / OPT['rl_scale']
            if not cumulative_reward:
                cumulative_reward = batch_reward
            cumulative_reward = 0.95 * cumulative_reward + 0.05 * batch_reward
            # evaluate and print less frequently
            if (NUM_ITER_PER_EPOCH * epoch_id + iter_id) % 100 == 0:
                # evaluate the current models
                world.qbot.eval()
                world.abot.eval()
                eval_results = evaluate_world_fixed(world, dataset, OPT['kill_epoch'], OPT['kill_epoch'])
                log_data.append((epoch_id, eval_results))
                if not accuracy['train'] or np.isnan(accuracy['train']):
                    accuracy['train'] = eval_results['train']['old_qbot_old_abot']['both']
                accuracy['train'] = 0.9 * accuracy['train'] + 0.1 * eval_results['train']['old_qbot_old_abot']['both']
                # TODO: figure out why this turns out to be nan sometimes
                #import pdb; pdb.set_trace()
                if not accuracy['val'] or np.isnan(accuracy['val']):
                    accuracy['val'] = eval_results['val']['old_qbot_old_abot']['both']
                accuracy['val'] = 0.9 * accuracy['val'] + 0.1 * eval_results['val']['old_qbot_old_abot']['both']
                world.qbot.train()
                world.abot.train()
                # log a few metrics
                timestamp = datetime.strftime(datetime.utcnow(), '%a, %d %b %Y %X')
                line = '[{}][Epoch: {:.2f}]'.format(timestamp, epoch_id)
                line += '[Reward: {:.4f}]'.format(cumulative_reward)
                line += '[Smooth (tr,val) ({:.2f},{:.2f})]'.format(accuracy['train'], accuracy['val'])
                for key in ['both', 'first', 'second', 'atleast']:
                    line += '[YY (tr,val) '
                    line += '({:.2f},{:.2f})]'.format(*[eval_results[dtype]['young_qbot_young_abot'][key] for dtype in ['train', 'val']])
                    line += '[YO (tr,val) '
                    line += '({:.2f},{:.2f})]'.format(*[eval_results[dtype]['young_qbot_old_abot'][key] for dtype in ['train', 'val']])
                    line += '[OY (tr,val) '
                    line += '({:.2f},{:.2f})]'.format(*[eval_results[dtype]['old_qbot_young_abot'][key] for dtype in ['train', 'val']])
                    line += '[OO (tr,val) '
                    line += '({:.2f},{:.2f})]'.format(*[eval_results[dtype]['old_qbot_old_abot'][key] for dtype in ['train', 'val']])
                    line += '[<-{}]'.format(key)
                if verbose:
                    print(line)
                # checkpoint criteria
                if accuracy['val'] > best_val_acc: # NOTE: bool(nan > 0) == False
                    best_val_acc = accuracy['val']
                    best_val_epoch = epoch_id
                    save_path = os.path.join(OPT['save_path'], 'best_world.pth')
                    world.save_agents(save_path, extra={
                                        'epoch': epoch_id,
                                        'best_val_acc': accuracy['val']})
                if accuracy['train'] > best_train_acc:
                    best_train_acc = accuracy['train']
                    best_train_epoch = epoch_id
                    save_path = os.path.join(OPT['save_path'], 'best_train_world.pth')
                    world.save_agents(save_path, extra={
                                        'epoch': epoch_id,
                                        'best_train_acc': accuracy['train']})

        # regularly save checkpoints
        if epoch_id % OPT['save_epoch'] == 0:
            save_path = os.path.join(OPT['save_path'], 'world_epoch_%s.pth' % str(epoch_id).zfill(5))
            world.save_agents(save_path)

        # stop when appropriate
        if epoch_id - best_val_epoch > 200000:
            break

        # usher in a new generation
        if OPT['kill_epoch'] > 0 and epoch_id % OPT['kill_epoch'] == 0:
            who_dies = decide_who_dies(eval_results, OPT, world)
            if OPT['symmetric']:
                kill_bot_i = who_dies
                replace_bot(kill_bot_i, epoch_id)
            else:
                kill_qbot_i, kill_abot_i = who_dies
                if OPT['kill_type_policy'] == 'both':
                    replace_abot(kill_abot_i, epoch_id)
                    replace_qbot(kill_qbot_i, epoch_id)
                elif OPT['kill_type_policy'] == 'random':
                    if random.randint(0, 1) == 0:
                        replace_abot(kill_abot_i, epoch_id)
                    else:
                        replace_qbot(kill_qbot_i, epoch_id)
                elif OPT['kill_type_policy'] == 'alternate':
                    kill_iter = epoch_id // OPT['kill_epoch']
                    if kill_iter % 2 == 0:
                        replace_abot(kill_abot_i, epoch_id)
                    else:
                        replace_qbot(kill_qbot_i, epoch_id)
                elif OPT['kill_type_policy'] == 'all':
                    for ai in kill_abot_i:
                        replace_abot(ai, epoch_id)
                    for qi in kill_qbot_i:
                        replace_qbot(qi, epoch_id)
                else:
                    raise Exception('Unknown kill_type_policy {}'.format(OPT['kill_type_policy']))

    # save final world checkpoint with a time stamp
    timestamp = datetime.strftime(datetime.utcnow(), '%d-%b-%Y-%X')
    final_save_path = os.path.join(OPT['save_path'], 'final_world_{}.pth'.format(timestamp))
    if verbose:
        print('Saving at final world at: {}'.format(final_save_path))
    world.save_agents(final_save_path)
    return log_data


def main(OPT):
    '''Script for training the questioner and answerer agents in dialog world.
    Both agents hold multiple rounds of dialoues per episode, after which qbot
    makes a prediction about the attributes of image,  according to the
    assigned task.

    Few variables defined here are explained:

    Global Variables
    ----------------
    OPT : dict
        Command-line arguments. Refer ``options.py``

    matches : dict
        Has keys 'train' and 'val'. Contains tensor of booleans as values. i-th true value represents
        i-th example's ground truth matching prediction in previous iteration. This dict is useful
        for sampling negative examples for next iteration training.
    accuracy : dict
        Has keys 'train' and 'val'. Will have training and validation accuracies updated every epoch.
        This dict is useful for early stopping mechanism. Training stops if training accuracy hits 1.

    reward : torch.FloatTensor or torch.cuda.FloatTensor
        Tensor of length equal to batch size, sets reward 1 for correctly classified example and -10
        for negatively classified sample. Re-used every episode.
    cumulative_reward : float
        Scalar reward for both the bots. Same for both bots as the game is perfectly cooperative.

    dataset : ShapesQADataset (torch.utils.data.Dataset)
    questioner : Questioner (parlai.core.agents.Agent, nn.Module)
    answerer : Answerer (parlai.core.agents.Agent, nn.Module)
    world : QAWorld (parlai.core.worlds.DialogPartnerWorld)
    optimizer : optim.Adam
    '''

    # seed random for reproducibility
    #if OPT.get('use_gpu'):
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(OPT['seed'])
    torch.manual_seed(OPT['seed'])
    random.seed(OPT['seed'])

    # setup dataset and opts
    dataset = ShapesQADataset(OPT)
    # pull out few attributes from dataset in main opts for other bots to use
    OPT['props'] = dataset.properties
    OPT['task_vocab'] = len(dataset.task_defn)

    # make a directory to save checkpoints
    os.makedirs(OPT['save_path'], exist_ok=True)

    # setup experiment
    if OPT['symmetric']:
        assert OPT['q_out_vocab'] == OPT['a_out_vocab']
        bots = []
        num_bots = max(OPT['num_qbots'], OPT['num_abots'])
        for _ in range(num_bots):
            bots.append(QABot(OPT))
        questioners = list(bots)
        answerers = list(bots)
    else:
        questioners = []
        for _ in range(OPT['num_qbots']):
            questioners.append(Questioner(OPT))
        answerers = []
        for _ in range(OPT['num_abots']):
            answerers.append(Answerer(OPT))
    if OPT.get('use_gpu'):
        questioners = [q.cuda() for q in questioners]
        answerers = [a.cuda() for a in answerers]

    print('Questioner and Answerer Bots: ')
    print('{} Questioners'.format(len(questioners)))
    print(questioners[0])
    print('{} Answerers'.format(len(answerers)))
    print(answerers[0])

    world = QAWorld(OPT, questioners, answerers)
    train(world, dataset)


if __name__ == '__main__':
    OPT = options.read()
    main(OPT)
