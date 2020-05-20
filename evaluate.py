import json
import itertools
import os.path as pth
import glob
import shlex
import tempfile
import random

import numpy as np
import torch
from torch.autograd import Variable
from parlai.core.params import ParlaiParser

from bots import Questioner, Answerer, QABot
from dataloader import ShapesQADataset
from world import QAWorld


def stop_strategy(model_dir, map_location=None):
    world = load_world(model_dir, 0, map_location=map_location)[0]
    if 'adapted_toy64_split' in world.opt['data_path']:
        return 'stop_at_train'
    elif 'comp_toy64_split' in world.opt['data_path']:
        return 'stop_at_train'
    elif 'emnlp_toy64' in world.opt['data_path']:
        return 'stop_at_val'
    elif 'comp_toy64_te0.2_folds' in world.opt['data_path']:
        return 'stop_at_val'
    elif 'karan_large' in world.opt['data_path'] or 'test_dataset' in world.opt['data_path']:
        return 'stop_at_train'
    else:
        raise Exception('Use some other hack to specify stopping criteria')

def get_final_epoch(model_dir, map_location=None):
    model_path = find_model(model_dir, None, map_location=map_location)
    world_dict = torch.load(model_path, map_location=map_location)
    return world_dict['epoch']

def get_final_dict(model_dir, map_location=None):
    model_path = find_model(model_dir, None, map_location=map_location)
    world_dict = torch.load(model_path, map_location=map_location)
    return world_dict

def find_model(model_dir, epoch=None, map_location=None):
    if epoch == 'initial':
        model_path = pth.join(model_dir, 'initial_world.pth')
    elif epoch is not None:
        model_path = pth.join(model_dir, 'world_epoch_{:0>5d}.pth'.format(epoch))
    else:
        strategy = stop_strategy(model_dir, map_location=map_location)
        if strategy == 'stop_at_val':
            model_path = pth.join(model_dir, 'best_world.pth')
        elif strategy == 'stop_at_train':
            model_path = pth.join(model_dir, 'best_train_world.pth')
        if not pth.exists(model_path):
            paths = glob.glob(pth.join(model_dir, 'world_epoch_*.pth'))
            epochs = [int(pth.basename(path)[12:-4]) for path in paths]
            import pdb; pdb.set_trace()
    return model_path

#-------------------------------------------------------------------------------------------------
# setup dataset and world from checkpoint
#-------------------------------------------------------------------------------------------------

def load_world(model_dir, epoch=None, training=False, map_location=None, init_ages=False):
    model_path = find_model(model_dir, epoch)
    world_dict = torch.load(model_path, map_location=map_location)

    dataset = ShapesQADataset(world_dict['opt'])
    if world_dict['opt'].get('symmetric'):
        bots = []
        for bstate in world_dict['qbots']:
            bot = QABot(world_dict['opt'])
            # TODO: gpu use should depend on eval script settings
            if world_dict['opt'].get('use_gpu'):
                bot = bot.cuda()
            bots.append(bot)
            bot.train(training)
            bot.load_state_dict(bstate)
        questioners = list(bots)
        answerers = list(bots)
    else:
        questioners = []
        for qi, qstate in enumerate(world_dict['qbots']):
            questioner = Questioner(world_dict['opt'])
            if world_dict['opt'].get('use_gpu'):
                questioner = questioner.cuda()
            if init_ages:
                questioner.age = world_dict['qbot_ages'][qi] if 'qbot_ages' in world_dict else 0
                questioner.generation = world_dict['qbot_generations'][qi] if 'qbot_generations' in world_dict else 0
            questioners.append(questioner)
            questioner.train(training)
            questioner.load_state_dict(qstate)
        answerers = []
        for ai, astate in enumerate(world_dict['abots']):
            answerer = Answerer(world_dict['opt'])
            if world_dict['opt'].get('use_gpu'):
                answerer = answerer.cuda()
            if init_ages:
                answerer.age = world_dict['abot_ages'][ai] if 'abot_ages' in world_dict else 0
                answerer.generation = world_dict['abot_generations'][ai] if 'abot_generations' in world_dict else 0
            answerer.load_state_dict(astate)
            answerer.train(training)
            answerers.append(answerer)

    world = QAWorld(world_dict['opt'], questioners, answerers)
    print('Loaded world from checkpoint: {}'.format(model_path))
    return world, dataset, model_path


def evaluate_world_fixed(world, dataset, q_t=0, a_t=0, report_script=False,
                         analyze_lang=False, how_to_sample=0):
    '''
    Measure how well each pair of bots in the world converses with eachother.
    This is fixed in contrast with evaluate_world() because that function
    also evaluates how well bots teach and learn (so the learning bot changes).

    q_t: Threshold age for qbot. Qbots of this age or greater are "old"
    a_t: Threshold age for abot. Abots of this age or greater are "old"
    '''

    Nq = len(world.qbots)
    Na = len(world.abots)
    assert how_to_sample >= 0, 'either 0 for argmax or >0 for sampling'
    nsamples = 1 if how_to_sample == 0 else how_to_sample

    # script of conversation
    script = {'train': [[[None for _ in range(Na)] for _ in range(Nq)] for _ in range(nsamples)],
                'val': [[[None for _ in range(Na)] for _ in range(Nq)] for _ in range(nsamples)],
               'test': [[[None for _ in range(Na)] for _ in range(Nq)] for _ in range(nsamples)]}

    # different accuracy metrics for train and val data
    first_accuracy = {'train': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                        'val': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                       'test': [torch.zeros(Nq, Na) for _ in range(nsamples)]}
    second_accuracy = {'train': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                         'val': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                        'test': [torch.zeros(Nq, Na) for _ in range(nsamples)]}
    atleast_accuracy = {'train': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                          'val': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                         'test': [torch.zeros(Nq, Na) for _ in range(nsamples)]}
    both_accuracy = {'train': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                       'val': [torch.zeros(Nq, Na) for _ in range(nsamples)],
                      'test': [torch.zeros(Nq, Na) for _ in range(nsamples)]}

    #-------------------------------------------------------------------------------------------------
    # test agents
    #-------------------------------------------------------------------------------------------------
    if hasattr(world, 'qbot'):
        org_qbot, org_abot = world.qbot, world.abot
    else:
        org_qbot = None
    for qbot_i, abot_i in itertools.product(range(Nq), range(Na)):
        world.set_qbot(qbot_i)
        world.set_abot(abot_i)
        org_qbot_training = world.qbot.training
        org_abot_training = world.abot.training
        world.qbot.train(how_to_sample != 0)
        world.abot.train(how_to_sample != 0)

        for dtype in ['train', 'val', 'test']:
            if dtype not in dataset.data:
                continue
            batch = dataset.complete_data(dtype)
            #assert world.qbot.training == (how_to_sample != 0), 'make sure the bots are in sampling mode using train()'
            #assert world.abot.training == (how_to_sample != 0), 'make sure the bots are in sampling mode using train()'

            for sample_i in range(nsamples):
                # make variables volatile because graph construction is not required for eval
                world.qbot.observe({'batch': batch, 'episode_done': True})

                with torch.no_grad():
                    for _ in range(world.opt['num_rounds']):
                        world.parley()
                    guess_token, guess_distr = world.qbot.predict(batch['task'], 2)

                # check how much do first attribute, second attribute, both and at least one match
                first_match = guess_token[0].data == batch['labels'][:, 0].long()
                second_match = guess_token[1].data == batch['labels'][:, 1].long()
                both_matches = first_match & second_match
                atleast_match = first_match | second_match

                # compute accuracy according to matches
                first_accuracy[dtype][sample_i][qbot_i, abot_i] = 100 * torch.mean(first_match.float())
                second_accuracy[dtype][sample_i][qbot_i, abot_i] = 100 * torch.mean(second_match.float())
                atleast_accuracy[dtype][sample_i][qbot_i, abot_i] = 100 * torch.mean(atleast_match.float())
                both_accuracy[dtype][sample_i][qbot_i, abot_i] = 100 * torch.mean(both_matches.float())

                if report_script:
                    script[dtype][sample_i][qbot_i][abot_i] = dataset.talk_to_script(world.get_acts(), guess_token, batch)

        world.qbot.train(org_qbot_training)
        world.abot.train(org_abot_training)

    results = {
        'qbots': Nq,
        'abots': Na,
        # 0 - argmax, otherwise the number of samples
        'how_to_sample': how_to_sample,
        'ages': {
            'qbot': [qbot.age for qbot in world.qbots],
            'abot': [abot.age for abot in world.abots],
        },
        'generations': {
            'qbot': [qbot.generation for qbot in world.qbots],
            'abot': [abot.generation for abot in world.abots],
        },
        'train': [None for i in range(nsamples)],
        'val':   [None for i in range(nsamples)],
        'test':  [None for i in range(nsamples)],
    }

    qbot_age = torch.tensor(results['ages']['qbot'])
    abot_age = torch.tensor(results['ages']['abot'])
    # masks that select pairs with young/old qbot and young/old abot
    young_young = (qbot_age < q_t)[:, None].mm((abot_age < a_t)[None, :])
    young_old   = (qbot_age < q_t)[:, None].mm((abot_age >= a_t)[None, :])
    old_young   = (qbot_age >= q_t)[:, None].mm((abot_age < a_t)[None, :])
    old_old     = (qbot_age >= q_t)[:, None].mm((abot_age >= a_t)[None, :])
    for dtype in ['train', 'val', 'test']:
        if dtype not in both_accuracy:
            continue
        for sample_i in range(nsamples):

            results[dtype][sample_i] = {
                'both': both_accuracy[dtype][sample_i].tolist(),
                'first': first_accuracy[dtype][sample_i].tolist(),
                'second': second_accuracy[dtype][sample_i].tolist(),
                'atleast': atleast_accuracy[dtype][sample_i].tolist(),
                'script': script[dtype][sample_i],

                'young_qbot_young_abot': {
                    'both': both_accuracy[dtype][sample_i][young_young].mean().item(),
                    'first': first_accuracy[dtype][sample_i][young_young].mean().item(),
                    'second': second_accuracy[dtype][sample_i][young_young].mean().item(),
                    'atleast': atleast_accuracy[dtype][sample_i][young_young].mean().item(),
                },

                'young_qbot_old_abot': {
                    'both': both_accuracy[dtype][sample_i][young_old].mean().item(),
                    'first': first_accuracy[dtype][sample_i][young_old].mean().item(),
                    'second': second_accuracy[dtype][sample_i][young_old].mean().item(),
                    'atleast': atleast_accuracy[dtype][sample_i][young_old].mean().item(),
                },

                'old_qbot_young_abot': {
                    'both': both_accuracy[dtype][sample_i][old_young].mean().item(),
                    'first': first_accuracy[dtype][sample_i][old_young].mean().item(),
                    'second': second_accuracy[dtype][sample_i][old_young].mean().item(),
                    'atleast': atleast_accuracy[dtype][sample_i][old_young].mean().item(),
                },

                'old_qbot_old_abot': {
                    'both': both_accuracy[dtype][sample_i][old_old].mean().item(),
                    'first': first_accuracy[dtype][sample_i][old_old].mean().item(),
                    'second': second_accuracy[dtype][sample_i][old_old].mean().item(),
                    'atleast': atleast_accuracy[dtype][sample_i][old_old].mean().item(),
                },
            }

    if org_qbot:
        world.qbot, world.abot = org_qbot, org_abot

    if analyze_lang:
        import lang_analysis
        print('Analyzing language...')
        lang_analyzer = lang_analysis.LanguageAnalyzer({'expNA': {
                                                            'null': results
                                                        }},
                                                split=['test'])
        # Na x Na matrices of "distances" between languages
        abot_lang_dists = lang_analyzer.lang_dists('expNA', bot_type='abot')
        abot_lang_ents = lang_analyzer.lang_entropy('expNA', bot_type='abot')
        # TODO: add this back in once it makes sense
        #abot_with_qword_lang_dists = lang_analyzer.lang_dists('expNA',
        #                                        bot_type='abot', cond_word=True)
        qbot_lang_dists = lang_analyzer.lang_dists('expNA', bot_type='qbot')
        qbot_lang_ents = lang_analyzer.lang_entropy('expNA', bot_type='qbot')
        # lighten up <...>lang_analysis.json file size
        del results['train']
        del results['val']
        del results['test']
        results['test'] = {
            'abot_lang_dists': abot_lang_dists.tolist(),
            'abot_lang_entropies': abot_lang_ents.tolist(),
            #'abot_with_qword_lang_dists': abot_with_qword_lang_dists.tolist(),
            'qbot_lang_dists': qbot_lang_dists.tolist(),
            'qbot_lang_entropies': qbot_lang_ents.tolist(),
        }
        print('language analyzed.')

    # unpack argmax inference for backwards compatibility
    if how_to_sample == 0:
        for dtype in ['train', 'val', 'test']:
            assert len(results[dtype]) == 1
            results[dtype] = results[dtype][0]

    return results


def measure_learning_speed(init_qbot, init_abot, dataset, world_opt, train_code):
    new_opt = dict(world_opt)
    # leave these the same: learning_rate, bot parameters, use_gpu, batch_size,
    # rl_scale, num_rounds
    new_opt['kill_epoch'] = 0
    new_opt['num_epochs'] = 3000
    new_opt['save_epoch'] = new_opt['num_epochs']
    tmpdir = tempfile.TemporaryDirectory()
    new_opt['save_path'] = tmpdir.name

    # seed random for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(new_opt['seed'])
    torch.manual_seed(new_opt['seed'])
    random.seed(new_opt['seed'])

    # create a world with just two bots configured so only one learns
    qbot = Questioner(new_opt, generation=0)
    qbot = qbot.cuda() if new_opt['use_gpu'] else qbot
    if train_code[0] == '0':
        qbot.load_state_dict(init_qbot.state_dict())
        qbot.train(False)
    elif train_code[0] == '1':
        qbot.train(True)
    elif train_code[0] == '2':
        qbot.train(True)
        qbot.load_state_dict(init_qbot.state_dict())
    abot = Answerer(new_opt, generation=0)
    abot = abot.cuda() if new_opt['use_gpu'] else abot
    if train_code[1] == '0':
        abot.train(False)
        abot.load_state_dict(init_abot.state_dict())
    elif train_code[1] == '1':
        abot.train(True)
    elif train_code[1] == '2':
        abot.train(True)
        abot.load_state_dict(init_abot.state_dict())
    world = QAWorld(new_opt, [qbot], [abot])

    # did the fine-tuned pair get better train/val performance than the initial pair
    # plot: how many epochs did it take to get to performance x?

    # avoid circular imports
    import train
    log_data = train.train(world, dataset, verbose=False)
    for epoch_id, result in log_data:
        for dtype in ['train', 'val', 'test']:
            if dtype not in result:
                continue
            result[dtype] = result[dtype]['old_qbot_old_abot']
    tmpdir.cleanup()
    return log_data


def evaluate_world(world, dataset, report_script=False, speed=False,
                   analyze_lang=False):
    '''
    Measure a number of metrics about the bots in this world, including
    generalization to unseen instances and teachability.
    '''
    results = evaluate_world_fixed(world, dataset, report_script=report_script,
                                   analyze_lang=analyze_lang,
                                   how_to_sample=0 if not analyze_lang else 10)
    if not speed:
        return results

    Nq = len(world.qbots)
    Na = len(world.abots)
    # train qbot (1=True, 0=False); train abot (1=True, 0=False)
    learning_speed_results = {
        '01': [[None for _ in range(Na)] for _ in range(Nq)],
        '10': [[None for _ in range(Na)] for _ in range(Nq)],
        '12': [[None for _ in range(Na)] for _ in range(Nq)],
        '21': [[None for _ in range(Na)] for _ in range(Nq)],
        '22': [[None for _ in range(Na)] for _ in range(Nq)],
    }
    for qbot_i, abot_i in itertools.product(range(Nq), range(Na)):
        world.set_qbot(qbot_i)
        world.set_abot(abot_i)
        print('Training ({}, {}) fix Q, train A from scratch'.format(qbot_i, abot_i))
        learning_speed_results['01'][qbot_i][abot_i] = measure_learning_speed(world.qbot, world.abot, dataset, world.opt, '01')
        print('Training ({}, {}) train Q from scratch, fix A'.format(qbot_i, abot_i))
        learning_speed_results['10'][qbot_i][abot_i] = measure_learning_speed(world.qbot, world.abot, dataset, world.opt, '10')
        print('Training ({}, {}) train Q from scratch, train A'.format(qbot_i, abot_i))
        learning_speed_results['12'][qbot_i][abot_i] = measure_learning_speed(world.qbot, world.abot, dataset, world.opt, '12')
        print('Training ({}, {}) train Q, train A from scratch'.format(qbot_i, abot_i))
        learning_speed_results['21'][qbot_i][abot_i] = measure_learning_speed(world.qbot, world.abot, dataset, world.opt, '21')
        print('Training ({}, {}) train Q, train A'.format(qbot_i, abot_i))
        learning_speed_results['22'][qbot_i][abot_i] = measure_learning_speed(world.qbot, world.abot, dataset, world.opt, '22')
    return {
        'fixed_results': results,
        'learning_speed_results': learning_speed_results,
    }


def main(OPT):
    init_ages = False
    if OPT['custom']:
        save_path = pth.join(OPT['load_path'], 'results_custom.json')
    elif OPT['epoch'] and OPT['generations']:
        epoch = OPT['epoch']
        save_path = pth.join(OPT['load_path'], 'results_gen_{}.json'.format(epoch))
    elif OPT['epoch'] and OPT['analyze_lang']:
        epoch = OPT['epoch']
        save_path = pth.join(OPT['load_path'], 'results_lang_analysis_ep_{}.json'.format(epoch))
    elif OPT['epoch']:
        epoch = OPT['epoch']
        save_path = pth.join(OPT['load_path'], 'results_ep_{}.json'.format(epoch))
    elif OPT['generations']:
        save_path = pth.join(OPT['load_path'], 'results_gen.json')
    elif OPT['include_speed']:
        save_path = pth.join(OPT['load_path'], 'results_speed2.json')
    elif OPT['analyze_lang']:
        save_path = pth.join(OPT['load_path'], 'results_lang_analysis.json')
    elif OPT['many_epochs']:
        save_path = pth.join(OPT['load_path'], 'results_many_epochs.json')
    else:
        save_path = pth.join(OPT['load_path'], 'results.json')
    results_by_epoch = {}
    if OPT['custom']:
        init_ages = True
        epochs = [0, 1000, 3000, 5000, 10000, 15000, 176000, 180000, 185000, 190000]
    elif OPT['generations']:
        init_ages = True
        final_dict = get_final_dict(OPT['load_path'])
        if OPT['epoch']:
            final_epoch = OPT['epoch'] + 1
        else:
            final_epoch = final_dict['epoch']
        kill_epoch = final_dict['opt']['kill_epoch']
        if kill_epoch == 0:
            kill_epoch = 5000
        epochs = range(kill_epoch, final_epoch, kill_epoch)
        epochs = [e - 1000 for e in epochs][::-1]
    elif OPT['epoch']:
        epochs = [OPT['epoch']]
    elif OPT['many_epochs'] and not OPT['use_initial_epoch']:
        final_epoch = get_final_epoch(OPT['load_path'])
        # note that the None (best) epoch might not be a multiple of 1000
        # TODO: use final_epoch instead of None so that information (final_epoch) is in the result file
        epochs = list(range(0, final_epoch, 1000)) + [None]
    elif OPT['use_initial_epoch']:
        epochs = ['initial']
    else:
        epochs = [None]


    for epi, epoch in enumerate(epochs):
        try:
            world, dataset, model_path = load_world(OPT['load_path'], epoch,
                                                   training=OPT['analyze_lang'],
                                                   init_ages=init_ages)
        except FileNotFoundError:
            continue
        if epi == 0:
            print('Questioner and Answerer Bots: ')
            print(world.qbots[-1])
            print(world.abots[-1])

        results = evaluate_world(world, dataset, report_script=True,
                                    speed=OPT['include_speed'],
                                    analyze_lang=OPT['analyze_lang'])
        results['model_path'] = model_path
        results_by_epoch[epoch] = results

    with open(save_path, 'w') as f:
        json.dump(results_by_epoch, f)


if __name__ == '__main__':
    parser = ParlaiParser()
    parser.add_argument('--load-path', type=str,
                        help='directory where pth files are stored (recursively)')
    parser.add_argument('--print-conv', default=False, action='store_true',
                        help='whether to print the conversation between bots or not')
    parser.add_argument('--include-speed', default=False, action='store_true',
                        help='set this to do a (slow) learning speed analysis')
    parser.add_argument('--many-epochs', default=False, action='store_true',
                        help='measure performance over many checkpoints to get a learning curve')
    parser.add_argument('--epoch', type=int, help='which checkpoint to evaluate')
    parser.add_argument('--generations', default=False, action='store_true', help='average over '
                        'many generations using oldest bot from those generations')
    parser.add_argument('--use-initial-epoch', default=False, action='store_true')
    parser.add_argument('--analyze-lang', default=False, action='store_true',
                        help='analyze languages with extra metrics from lang_analysis.py')
    parser.add_argument('--custom', default=False, action='store_true')

    OPT = parser.parse_args()
    main(OPT)
