import os
import os.path as pth

import evaluate
from world import QAWorld

import torch


def main():
    cpu = torch.device('cpu')
    other_model_dir = pth.join('data/experiments/', 'exp7.0.1.0.0.0', 'models/')
    other_world = evaluate.load_world(other_model_dir, map_location=cpu)
    
    for model in [0, 1, 2, 3]:
        for fold in [0, 1, 2, 3]:
            final_epochs = []
            for seed in [0, 1, 2, 3]:
                exp_code = 'exp7.{model}.0.0.{fold}.{seed}'.format(**locals())
                model_dir = pth.join('data/experiments/', exp_code, 'models/')
                final_epoch = evaluate.get_final_epoch(model_dir, map_location=cpu)
                final_epochs.append(final_epoch)

            last_epoch = min(max(final_epochs), min(final_epochs) + 200000)
            last_epoch = ((last_epoch // 1000) + 1) * 1000
            epochs = list(range(0, last_epoch, 1000))
            # comment this line out to generate all the epochs (for early stopping)
            epochs = [39000, epochs[-1]]
            for epoch in epochs:
                qbots = []
                abots = []
                sources = []
                for seed in [0, 1, 2, 3]:
                    exp_code = 'exp7.{model}.0.0.{fold}.{seed}'.format(**locals())
                    model_dir = pth.join('data/experiments/', exp_code, 'models/')
                    sources.append((model_dir, epoch))
                    world = evaluate.load_world(model_dir, epoch, map_location=cpu)[0]
                    qbots.append(world.qbots[0])
                    abots.append(world.abots[0])

                new_exp_code = 'exp9.{model}.0.0.{fold}.-1'.format(**locals())
                print(new_exp_code)
                new_model_dir = pth.join('data/experiments/', new_exp_code, 'models/')
                new_model_path = pth.join(new_model_dir, 'world_epoch_%s.pth' % str(epoch).zfill(5))
                os.makedirs(new_model_dir, exist_ok=True)
                opt = dict(world.opt)
                opt['num_qbots'] = len(qbots)
                opt['num_abots'] = len(abots)
                opt['save_path'] = new_model_dir
                opt['seed'] = -1
                opt['starttime'] = None
                opt['sources'] = sources
                opt['override']['num_qbots'] = len(qbots)
                opt['override']['num_abots'] = len(abots)
                opt['override']['save_path'] = new_model_dir
                opt['override']['seed'] = -1
                new_world = QAWorld(opt, qbots, abots)
                new_world.save_agents(new_model_path)

            best_model_path = pth.join(new_model_dir, 'best_world.pth')
            # NOTE: relies on the ordering of epochs
            new_world.save_agents(best_model_path, extra={'epoch': epoch}),





if __name__ == '__main__':
    main()

