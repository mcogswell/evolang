from parlai.core.params import ParlaiParser


def read():
    parser = ParlaiParser()

    parser.add_argument_group('Model Parameters')
    parser.add_argument('--q-out-vocab', default=3, type=int,
                        help='Output vocabulary for questioner')
    parser.add_argument('--a-out-vocab', default=4, type=int,
                        help='Output vocabulary for answerer')
    parser.add_argument('--img-feat-size', default=20, type=int,
                        help='Image feature size for each attribute')
    parser.add_argument('--embed-size', default=20, type=int,
                        help='Embed size for words')
    parser.add_argument('--hidden-size', default=100, type=int,
                        help='Hidden Size for the language models')
    parser.add_argument('--symmetric', default=0, type=int,
                    help='0 - abots and qbots have different architectures\n'
                         '1 - create max(num_abots, num_qbots) bots, all with\n'
                         '    the same A/Q agnostic architecture')
    parser.add_argument('--init', default='normal',
                        choices=['normal', 'sorted'])
    parser.add_argument('--critic', default='none',
                        choices=['none', 'value'])

    parser.add_argument_group('Evolution Parameters')
    parser.add_argument('--num-abots', default=1, type=int,
                        help='Number of abots to use')
    parser.add_argument('--num-qbots', default=1, type=int,
                        help='Number of qbots to use')
    parser.add_argument('--kill-epoch', default=0, type=int,
                        help='Kill models ever x epochs (or not at all if 0)')
    parser.add_argument('--kill-policy', default='random',
        help='How to decide which bots to kill.\n'
             'random - uniform random\n'
         'simple_meritocratic - sample from softmax parameterized by val acc\n'
         'eps_greedy<eps> - uniform random with probability eps;\n'
         '                  otherwise choose bot with worst val performance')
    parser.add_argument('--kill-type-policy', default='both',
        help='How to decide which type(s) of agents to kill\n'
            'both - always kill 1 qbot and 1 abot\n'
            'alternate - alternate between killing an abot and a qbot\n'
            'random - random uniform choice of abot or qbot')

    parser.add_argument_group('Loss Parameters')
    parser.add_argument('--task-reward-freq', default=1, type=int,
                        help='include task reward in the loss every _ epochs')
    parser.add_argument('--teaching-freq', default=0, type=int,
                        help='include teaching error in loss every _ epochs')
    parser.add_argument('--teacher-policy', default='simple_meritocratic',
                help='How to decide which bot teaches others.\n'
         'simple_meritocratic - sample from softmax parameterized by val acc')
    parser.add_argument('--reward', default='both:1', type=str,
                        help='separate kinds of rewards by _, '
                'as in "both:1.5_vocab:2"\n'
                'both:<w> - reward weight <w> for getting both correct\n'
                'one:<w> - reward weight <w> for getting one correct\n'
                '         (if both are correct this is added twice, so it\n'
                '         can replace both with weight 2*<w>)\n'
                'vocab:<w> - reward weight <w> for the r_c vocab penalty from\n'
                '           Mordatch and Abbeel')

    parser.add_argument_group('Dialog Episode Parameters')
    parser.add_argument('--rl-scale', default=100.0, type=float,
                        help='Weight given to rl gradients')
    parser.add_argument('--num-rounds', default=2, type=int,
                        help='Number of rounds between Q and A')
    parser.add_argument('--memoryless-abot', dest='memoryless_abot', action='store_true',
                        help='Turn on/off for ABot with memory')

    parser.add_argument_group('Dataset Parameters')
    parser.add_argument('--data-path', default='data/synthetic_dataset.json', type=str,
                        help='Path to the training/val dataset file')
    parser.add_argument('--neg-fraction', default=0.8, type=float,
                        help='Fraction of negative examples in batch')

    parser.add_argument_group('Optimization Hyperparameters')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Batch size during training')
    parser.add_argument('--num-epochs', default=1000000, type=int,
                        help='Max number of epochs to run')
    parser.add_argument('--learning-rate', default=1e-2, type=float,
                        help='Initial learning rate')
    parser.add_argument('--save-epoch', default=100, type=int,
                        help='Save model at regular intervals of epochs.')
    parser.add_argument('--save-path', default='checkpoints', type=str,
                        help='Directory path to save checkpoints.')
    parser.add_argument('--use-gpu', dest='use_gpu', default=False, action='store_true')

    parser.add_argument('--seed', default=1338)

    return parser.parse_args()
