import argparse
from engine import Engine

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--phase', type=str,
                      help='Train a classifier. Options: [train, submission]',
                      default='train')
  parser.add_argument('--model_name', type=str,
      default='rnn')
  parser.add_argument('--logdir_path', type=str,
      default='./logdir')
  parser.add_argument('--exp_name', type=str,
      default='exp')
  parser.add_argument('--data_path', type=str,
      default='./data')
  parser.add_argument('--train_config_path', type=str,
      default='./data/configs/train/train.txt')
  parser.add_argument('--test_config_path', type=str,
      default='./data/configs/test/test.txt')
  parser.add_argument('--submission_test_config_path', type=str,
      default='./data/configs/submission/test.txt')
  parser.add_argument('--path_to_ckpt',
      default='./logdir/exp_2/models/model_20000.pt')
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--n_hidden', type=int, default=512)
  parser.add_argument('--n_iters', type=int, default=100000)
  parser.add_argument('--pretrained', type=int, default=1)

  return parser.parse_args()

def main(args):
  if args.phase == 'train':
    engine = Engine(args)
    engine.train(n_iters=args.n_iters)
  elif args.phase == 'submission':
    engine = Engine(args)
    engine.submission(path_to_ckpt=args.path_to_ckpt,
                      submission_test_config_path=args.submission_test_config_path)

if __name__ == '__main__':
  args = parse_args()
  main(args)
