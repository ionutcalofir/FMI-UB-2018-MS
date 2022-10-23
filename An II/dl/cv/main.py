import argparse
from engine import Engine

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--phase', type=str,
                      help='Train a classifier. Options: [train, submission]',
                      default='train')
  parser.add_argument('--data_path', type=str,
      default='./data/images')
  parser.add_argument('--train_config_path', type=str,
      default='./data/configs/train/train.txt')
  parser.add_argument('--test_config_path', type=str,
      default='./data/configs/test/test.txt')
  parser.add_argument('--submission_test_config_path', type=str,
      default='./data/configs/submission_test/submission_test.txt')
  parser.add_argument('--path_to_ckpt', type=str,
      default='./logdir/exp_1/models/model_9.pt')
  parser.add_argument('--logdir_path', type=str,
      default='./logdir')
  parser.add_argument('--exp_name', type=str,
      default='exp')
  parser.add_argument('--train_transforms', default=[], nargs='+')
  parser.add_argument('--test_transforms', default=[], nargs='+')
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--class_weights', default=['0.58', '3.38'], nargs='+')
  parser.add_argument('--optimizer', default='sgd')

  return parser.parse_args()

def main(args):
  if args.phase == 'train':
    engine = Engine(args)
    engine.train(data_path=args.data_path,
                 train_transforms=args.train_transforms,
                 test_transforms=args.test_transforms,
                 train_config_path=args.train_config_path,
                 test_config_path=args.test_config_path,
                 batch_size=args.batch_size,
                 epochs=args.epochs)
  elif args.phase == 'submission':
    engine = Engine(args)
    engine.submission(path_to_ckpt=args.path_to_ckpt,
                      test_config_path=args.test_config_path,
                      data_path=args.data_path,
                      test_transforms=args.test_transforms,
                      submission_test_config_path=args.submission_test_config_path,
                      batch_size=args.batch_size)

if __name__ == '__main__':
  args = parse_args()
  main(args)
