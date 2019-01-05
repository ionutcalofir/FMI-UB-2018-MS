import argparse

from lda import LDA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train',
                        action='store_true')
    parser.add_argument('--predict',
                        action='store_true')
    parser.add_argument('--plot',
                        action='store_true')
    parser.add_argument('--similarity',
                        action='store_true')
    parser.add_argument('--predict_data_path',
                        default='data/predict/data.txt')
    parser.add_argument('--train_data_path_basic',
                        default='data/basic/train_data.txt')
    parser.add_argument('--train_data_path_correlated',
                        default='data/correlated/train_data_real.txt')
    parser.add_argument('--train_data_path_dynamic',
                        default='data/dynamic/train_data_real_year')
    parser.add_argument('--lda_type',
                        choices=['basic', 'correlated', 'dynamic'],
                        default='basic')
    parser.add_argument('--n_topics',
                        default=2,
                        type=int)
    parser.add_argument('--n_period',
                        default=2,
                        type=int,
                        help='Used only when lda_type = dynamic')

    args = parser.parse_args()

    if args.train:
        lda = LDA()
        if args.lda_type == 'basic':
            lda.fit(args.train_data_path_basic,
                    args.lda_type,
                    args.n_topics,
                    args.plot)
            lda.save_model()
        elif args.lda_type == 'correlated':
            lda.fit(args.train_data_path_correlated,
                    args.lda_type,
                    args.n_topics,
                    args.plot)
        elif args.lda_type == 'dynamic':
            lda.fit(args.train_data_path_dynamic,
                    args.lda_type,
                    args.n_topics,
                    args.plot,
                    args.n_period)
    elif args.similarity:
        lda = LDA()
        lda.load_model()
        lda.get_similarity(plot=args.plot)
    elif args.predict:
        lda = LDA()
        lda.load_model()
        lda.predict(data_path=args.predict_data_path, plot=args.plot)
