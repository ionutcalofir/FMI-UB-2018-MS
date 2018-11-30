import argparse
import numpy as np
import pymc as pm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--constant', action='store_true')
    parser.add_argument('--uniform', action='store_true')

    args = parser.parse_args()

    if args.constant:
        c = 5
        X = pm.DiscreteUniform('X', lower=c, upper=c)

        model = pm.Model([X])

        mcmc = pm.MCMC(model)
        mcmc.sample(40000, 10000, 1)
        X_samples = mcmc.trace('X')[:]

        eta = X_samples / (X_samples + c)
        err = np.array([min(el, 1 - el) for el in eta])
        print('Eroarea L* = {0}'.format(err.mean()))
    elif args.uniform:
        c = 5
        X = pm.Uniform('X', lower=0, upper=4 * c)

        model = pm.Model([X])

        mcmc = pm.MCMC(model)
        mcmc.sample(40000, 10000, 1)
        X_samples = mcmc.trace('X')[:]

        eta = X_samples / (X_samples + c)
        err = np.array([min(el, 1 - el) for el in eta])
        print('Eroarea L* = {0}'.format(err.mean()))
