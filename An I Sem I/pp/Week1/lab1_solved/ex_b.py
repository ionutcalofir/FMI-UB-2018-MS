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

        # not sure if this is correct because Y is pm.deterministic, but inside the
        # the function I call pm.Bernoulli
        @pm.deterministic
        def Y(X=X):
            prob = X / (X + c)
            bl = pm.Bernoulli('bl', p=prob)
            return bl.value

        model = pm.Model([X, Y])

        mcmc = pm.MCMC(model)
        mcmc.sample(40000, 10000, 1)
        X_samples = mcmc.trace('X')[:]
        Y_samples = mcmc.trace('Y')[:]

        g = np.array([1 if el > c else 0 for el in X_samples])
        err = np.array([1 if g[i] == Y_samples[i] else 0 for i in range(g.shape[0])])

        print('Eroarea L* = {0}'.format((1 - err).mean()))
    elif args.uniform:
        c = 5
        X = pm.Uniform('X', lower=0, upper=4 * c)

        # not sure if this is correct because Y is pm.deterministic, but inside the
        # the function I call pm.Bernoulli
        @pm.deterministic
        def Y(X=X):
            prob = X / (X + c)
            bl = pm.Bernoulli('bl', p=prob)
            return bl.value

        model = pm.Model([X, Y])

        mcmc = pm.MCMC(model)
        mcmc.sample(40000, 10000, 1)
        X_samples = mcmc.trace('X')[:]
        Y_samples = mcmc.trace('Y')[:]

        g = np.array([1 if el > c else 0 for el in X_samples])
        err = np.array([1 if g[i] == Y_samples[i] else 0 for i in range(g.shape[0])])

        print('Eroarea L* = {0}'.format((1 - err).mean()))
