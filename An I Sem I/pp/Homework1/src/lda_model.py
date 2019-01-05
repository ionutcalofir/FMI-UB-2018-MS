import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

from text_processing import TextProcessing

class LDAModel():
    def __init__(self,
                 topics=2,
                 lda_type='basic',
                 n_period=2):
        self._topics = topics
        self._tp = TextProcessing()
        self._lda_type = lda_type
        self._n_period = n_period

    def plot(self, name, phi, theta, corpus_text):
        reverse_dw = {v: k for k, v in self._tp._dw.items()}
        topic_words = [[(prob, reverse_dw[i]) for i, prob in enumerate(phi[t])]
                            for t in range(self._topics)]
        for t in range(self._topics):
            topic_words[t].sort(key=lambda k : k[0], reverse=True)

        tp = [[t[0] for t in topic_words[t]] for t in range(self._topics)]
        ws = [[t[1] for t in topic_words[t]] for t in range(self._topics)]

        n = 10
        for t in range(self._topics):
            fig = plt.figure('Topic {0} - Model Type {1}'.format(t, name))
            ax = fig.subplots()

            ax.bar(np.arange(len(tp[t][:n])), tp[t][:n])
            ax.set_xticks(np.arange(len(ws[t][:n])))
            ax.set_xticklabels(ws[t][:n])

        topics_name = ['Topic{0}'.format(i) for i in range(self._topics)]
        for d in range(len(corpus_text)):
            fig = plt.figure('Document {0} - Model Type: {1}'.format(d, name))
            ax = fig.subplots()

            ax.bar(np.arange(len(theta[d])), theta[d])
            ax.set_xticks(np.arange(len(topics_name)))
            ax.set_xticklabels(topics_name)

        plt.show()

    def fit(self, data_path, plot):
        if self._lda_type == 'basic' or self._lda_type == 'correlated':
            text = self._get_text(data_path)
            text = self._tp.analyze(text)
        elif self._lda_type == 'dynamic':
            text = []
            for i in range(self._n_period):
                t = self._get_text(data_path + '_' + str(i) + '.txt')
                t = self._tp.analyze(t)

                text.append(t)

        self.corpus = text

        if self._lda_type == 'basic':
            self.theta, self.phi, self.z = self._mcmc_lda(text)
        elif self._lda_type == 'correlated':
            self.theta, self.phi, self.z = self._mcmc_lda_correlated(text)
        elif self._lda_type == 'dynamic':
            self.theta, self.phi, self.z = self._mcmc_lda_dynamic(text)

        if plot:
            if self._lda_type == 'dynamic':
                for y in range(self._n_period):
                    self.plot(self._lda_type + ', time: {0}'.format(y), self.phi[y], self.theta[y], self.corpus[y])
            else:
                self.plot(self._lda_type, self.phi, self.theta, self.corpus)

        with open('theta.txt', 'w') as f:
            f.write(str(self.theta))
        with open('phi.txt', 'w') as f:
            f.write(str(self.phi))
        with open('z.txt', 'w') as f:
            f.write(str(self.z))

    def _mcmc_lda(self, text):
        beta = self._tp.vocab_size() * [1]
        alfa = self._topics * [1]

        # we need to use pm.Container in order to use a list of variables
        # CompletedDirichlet because dirichlet only shows k-1 probabilities
        # phi - for each topic we draw its word distribution
        phi = pm.Container([pm.CompletedDirichlet('cphi_{0}'.format(k),
                                                  pm.Dirichlet('phi_{0}'.format(k), theta=beta)) for k in range(self._topics)])

        # theta - for each document we draw its topic distribution
        theta = pm.Container([pm.CompletedDirichlet('ctheta_{0}'.format(d),
                                                    pm.Dirichlet('theta_{0}'.format(d), theta=alfa)) for d in range(len(text))])

        # categorical = multinomial with 1 try
        # z - assign every word position a topic
        # if I use p=theta[m].value, then the theta values don't change when I call trace on theta so I used p=theta[m] which works
        z = pm.Container([[pm.Categorical('z_{0}_{1}'.format(m, n), p=theta[m])
                            for n in range(len(text[m]))]
                                for m in range(len(text))])

        # w - draw a word from the specified topic
        w = pm.Container([[pm.Categorical('w_{0}_{1}'.format(m, n),
                            # p=phi[z[m][n].value], # same reason as above, avoid using V.value
                            p=pm.Lambda('pw_{0}_{1}'.format(m, n), lambda zz=z[m][n], pp=phi : pp[zz]), # get z[m][n] value then get words distribution for that topic
                            value=text[m][n],
                            observed=True)
                                for n in range(len(text[m]))]
                                    for m in range(len(text))])

        model = pm.Model([phi, theta, z, w])

        mcmc = pm.MCMC(model)
        mcmc.sample(40000, 10000, 1)

        theta_result = []
        for i in range(len(text)):
            tt = mcmc.trace('ctheta_{0}'.format(i))[2999]
            theta_result.append(tt.flatten().tolist())
        theta_result = np.array(theta_result)

        phi_result = []
        for i in range(self._topics):
            pp = mcmc.trace('cphi_{0}'.format(i))[2999]
            phi_result.append(pp.flatten().tolist())
        phi_result = np.array(phi_result)

        z_result = []
        for m in range(len(text)):
            zz = []
            for n in range(len(text[m])):
                val = mcmc.trace('z_{0}_{1}'.format(m, n))[2999]
                zz.append(val)
            z_result.append(np.array(zz))
        z_result = np.array(z_result)

        return theta_result, phi_result, z_result

    def _mcmc_lda_correlated(self, text):
        beta = self._tp.vocab_size() * [1]
        miu_lower = -0.01
        miu_upper = 0.01

        phi = pm.Container([pm.CompletedDirichlet('cphi_{0}'.format(k),
                                                  pm.Dirichlet('phi_{0}'.format(k), theta=beta)) for k in range(self._topics)])

        miu = pm.Container([[pm.Uniform('miu_{0}_{1}'.format(d, k), lower=miu_lower, upper=miu_upper)
                                for k in range(self._topics)]
                                    for d in range(len(text))])
        tau = pm.Container([pm.Wishart('tau_{0}'.format(d), n=self._topics + 1, Tau=np.eye(self._topics))
                                for d in range(len(text))])

        eta = pm.Container([pm.MvNormal('eta_{0}'.format(d), mu=miu[d], tau=tau[d])
                                for d in range(len(text))])

        z = pm.Container([[pm.Categorical('z_{0}_{1}'.format(m, n),
                                          p=pm.Lambda('pz_{0}_{1}'.format(m, n), lambda e=eta[m] : np.exp(e) / np.sum(np.exp(e))))
                            for n in range(len(text[m]))]
                                for m in range(len(text))])

        w = pm.Container([[pm.Categorical('w_{0}_{1}'.format(m, n),
                            p=pm.Lambda('pw_{0}_{1}'.format(m, n), lambda zz=z[m][n], pp=phi : pp[zz]),
                            value=text[m][n],
                            observed=True)
                                for n in range(len(text[m]))]
                                    for m in range(len(text))])

        model = pm.Model([phi, miu, tau, eta, z, w])

        mcmc = pm.MCMC(model)
        mcmc.sample(20000, 5000, 1)

        theta_result = []
        for i in range(len(text)):
            tt = mcmc.trace('eta_{0}'.format(i))[2999]
            theta_result.append((np.exp(tt) / np.sum(np.exp(tt))).flatten().tolist())
        theta_result = np.array(theta_result)

        phi_result = []
        for i in range(self._topics):
            pp = mcmc.trace('cphi_{0}'.format(i))[2999]
            phi_result.append(pp.flatten().tolist())
        phi_result = np.array(phi_result)

        z_result = []
        for m in range(len(text)):
            zz = []
            for n in range(len(text[m])):
                val = mcmc.trace('z_{0}_{1}'.format(m, n))[2999]
                zz.append(val)
            z_result.append(np.array(zz))
        z_result = np.array(z_result)

        return theta_result, phi_result, z_result

    def _mcmc_lda_dynamic(self, text):
        alfa = self._topics * [1]
        pi = self._n_period * [0]
        theta = self._n_period * [0]
        z = self._n_period * [0]
        w = self._n_period * [0]
        miu_lower = -0.01
        miu_upper = 0.01
        tau = 1

        miu = pm.Container([[pm.Uniform('miu_{0}_{1}'.format(k, v), lower=miu_lower, upper=miu_upper)
                                for v in range(self._tp.vocab_size())]
                                    for k in range(self._topics)])

        pi[0] = [pm.MvNormal('pi_0_{0}'.format(k), mu=miu[k], tau=tau * np.eye(self._tp.vocab_size()))
                    for k in range(self._topics)]

        theta[0] = [pm.CompletedDirichlet('ctheta_0_{0}'.format(d),
                        pm.Dirichlet('theta_0_{0}'.format(d), theta=alfa)) for d in range(len(text[0]))]

        z[0] = [[pm.Categorical('z_0_{0}_{1}'.format(m, n), p=theta[0][m])
                            for n in range(len(text[0][m]))]
                                for m in range(len(text[0]))]

        w[0] = pm.Container([[pm.Categorical('w_0_{0}_{1}'.format(m, n),
                    p=pm.Lambda('pw_0_{0}_{1}'.format(m, n), lambda zz=z[0][m][n], pp=pi[0] : np.exp(pp[zz]) / np.sum(np.exp(pp[zz]))),
                            value=text[0][m][n],
                            observed=True)
                                for n in range(len(text[0][m]))]
                                    for m in range(len(text[0]))])

        for t in range(1, self._n_period):
            pi[t] = [pm.MvNormal('pi_{0}_{1}'.format(t, k), mu=pi[t - 1][k], tau=tau * np.eye(self._tp.vocab_size()))
                            for k in range(self._topics)]

            theta[t] = [pm.CompletedDirichlet('ctheta_{0}_{1}'.format(t, d),
                        pm.Dirichlet('theta_{0}_{1}'.format(t, d), theta=alfa)) for d in range(len(text[t]))]

            z[t] = [[pm.Categorical('z_{0}_{1}_{2}'.format(t, m, n), p=theta[t][m])
                            for n in range(len(text[t][m]))]
                                for m in range(len(text[t]))]

            w[t] = pm.Container([[pm.Categorical('w_{0}_{1}_{2}'.format(t, m, n),
                p=pm.Lambda('pw_{0}_{1}_{2}'.format(t, m, n), lambda zz=z[t][m][n], pp=pi[t] : np.exp(pp[zz]) / np.sum(np.exp(pp[zz]))),
                            value=text[t][m][n],
                            observed=True)
                                for n in range(len(text[t][m]))]
                                    for m in range(len(text[t]))])

        pi_c = pm.Container(pi)
        theta_c = pm.Container(theta)
        z_c = pm.Container(z)
        w_c = pm.Container(w)

        model = pm.Model([miu, pi_c, theta_c, z_c, w_c])

        mcmc = pm.MCMC(model)
        mcmc.sample(10000, 1000, 1)

        theta_result = []
        for t in range(self._n_period):
            t_pos = []
            for i in range(len(text[t])):
                tt = mcmc.trace('ctheta_{0}_{1}'.format(t, i))[2999]
                t_pos.append(tt.flatten().tolist())
            theta_result.append(t_pos)
        theta_result = np.array(theta_result)

        phi_result = []
        for t in range(self._n_period):
            t_pos = []
            for i in range(self._topics):
                pp = mcmc.trace('pi_{0}_{1}'.format(t, i))[2999]
                t_pos.append((np.exp(pp) / np.sum(np.exp(pp))).flatten().tolist())
            phi_result.append(t_pos)
        phi_result = np.array(phi_result)

        z_result = []
        for t in range(self._n_period):
            t_pos = []
            for m in range(len(text[t])):
                zz = []
                for n in range(len(text[t][m])):
                    val = mcmc.trace('z_{0}_{1}_{2}'.format(t, m, n))[2999]
                    zz.append(val)
                t_pos.append(np.array(zz))
            z_result.append(t_pos)
        z_result = np.array(z_result)

        return theta_result, phi_result, z_result

    def _get_text(self, data_path):
        text = []
        with open(data_path, 'r') as f:
            for line in f:
                text.append(line)

        return text

    def get_similarity(self, plot=False, verbose=True):
        sim = []
        for d in self.theta:
            sim_curr = []
            for d_curr in self.theta:
                val = np.sum(np.power(np.sqrt(d) - np.sqrt(d_curr), 2)) # compute Hellinger distance on posterior probabilities
                sim_curr.append(val)
            sim.append(sim_curr)
        sim = np.array(sim)

        if verbose:
            print(sim)

        return sim

    def predict(self, data_path, plot=False, verbose=True):
        text = self._get_text(data_path)
        text = self._tp.analyze(text)

        total_words = 0
        for k, v in self._tp._cw.items():
            total_words += v

        topics = self._topics * [0]
        for d in self.z:
            for word in d:
                topics[word] += 1
        topics = np.array(topics)

        t_result = [] # assign to every word from documents a topic
        for doc in text:
            t_doc = []
            for word in doc:
                p_t_w = [] # compute P(topic | word)
                for t in range(self._topics):
                    p_d_t = self.phi[t][word] # compute P(word | topic)
                    p_t = topics[t] / np.sum(topics) # compute P(topic) in the whole corpus
                    p_w =  self._tp._cw[word] / total_words # compute P(word) in the whole corpus

                    p_t_w.append((p_d_t * p_t) / p_w)

                p_t_w = np.array(p_t_w)
                p_t_w_prob = np.exp(p_t_w) / np.sum(np.exp(p_t_w)) # convert array to array of probabilities
                top = np.random.multinomial(1, p_t_w_prob).argmax() # draw a random number from the above probabilities
                t_doc.append(top)
            t_result.append(t_doc)

        t_result = np.array(t_result)

        if verbose:
            print(t_result)

        return t_result
