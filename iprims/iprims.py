from __future__ import print_function

import sys
import argparse
import numpy.matlib
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from record_traj import RecordTrajectory


class InteractionPrimitive:
    """
    This is a (mostly complete) implementation of the Interaction Primitives
    framework described Dr. Heni Ben Amor and others at T.U. Darmstadt.

    See these papers for reference:

    Ewerton, M.; Neumann, G.; Lioutikov, R.; Ben Amor, H.; Peters, J.; Maeda, G. (2015).
        Learning Multiple Collaborative Tasks with a Mixture of Interaction Primitives,
        Proceedings of 2015 IEEE International Conference on Robotics and Automation (ICRA).

    Ben Amor, H.; Neumann, G.; Kamthe, S.; Kroemer, O.; Peters, J. (2014).
        Interaction Primitives for Human-Robot Cooperation Tasks ,
        Proceedings of 2014 IEEE International Conference on Robotics and Automation (ICRA).

    Maeda, G.J.; Ewerton, M.; Lioutikov, R.; Ben Amor, H.; Peters, J.; Neumann, G. (2014).
        Learning Interaction for Collaborative Tasks with Probabilistic Movement Primitives,
        Proceedings of the International Conference on Humanoid Robots (HUMANOIDS)

    Probabilistic movement primitives. A Paraschos, C Daniel, JR Peters, G Neumann.
        Advances in neural information processing systems, 2013.
    """

    def __init__(self,
                num_bases=20,
                num_dof_obs=2,
                num_dof_con=2,
                noise=.0001,
                timesteps=100):
        self.num_bases = num_bases
        self.num_dof_obs = num_dof_obs
        self.num_dof_con = num_dof_con
        self.noise = noise

        self.timesteps = timesteps
        self.trajectories = []
        self.psi_matrix = None
        #self.lambda_promp = 1e-5

        self.compute_basis_functions()



    ############################################################################
    # Learn weights from example
    ############################################################################

    def get_weights(self, traj, dof):
        """
        Compute the weights for a given trajectory
        """
        X = self.psi_matrix
        # TODO: replace y with phase based representation for this dof
        #y = traj[:,dof]
        y = self.get_phase_representation_for_traj(traj, dof)

        w = np.linalg.solve(
                np.dot(X, X.T) + np.eye(self.num_bases), # * self.lambda_promp,
                np.dot(X, y))
        return w.T


    def compute_interaction_promp_prior(self):
        '''
        Assumes n trajectories have already been recorded

        q = [qo.T, qc.T].T = recorded trajectories of human and robot
        w = [wo.T, wc.T].T = weight vectors of human and robot for one demo
        '''
        # compute normal dist, w = N(uw, Ew), from n demos
        # get normal distribution over weights from trajectories
        #weights = np.zeros((len(self.trajectories), 2 * self.num_bases * self.num_dof))
        weights = np.zeros((len(self.trajectories), self.num_bases * (self.num_dof_obs + self.num_dof_con) ))
        for i in range(len(self.trajectories)):
            traj = self.trajectories[i]
            w_obs, w_con = [], []

            for d in range(self.num_dof_obs):
                w_obs.append(self.get_weights(traj[0], d))

            for d in range(self.num_dof_con):
                w_con.append(self.get_weights(traj[1], d))

            w_obs = np.hstack(w_obs)
            w_con = np.hstack(w_con)
            weights[i,:] = np.hstack((w_obs, w_con))

        mean_weights = weights.mean(axis=0)
        self.Uw = mean_weights

        # Transpose the weights so that each row represents a rand var, i.e. w_i
        # this gives a cov matrix that is of shape (bfs, bfs) == (nweights, nweights)
        cov = np.cov(weights.T)
        self.Ew = cov


    def make_H(self, t):
        """
        Compute and return the matrix H for time t
        """
        #for now assume that both obs, cont agents have same dof
        H = np.zeros((self.num_dof_obs+self.num_dof_con, self.num_bases*(self.num_dof_obs+self.num_dof_con) ))
        for i in range(self.num_dof_obs):
            # for each dof of the observed agent
            # set the elements of the matrix
            for j in range(self.num_bases): #*self.num_dof_obs):
                #print i, j, t, self.num_bases*i+j, self.psi_matrix.shape
                H[i][self.num_bases*i+j] = self.psi_matrix[j][t]

        #print "H: ", H.shape
        return H.T


    def make_K(self, H, Ew_cur):
        """
        Compute and return the matrix K
        """
        # Generate noise matrix
        obs_noise = np.random.normal(0, self.noise, (H.T.shape[0], H.shape[1]))

        # Produce K by formula
        K = np.dot(np.dot(Ew_cur, H),
            np.linalg.pinv(obs_noise + np.dot(np.dot(H.T, Ew_cur), H)))

        #print "K: ", K.shape
        return K


    def get_dist_from_sample(self, D):
        """
        Get posterior distribution Uw, Ew given a partially observed trajectory

        Given partial trajectory, D, of human, where D = q_1:m for m < T
        Compute p(q_1:T|D) by formula (4) in `Learning Multiple Collaborative Tasks
        with a Mixture of Interaction Primitives`
        """
        Uw_cur, Ew_cur = self.Uw, self.Ew
        for t in range(D.shape[0]): # the number of rows, aka, steps
            # for now use only dof 0
            obs = np.concatenate( (D[t], np.zeros((self.num_dof_con,))) )

            H = self.make_H(t)
            K = self.make_K(H, Ew_cur)

            # print (obs.shape)
            # print (obs)
            # print (D[t])
            # print (np.dot(H.T, Uw_cur))
            # print (obs - np.dot(H.T, Uw_cur))
            # print

            # 1. Compute Uw_new = Uw + K(D - Ht.T * Uw)
            # 2. Compute Ew_new = Ew - K(Ht.T * Ew)
            Uw_cur = Uw_cur + np.dot(K, obs - np.dot(H.T, Uw_cur))
            Ew_cur = Ew_cur - np.dot(K, np.dot(H.T, Ew_cur))

        return Uw_cur, Ew_cur


    def get_weights_from_dist(self, u):
        """
        Take u = (bfs,) and given u[0:num_bases], u[num_bases:2*num_bases]
        """
        obs_w = u[0:self.num_bases*self.num_dof_obs]
        obs_w = obs_w.reshape(self.num_dof_obs, self.num_bases).T
        con_w = u[self.num_bases*self.num_dof_obs:self.num_bases*(self.num_dof_obs+self.num_dof_con)]
        con_w = con_w.reshape(self.num_dof_con, self.num_bases).T
        return obs_w, con_w


    def get_trajectory_from_weights(self, w):
        """
        Return the trajectory described by w
        """
        traj = np.dot(self.psi_matrix.T, np.squeeze(w))
        return traj
        # # psi_matrix is (num_bases, timesteps)
        # # w is (num_bases, dofs)
        # traj = np.empty((self.timesteps, w.shape[1]))
        # for dof in range(w.shape[1]):
        #     # for each dof in w
        #     # multiply row of w by bases to get trajectory
        #     wdof = w[:,dof]
        #     wmat = np.matlib.repmat(wdof, self.timesteps, 1).T
        #     prod = self.psi_matrix * wmat
        #     tdof = np.sum(prod, axis=0)
        #     traj[:,dof] = tdof
        # return traj



    ############################################################################
    # PSI matrix generation
    ############################################################################

    def compute_basis_functions(self, nder=1):
        """
        This function requires:
            num_bases - the number of basisfunctions
            timesteps - the number of timesteps
            num_dof - number of dofs
            nder - number of derivatives (default 1)
        """
        c, h = self.get_basis_func_params()
        z = np.linspace(0, 1, self.timesteps)

        # compute the basis function activations at each timestep
        b = np.exp(-1 * np.square(
                             (np.matlib.repmat(z, self.num_bases, 1) - np.matlib.repmat(c[np.newaxis].T, 1, self.timesteps))
                        ) / (2.0 * h))
        sum_b = np.matlib.repmat(np.sum(b, axis=0), self.num_bases, 1)
        b_norm = np.divide(b, sum_b)
        self.psi_matrix = b_norm


    def get_basis_func_params(self):
        c, h = None, None
        if (self.num_bases > 5):
            p = 1.0 / (self.num_bases - 3)
            a1 = np.array([0.0 - p])
            a2 = np.array([1.0 + p])
            mid = np.linspace(0.0, 1.0, self.num_bases - 2)
            c = np.concatenate((a1, mid, a2))
            h = (2 * 0.5 * (c[1] - c[0])) ** 2.0
        else:
            c = np.linspace(0, 1, self.num_bases)
            h = (2 * 0.5 * (c[1] - c[0])) ** 2.0 #1.0/self.num_bases
        return c, h



    ############################################################################
    # Generating trajectories
    ############################################################################

    def get_phase_representation_for_traj(self, traj, dof):
        """
        Computes a phase based representation for the given dof of this trajectory
        returns a numpy array of shape (timesteps,)
        """
        alpha = np.empty((self.timesteps,))
        tdof = traj[:,dof]
        tlen = len(tdof)
        for i in range(self.timesteps):
            alpha[i] = tdof[int(np.floor(float(i) / self.timesteps * tlen))]
        return alpha


    def set_training_trajectories(self, trajs):
        """
        trajs - must be a list of [human, robot] traj pairs where each traj is
                a numpy array of shape (timesteps, dof)
        """
        self.trajectories = trajs


    def set_controlled_training_trajectories(self, con_trajs, generate_random_obs=False):
        """
        con_trajs - must be a list of numpy arrays where each traj is
                a numpy array of shape (timesteps, dof)
        """
        # generate the matching number of random observed trajectories
        # zip them and store in self.trajectories
        obs_trajs = []
        for obs_t in range(len(con_trajs)):
            traj_obs = self.get_random_trajectory_obs()
            obs_trajs.append(traj_obs)
        self.trajectories = zip(obs_trajs, con_trajs)


    def set_observed_training_trajectories(self, obs_trajs, generate_random_con=False):
        """
        obs_trajs - must be a list of numpy arrays where each traj is
                a numpy array of shape (timesteps, dof)
        """
        pass


    def generate_training_trajectories(self, num_training_traj=20, random=False):
        min_len_traj = np.inf
        if (random):
            for t in range(num_training_traj):
                traj_obs = self.get_random_trajectory_obs()
                traj_con = self.get_random_trajectory_con()
                self.trajectories.append([traj_obs, traj_con])
            min_len_traj = self.timesteps
        else:
            rec = RecordTrajectory(num_trajs=2)
            for t in range(num_training_traj):
                trajs = rec.record_trajectory()
                traj_obs = np.array(trajs[0])
                traj_con = np.array(trajs[1])
                if len(traj_obs) < min_len_traj:
                    min_len_traj = len(traj_obs)
                if len(traj_con) < min_len_traj:
                    min_len_traj = len(traj_con)
                self.trajectories.append([traj_obs,traj_con])
        self.truncate_trajectories(min_len_traj)
        self.interpolate_trajectories(min_len_traj)


    def get_random_trajectory_obs(self, amp=10.0, freq=2*np.pi*.02):
        x = np.arange(0, self.timesteps)
        ys = []
        for d in range(self.num_dof_obs):
             y = 1.0/10000 * (x - 50.0)**3 + 12.5
             ys.append(y)

        return np.stack((ys)).T


    def get_random_trajectory_con(self, amp=10.0, freq=2*np.pi*.02):
        x = np.arange(0, self.timesteps)
        ys = []
        for d in range(self.num_dof_con):
            y = -1.0/10000 * (x - 50.0)**3 + 12.5
            ys.append(y)

        return np.stack((ys)).T


    def get_partial_sample(self, random=False, length=25):
        if (random):
            sample = self.get_random_trajectory_obs()[0:length]
            return sample
        else:
            rec = RecordTrajectory(num_trajs=1)
            sample = np.array(rec.record_trajectory()[0])
            sample = sample[0:self.timesteps]
            return sample


    def truncate_trajectories(self, traj_len):
        """
        Make sure each trajectory has the same length. Truncate to tlen.
        """
        for i in range(len(self.trajectories)):
            self.trajectories[i][0] = self.trajectories[i][0][0:traj_len]
            self.trajectories[i][1] = self.trajectories[i][1][0:traj_len]


    def interpolate_trajectories(self, min_len_traj):
        """
        Interpolate trajectories so that each is 100 timesteps
        """
        # min_len_traj numbers spaced from 0 to 100
        x = np.linspace(0, self.timesteps, min_len_traj)
        for t in range(len(self.trajectories)):
            for i in range(2): # for obs and con
                num_dof = self.num_dof_obs if (i == 0) else self.num_dof_con
                new_traj = np.empty((self.timesteps, num_dof))
                for d in range(num_dof):
                    traj = self.trajectories[t][i] # get the t-th trajectory and select observed or controlled
                    # select correct dof
                    f_path_interpolator = scipy.interpolate.interp1d(x, traj[:,d])
                    # interpolate new traj
                    for j in range(self.timesteps):
                        new_traj[j][d] = f_path_interpolator(j)
                self.trajectories[t][i] = new_traj



    ############################################################################
    # Plotting
    ############################################################################

    def plot_all_trajectories(self):
        for t in range(len(self.trajectories)):
            obs, con = self.trajectories[t]
            for d in range(self.num_dof_obs):
                plt.subplot(len(self.trajectories), self.num_dof_obs, t*self.num_dof_obs+d+1)
                y_obs = obs[:,d]
                plt.plot(y_obs, lw=1)
        plt.show()

        for t in range(len(self.trajectories)):
            obs, con = self.trajectories[t]
            for d in range(self.num_dof_con):
                plt.subplot(len(self.trajectories), self.num_dof_con, t*self.num_dof_con+d+1)
                y_con = con[:,d]
                plt.plot(y_con, lw=1)
        plt.show()


    def plot_trajectory(self, traj, dof=0):
        """
        traj is a pair [obs, con] to be plotted
        """
        obs, con = traj

        for d in range(self.num_dof_obs):
            plt.subplot(1, self.num_dof_obs, d+1)
            y_obs = obs[:,d]

            plt.title('Observed Trajectory, DOF='+str(d))
            plt.xlabel('timesteps')
            plt.ylabel('position')
            plt.plot(y_obs, lw=1)
        plt.show()

        for d in range(self.num_dof_con):
            plt.subplot(1, self.num_dof_con, d+1)
            y_con = con[:,d]

            plt.title('Controlled Trajectory, DOF='+str(d))
            plt.xlabel('timesteps')
            plt.ylabel('position')
            plt.plot(y_con, lw=1)
        plt.show()


    def plot_basis_functions(self):
        for b in range(self.num_bases):
            plt.plot(self.psi_matrix[b,:])
        plt.title('Basis functions')
        plt.xlabel('timesteps')
        plt.ylabel('activations')
        plt.show()


    def plot_trajectory_approximation(self, w, title=None):
        """
        Plot trajectory that is represented by the bases and the weights
        """
        traj = self.get_trajectory_from_weights(w)
        plt.plot(traj)
        if (title is not None):
            plt.title(title)
        else:
            plt.title('Probabilistic Trajectory')
        plt.xlabel('timesteps')
        plt.ylabel('position')
        plt.show()


    def plot_trajectory_approximation_given_sample(self, w, sample, obs=True, title=None):
        sample = np.squeeze(sample)
        traj = self.get_trajectory_from_weights(w)

        if (obs):
            plt.plot(sample, lw=2)
            traj[0:len(sample)] = sample

        plt.plot(traj)
        if (title is not None):
            plt.title(title)
        else:
            plt.title('Probabilistic Trajectory')
        plt.xlabel('timesteps')
        plt.ylabel('position')
        plt.show()



if __name__ == "__main__":
    plot = True
    rnd = True

    ip = InteractionPrimitive(num_bases=20, num_dof_obs=5, num_dof_con=2)
    ip.generate_training_trajectories(random=rnd)
    ip.compute_interaction_promp_prior()
    obs_prior, con_prior = ip.get_weights_from_dist(ip.Uw)

    if (plot):
        ip.plot_all_trajectories()
        ip.plot_basis_functions()

        ip.plot_trajectory_approximation(obs_prior, title='Trajectory approx. for observed prior')
        ip.plot_trajectory_approximation(con_prior, title='Trajectory approx. for controlled prior')

    sample = ip.get_partial_sample(random=rnd, length=25)
    Usample, Esample = ip.get_dist_from_sample(sample)
    obs_post, con_post = ip.get_weights_from_dist(Usample)

    if (plot):
        ip.plot_trajectory_approximation(obs_post, title='Trajectory approx. for observed posterior')
        ip.plot_trajectory_approximation(con_post, title='Trajectory approx. for controlled posterior')

        ip.plot_trajectory_approximation_given_sample(obs_post, sample, obs=True, title='Trajectory approx. for observed posterior given sample')
        ip.plot_trajectory_approximation_given_sample(con_post, sample, obs=False, title='Trajectory approx. for controlled posterior given sample')
