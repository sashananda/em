import numpy as np
import math as m

scores = [[np.NAN, 63, 65, 70, 63], [53, 61, 72, 64, 73], [51, 67, 65, 65, np.NAN], [np.NAN, 69, 53, 53, 53], \
         [np.NAN, 69, 61, 55, 45], [np.NAN, 49, 62, 63, 62], [44, 61, 52, 62, np.NAN], [49, 41, 61, 49, np.NAN], \
         [30, 69, 50, 52, 45], [np.NAN, 59, 51, 45, 51], [np.NAN, 40, 56, 54, np.NAN], [42, 60, 54, 49, np.NAN], \
         [np.NAN, 63, 53, 54, np.NAN], [np.NAN, 55, 59, 53, np.NAN], [np.NAN, 49, 45, 48, np.NAN], [17, 53, 57, 43, 51], \
         [39, 46, 46, 32, np.NAN], [48, 38, 41, 44, 33], [46, 40, 47, 29, np.NAN], [30, 34, 43, 46, 18], \
         [np.NAN, 30, 32, 35, 21], [np.NAN, 26, 15, 20, np.NAN]]

data = np.array(scores)

class Params:
    """
    A class which represents the parameters in the algorithm.
        - mu : a k-dimension vector of means, where k is the number of non-missing values
        - lam : coefficient matrix, lambda (numpy matrix)
        - sigma : covariance matrix of W_t process
    """
    def __init__(self, p, data):
        self.mu = np.apply_along_axis(np.nanmean, 0, data)
        self.lam = np.ones(data.shape[1])
        self.sigma = np.identity(data.shape[1])
        self.p = p
        self.data = data
        self.k = data.shape[1]

class Model:
    """
    A class representing a full E-M model.
        - params : model parameters
    """

    def __init__(self, params):
        self.params = params
        self.log_p_ys = []
        self.H_ts = []
        self.y_ts = []
        self.X_t_hats = []
        self.X_t_X_tr_hats = []

    def re_init(self):
        self.H_ts = []
        self.y_ts = []
        self.X_t_hats = []
        self.X_t_X_tr_hats = []

    def compute_H_t(self, observation):
        H_t = np.identity(self.params.k)
        missing_rows = []
        for i in range(len(observation)):
            if (np.isnan(observation[i])):
                missing_rows.append(i)
        H_t = np.delete(H_t, missing_rows, 0)
        self.H_ts.append(H_t)
        return H_t

    def compute_R(self, observation):
        lam = np.array([self.params.lam])
        R = np.dot(lam.T, lam) + self.params.sigma
        self.R = R
        return R

    def compute_log_p_y_t(self, observation):
        H_t = self.compute_H_t(observation)
        k_t = H_t.shape[0]
        R = self.compute_R(observation)
        R_t = np.dot(H_t, np.dot(R, np.transpose(H_t)))
        mu_t = np.dot(H_t, self.params.mu)
        y_t = np.random.multivariate_normal(mu_t, R_t, 1)[0]
        self.y_ts.append(y_t)
        log_p_y_t = (-1/2) * (np.dot(np.transpose(y_t - mu_t), np.dot(np.linalg.inv(R_t), (y_t - mu_t))) + \
                              np.log(np.linalg.det(R_t)) + k_t * np.log(2 * m.pi))
        return log_p_y_t

    def compute_log_p_y(self):
        log_p_y = 0
        for i in range(self.params.data.shape[0]):
            log_p_y_t = self.compute_log_p_y_t(self.params.data[i])
            log_p_y += log_p_y_t
        self.log_p_ys.append(log_p_y)
        return log_p_y

    def expectation_step(self):
        """
        E step of the algorithm, computing the neccessary expected values:
            - E[X_t_tilde] (equation 6)
            - E[X_t_tilde.T]
            - Q_t_tilde (modified version of equation 7- see my supplementary derivation)
            - E[(X_t_tilde * X_t_tilde.T)]
        """
        for i in range(self.params.data.shape[0]):
            # Get E[X_t]
            H_t = self.H_ts[i]
            lambda_t = np.dot(H_t, self.params.lam)
            R_t = np.dot(H_t, np.dot(self.R, np.transpose(H_t)))
            mu_t = np.dot(H_t, self.params.mu)
            X_t_hat = np.dot(np.transpose(lambda_t), np.dot(np.linalg.inv(R_t), (self.y_ts[i] - mu_t)))
            X_t_hat = np.array([np.array([X_t_hat, 1])]).T
            self.X_t_hats.append(X_t_hat)

            # Get E[X_t.T]
            X_t_tr_hat = np.dot(np.transpose(self.y_ts[i] - mu_t), np.dot(np.transpose(np.linalg.inv(R_t)), lambda_t))
            X_t_tr_hat = np.array([np.array([X_t_tr_hat, 1])])

            # Get Q_t_tilde
            Q = np.identity(self.params.p) - np.dot(np.transpose(lambda_t), np.dot(np.linalg.inv(R_t), lambda_t))
            Q_tilde = np.zeros((self.params.p + 1, self.params.p + 1))
            Q_tilde[0, 0] = Q[0]

            # Get E[(X_t_tilde * X_t_tilde.T)]
            X_t_X_tr_hat = Q_tilde + np.dot(X_t_hat, X_t_tr_hat)
            self.X_t_X_tr_hats.append(X_t_X_tr_hat)

    def maximization_step(self):
        """
        M step of the algorithm, updating the trainable parameters.
        """
        self.update_Lambda()
        self.update_mu()
        self.update_lam()
        self.update_sigma()
        self.re_init()

    def get_Lambda_j(self, j):
        prod_1 = np.zeros(self.params.p + 1)
        prod_2 = np.zeros((self.params.p + 1, self.params.p + 1))
        for i in range(self.params.data.shape[0]):
            H_t = self.H_ts[i]
            z_t = np.dot(np.transpose(H_t), self.y_ts[i])
            z_t_j = z_t[j]
            prod = (z_t_j * self.X_t_hats[i]).T
            prod_1 = prod_1 + prod[0]

            # Get the rhs term in equation 13 - see supplementary working for the derivation
            indicator_j = 0 if np.isnan(self.params.data[i][j]) else 1
            X_t_X_tr_hat = indicator_j * self.X_t_X_tr_hats[i]
            prod_2 = prod_2 + X_t_X_tr_hat
        return np.dot(prod_1, np.linalg.inv(prod_2))

    def update_Lambda(self):
        Lambda = []
        for j in range(self.params.k):
            Lambda_j = self.get_Lambda_j(j)
            Lambda.append(Lambda_j)
        Lambda = np.array(Lambda)
        self.Lambda = Lambda

    def update_mu(self):
        self.params.mu = self.Lambda[:, 1]

    def update_lam(self):
        self.params.lam = self.Lambda[:, 0]

    def update_sigma(self):
        sum_lhs = np.zeros((self.params.k, self.params.k))
        sum_H_ts = np.zeros((self.params.k, self.params.k))
        for i in range(len(self.params.data)):
            # LHS of equation 16
            H_t = self.H_ts[i]
            z_t = np.array([np.dot(np.transpose(H_t), self.y_ts[i])])
            lam_x = self.Lambda @ self.X_t_hats[i]
            lam_x = lam_x.reshape(-1, 1)
            prod = np.dot(z_t.T, z_t) - np.dot(lam_x, z_t)
            sum_lhs = sum_lhs + prod

            # RHS of equation 16
            prod = np.dot(np.transpose(H_t), H_t)
            sum_H_ts = sum_H_ts + prod
        sum_lhs = np.diag(np.diag(sum_lhs))
        sum_H_ts = np.linalg.inv(sum_H_ts)
        sigma = np.dot(sum_lhs, sum_H_ts)
        self.params.sigma = sigma

p = 1
params = Params(p, data)

model = Model(params)

for i in range(10):
    model.compute_log_p_y()
    model.expectation_step()
    model.maximization_step()

print("Mu:")
print(model.params.mu)
print("Lambda: ")
print(model.params.lam)
print("Sigma")
print(model.params.sigma)
