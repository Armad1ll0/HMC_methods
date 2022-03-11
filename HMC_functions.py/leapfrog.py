#Basic leapfrog algorithm 
def leapfrog(step_size, trajectory_length, p, x, NLP_grad, inv_cov, M):
    p = p - 0.5*step_size*(M @ neg_log_prob_grad(x, A))
    in_between = []
    for i in range(int(trajectory_length) - 1):
        x = x + step_size*(p @ np.linalg.inv(M))
        in_between.append(x)
        p = p - 0.5*step_size*(M @ neg_log_prob_grad(x, A))
    x = x + step_size*(p @ np.linalg.inv(M))
    p = p - 0.5*step_size*(M @ neg_log_prob_grad(x, A))
    return x, p, in_between 
