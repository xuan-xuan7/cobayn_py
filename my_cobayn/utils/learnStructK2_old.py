# import numpy as np
# from pgmpy.estimators import K2Score
# from pgmpy.models import BayesianNetwork

# def learn_struct_K2(data_T, ns, order):

#     data = data_T.transpose()
#     n, ncases = data.shape
#     clamped = np.zeros((n, ncases))

#     max_fan_in = n
#     verbose = 0

#     dag = np.zeros((n, n))

#     for i in range(n):
#         ps = []
#         j = order[i]
#         u = np.where(clamped[j] == 0)[0]
#         score = score_family(j, ps, data.iloc[:, u], ns)
#         if verbose:
#             print(f"\nnode {j}, empty score {score:.4f}")
#         done = False
#         while not done and len(ps) <= max_fan_in:
#             pps = np.setdiff1d(order[:i], ps)  # potential parents
#             nps = len(pps)
#             pscore = np.zeros(nps)
#             for pi in range(nps):
#                 p = pps[pi]
#                 pscore[pi] = score_family(j, ps + [p], data.iloc[:, u], ns)
#                 if verbose:
#                     print(f"considering adding {p} to {j}, score {pscore[pi]:.4f}")
#             if nps > 0:
#                 best_pscore, best_p = np.max(pscore), pps[np.argmax(pscore)]
#                 if best_pscore > score:
#                     score = best_pscore
#                     ps.append(best_p)
#                     if verbose:
#                         print(f"* adding {best_p} to {j}, score {best_pscore:.4f}")
#                 else:
#                     done = True
#             else:
#                 done = True
#         if ps:
#             dag[np.ix_(ps, [j])] = 1

#     return dag

# def score_family(j, ps, data, ns):
    
#     model = BayesianNetwork()
#     model.add_edges_from([(p, j) for p in ps])
#     k2score = K2Score(data=data, state_names=ns, complete_samples_only=True)
#     score = k2score.score(model)

#     return score
