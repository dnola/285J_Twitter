
def jaccard_index(a, b): # Where a, b are lists of top words
    return len(set(a) & set(b)) / float(len(set(a) | set(b)))

def average_jaccard(A, B): # Where a, b are lists of top words
    score = 0
    for i in range(len(A)):
        Ri = A[:i+1]
        Rj = B[:i+1]
        score += jaccard_index(Ri, Rj)
    final_score = score/float(len(A))
    return final_score

def compute_agreements(similarity):
    import munkres
    import numpy as np

    m = munkres.Munkres()
    print("Computing mapping...")
    similarity = munkres.make_cost_matrix(similarity, lambda cost: 1 - cost)
    indexes = m.compute(similarity)

    agreement = np.sum([1 - similarity[r][c] for r, c in indexes]) / len(similarity)
    print("Agreement:", agreement)
    return agreement

def main():
    stability_pairs = []
    for pkl in glob(join('Stability Results','*.pkl')):
        assignments = pickle.load(open(pkl,'rb'))
        print(pkl)

        similarities = []
        for pair in itertools.combinations(assignments,2):
            sx = np.array(pair[0][1])
            sy = np.array(pair[1][1])
            sx = np.array([[hash(y) for y in x] for x in sx]).astype(np.uint16) # Pairwise distances is a scikit function, and scikit hates strings
            sy = np.array([[hash(y) for y in x] for x in sy]).astype(np.uint16) # also hash() gives ridiculous numbers so we constrain them a bit to uint16s

            similarity = pairwise_distances(sx,sy,metric=average_jaccard,n_jobs=-1)
            similarities.append(similarity)
        print("Done computing distances...")

        agreements = Parallel(n_jobs=-1)(delayed(compute_agreements)(s) for s in similarities) # Hungarian method takes forever and is single threaded so we do all of them at once
        agreements = np.array(agreements).astype(np.float64)
        print(agreements)

        stability = np.mean(agreements)
        print("Stability for",pkl,"is",stability)
        stability_pairs.append((pkl,stability))

    print(stability_pairs)
    pickle.dump(stability_pairs,open('final results %s.pkl'%randint(0,1000),'wb'))
    print("Done!")

if __name__ == "__main__":
    import numpy as np
    from glob import glob
    from os.path import join
    import pickle
    import itertools
    from sklearn.metrics.pairwise import pairwise_distances
    from joblib import Parallel, delayed
    from random import randint

    main()
