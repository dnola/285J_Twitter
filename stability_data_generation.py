
def main():
    print("Starting...")

    ### Experiment Definitions ###
    class Experiments(): # Putting these in a class makes it easy to collapse this stuff in PyCharm
        @staticmethod
        def base_NMF():
            raw_text = pickle.load(open('raw_text_data.pkl', 'rb'))
            raw_text = [str(x) for x in raw_text]
            tf_idf = TfidfVectorizer(min_df=10)
            tf_idf_text = tf_idf.fit_transform(raw_text)
            topic_model = NMF(n_components=500, verbose=1, tol=.001, init='nndsvd')
            return (tf_idf_text,topic_model,tf_idf.get_feature_names(),'base_NMF_500')

        @staticmethod
        def get_clean_data():
            raw_text = pickle.load(open('raw_text_data.pkl', 'rb'))

            # Make sure NaNs turn into strings
            # (We probably don't want this in the long run)
            raw_text = [str(x) for x in raw_text]
            print("Number of Samples:", len(raw_text))

            clean_text = [" ".join([  # joins a list of words back together with spaces in between them
                                      re.sub(r'\W+', '',  # force alphanumeric (after doing @ and # checks)
                                             word.replace('"', '').lower())  # force lower case, remove double quotes
                                      for word in tweet.split()  # go word by word and keep them if...
                                      if len(word) > 2 and  # they are 3 characters or longer
                                      not word.startswith('@') and  # they don't start with @, #, or http
                                      not word.startswith('#') and
                                      not word.startswith('http')]
                                   ).encode('ascii', errors='ignore')
                          # force ascii encoding, ignore weird characters just in case
                          for tweet in raw_text]

            stop_words = []  # stop words file includes English, Spanish, and Catalan
            with open('stop_words.txt', 'r') as f:
                stop_words = [word.replace("\n", '') for word in
                              f.readlines()]  # Have to remove \n's because I didn't copy the stop words cleanly

            tf_idf = TfidfVectorizer(min_df=10, stop_words=stop_words)
            text_tf_idf = tf_idf.fit_transform(clean_text)
            return (text_tf_idf,tf_idf.get_feature_names())

        @staticmethod
        def improved_NMF(quantity=500,alpha=.1):
            (text_tf_idf,words) = Experiments.get_clean_data()
            topic_model = NMF(n_components=quantity, verbose=1, tol=.001, alpha=alpha, l1_ratio=.5, init='nndsvd')
            return (text_tf_idf,topic_model,words,'improved_NMF_%s_alpha_%s'%(quantity,int(alpha*10)))

        @staticmethod
        def base_LDA():
            (text_tf_idf, words) = Experiments.get_clean_data()
            topic_model = LatentDirichletAllocation(n_topics=500, verbose=1,n_jobs=-1,batch_size=int(text_tf_idf.shape[0]/20),max_doc_update_iter=1000,mean_change_tol=.0001)
            return (text_tf_idf,topic_model,words,'base_LDA_500')

        @staticmethod
        def improved_LDA():
            (text_tf_idf, words) = Experiments.get_clean_data()
            topic_model = LatentDirichletAllocation(n_topics=500, verbose=1,n_jobs=-1,batch_size=int(text_tf_idf.shape[0]/20),max_doc_update_iter=1000,mean_change_tol=.0001,doc_topic_prior=50.0/1000.0, topic_word_prior=.1)
            return (text_tf_idf,topic_model,words,'improved_LDA_500')
    ### ### ###

    ### Experiment Selection ###
    subsample_rate = .05
    num_subsamples = 5
    values = [0, .3, .5, .7, .9] # Only used for grid search over penalty term
    # data = shuffle(data, n_samples=int(data.shape[0] * .01))

    # Set experiment set here:
    # experiments = [Experiments.base_NMF(),Experiments.improved_NMF(),Experiments.improved_NMF(100),Experiments.improved_NMF(300)]
    # experiments = [Experiments.base_LDA(),Experiments.improved_LDA()]
    # experiments = [Experiments.improved_NMF(alpha=v) for v in values][:3] #basic grid search
    experiments = [Experiments.improved_NMF(alpha=v) for v in values][3:] #basic grid search

    print("Done generating experimental data...")
    ### ### ###

    ### Helper Functions ###
    def get_assignments(subsample, data, model, words):
        model = clone(model)  # we don't want to keep refitting same model - sometimes scikit treats multiple fits as 'adding in' more data
        print(subsample.shape[0])
        print(subsample[0])
        model.fit(subsample)
        H = model.components_

        print("Done fitting model...")
        W = model.transform(data)
        print("Done transforming data...")
        assignments = np.argmax(W, axis=1)
        top_words = [list(np.array(words)[np.argsort(x)[-10:]][::-1]) for x in H]
        print(top_words[0])
        return (assignments, top_words)
    ### ### ###

    ### Main Code ###
    for (data, model, words, name) in experiments:
        print("Running experiment %s..."%name)

        subsamples = [shuffle(data,n_samples=int(data.shape[0]*subsample_rate)) for i in range(num_subsamples)]
        print("Done generating subsamples...")
        # [print(s[:3],"\n") for s in subsamples]
        assignments = [get_assignments(s,data,model,words) for s in subsamples]
        print("Done generating assignments...")
        a = assignments[0]
        print(a[:20])
        print(len(a))

        pickle.dump(assignments, open(join('Stability Results','%s_assignments.pkl' % name),'wb'))
    ### ### ###

if __name__ == "__main__":

    import pickle
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.utils import shuffle
    from sklearn.base import clone
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from os.path import join
    import re

    main()