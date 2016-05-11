

if __name__ == "__main__": # sort of like with MPI, we need this to do multiprocessing on windows
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, PCA, TruncatedSVD, LatentDirichletAllocation
    import time
    import re  # regex

    start_time = time.time()

    raw_text = pickle.load(open('raw_text_data.pkl','rb'))

    # Make sure NaNs turn into strings
    # (We probably don't want this in the long run)
    raw_text = [str(x) for x in raw_text]
    print("Number of Samples:", len(raw_text))

    clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text]


    stop_words = [] # stop words file includes English, Spanish, and Catalan
    with open('stop_words.txt','r') as f:
        stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly

    print("Stop word examples:", stop_words[:10])

    print("\n----20 TWEETS----")
    # Lets make sure this looks right...
    for tweet in clean_text[:20]: # First 20 tweets!
        print(tweet) # the b before these means they are ascii encoded
    print("--------------")


    tf_idf = TfidfVectorizer(min_df=10,stop_words=stop_words)
    # min_df means ignore words that appear in less than that many tweets
    # we specify our stop words list here too

    text_tf_idf = tf_idf.fit_transform(clean_text) # like we talked about,
    # fit_transform is short hand for doing a .fit() then a .transform()
    # because 2 lines of code is already too much I guess...

    print("Dumping feature names to disk...")
    pickle.dump(tf_idf.get_feature_names(), open('TF_IDF_feature_names.pkl', 'wb'))


    print("TF-IDF Sample:")
    print(text_tf_idf[0],"\n") # Important note! TF-IDF spits out a sparse matrix by
    # default so don't be shocked when the print statement gives you
    # (0, INDEX) VALUE
    # Its just a more concise way of writing it - we leave out all the 0 entries
    # and only spit out the values and their indexes. Its 1D so the first index
    # will always be 0

    # These two are basically interchangable: (Go take a nap while they run...)
    # nmf = NMF(n_components=1000,verbose=1,tol=.001,alpha=.1,l1_ratio=.2)
    topic_model = LatentDirichletAllocation(n_topics=1000,n_jobs=-1,doc_topic_prior=50/1000.0, topic_word_prior=.1, verbose=100,batch_size=int(len(raw_text)/10),max_doc_update_iter=1000,mean_change_tol=.0001,learning_offset=30)

    # for LDA:
    # n_jobs=-1 means run on every logical core
    # doc_topic prior and topic_word prior are alpha and beta terms respectively
    # LatentDirichletAllocation runs in online mode by default, just make sure batch_size is small enough to fit in RAM
    # Learning offset is how much to reduce importance of early batches - higher values mean early batches have less weight (early batches tend to dominate in online training of LDA)


    # for NMF:
    # alpha is how much to regularize
    # l1 ratio is how much alpha to allocate to l1 vs l2 regularization (microblogs paper did both so we do too)
    # tol is how small violation must be for NMF to stop iterating.
    # I set it higher than default so it doesn't take as long to run. Smaller is usually better though
    # Sure lets compress to 100 topics why not
    # Verbose prints out how close to convergence we are after each iteration
    # When violation is less than .0001 by default, NMF is finished (set to .001 now)

    text_topic_model_W = topic_model.fit_transform(text_tf_idf) # NMF's .transform() returns W by
    # default, but we can get H as follows:
    text_topic_model_H = topic_model.components_
    print("Topic Model Components:")
    print(text_topic_model_W[0]) # topic memberships of tweet 0
    print(len(text_topic_model_H[0]))
    print(text_topic_model_H[0]) # this is relative word frequencies within topic 0.
    # Maybe. We might need to to transpose this...

    text_topic_model_WH = (text_topic_model_W,text_topic_model_H)

    pickle.dump(text_topic_model_WH, open('LDA_topics_WH.pkl','wb'), protocol=4) # Save it to
    pickle.dump(topic_model, open('LDA.pkl','wb'), protocol=4)
    # disk so we don't have to keep recalculating it later

    print("\n--- Completed in %s seconds! ---" % (time.time() - start_time))