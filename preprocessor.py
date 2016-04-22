import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import time
start_time = time.time()

raw_text = pickle.load(open('raw_text_data.pkl','rb'))

# Make sure NaNs turn into strings
# (We probably don't want this in the long run)
raw_text = [str(x) for x in raw_text]
print("Number of Samples:", len(raw_text))

print("\n----20 TWEETS----")
# Lets make sure this looks right...
for tweet in raw_text[:20]: # First 20 tweets!
    print(tweet)
print("--------------")

tf_idf = TfidfVectorizer() # The defaults are good enough I guess - feel free
#  to change parameters to make it more like the microblog paper!

text_tf_idf = tf_idf.fit_transform(raw_text) # like we talked about,
# fit_transform is short hand for doing a .fit() then a .transform()
# because 2 lines of code is already too much I guess...

print("TF-IDF Sample:")
print(text_tf_idf[0],"\n") # Important note! TF-IDF spits out a sparse matrix by
# default so don't be shocked when the print statement gives you
# (0, INDEX) VALUE
# Its just a more concise way of writing it - we leave out all the 0 entries
# and only spit out the values and their indexes. 

nmf = NMF(n_components=100,verbose=1) # Sure lets compress to 100 topics why not
# Verbose prints out how close to convergence we are after each iteration
# When violation is less than .0001 by default, NMF is finished

text_nmf_W = nmf.fit_transform(text_tf_idf) # NMF's .transform() returns W by
# default, but we can get H as follows:
text_nmf_H = nmf.components_
print("NMF Components:")
print(text_nmf_W[0]) # topic membership's of tweet 0
print(len(text_nmf_H[0]))
print(text_nmf_H[0]) # this is relative word frequencies within topic 0.
# Maybe. We might need to to transpose this...

text_nmf_WH = (text_nmf_W,text_nmf_H)

pickle.dump(text_nmf_WH, open('NMF_100_topics_WH.pkl','wb')) # Save it to
# disk so we don't have to keep recalculating it later

print("\n--- Completed in %s seconds! ---" % (time.time() - start_time))
