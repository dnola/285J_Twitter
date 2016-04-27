import pickle
import numpy as np

(W, H) = pickle.load(open('NMF_100_topics_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))

print(H.shape)
print(names[:100])
print(H[:20,:20])

sorted = [list(names[np.argsort(x)[-10:]][::-1]) for x in H]
for s in sorted:
    print(s)


# Interesting topics!:
# ['follow', 'please', 'make', 'dream', 'babe', 'much', 'girl', 'back', 'world', 'amazing']
# ['buenas', 'noches', 'tardes', 'descanso', 'sueos', 'felices', 'besos', 'tods', 'buen', 'dulces'] - things said at night, like sweet dreams and such (sueos is dreams)
# ['love', 'much', 'amazing', 'really', 'everything', 'dont', 'guys', 'best', 'know', 'adore']
# ['lol', 'hahaha', 'dont', 'know', 'like', 'yeah', 'haha', 'omg', 'jugar', 'want']
# ['gente', 'buena', 'gusta', 'mucha', 'esa', 'encanta', 'mala', 'asco', 'suerte', 'quiere'] love and hate terms - asco is disgust, suerte is luck