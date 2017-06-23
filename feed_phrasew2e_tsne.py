import gensim
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import time

modelfilename = "c:\\users\\maschram\\downloads\\testmodel_300.bin"
testqueryfilename = "c:\\users\\maschram\\downloads\\testqueries.tsv"
outdim=3
maxcount = 500
stoplib=set(stopwords.words('english'))

traineddata = gensim.models.KeyedVectors.load_word2vec_format(modelfilename, binary=True)

modelresult = [];

#convert the list of queries into word vectors
print("generating word vectors")
counter=0
t0 = time.clock()
with open(testqueryfilename, encoding="utf8") as queryfile:
    for line in queryfile:
        for phrase in line.split("\t")[0].split("|"):
            wordvec = [];
            for word in phrase.split(" "):
                if(word in stoplib):
                    continue
                if(word in traineddata.vocab):
                    if(wordvec == []):
                        wordvec = traineddata.word_vec(word)
                    else:
                        wordvec += traineddata.word_vec(word)
            if wordvec != []:
                modelresult.append( [phrase, wordvec] )
               
        counter+=1
        if(counter>maxcount and maxcount!=-1):
           break

wordveclist = [x[1] for x in modelresult]
print(str(time.clock()-t0)+" seconds")

#create the t-sne model and fit points
print("generating tsne")
t0=time.clock()
tsnemodel = TSNE(n_components = outdim, random_state = 0)
tsnevec = tsnemodel.fit_transform(wordveclist)
print(str(time.clock()-t0)+" seconds")

##cluster points together
#print("generating clusters")
#t0=time.clock()
#clustermodel = KMeans(n_clusters=12, random_state=0)
#tsnecluster = clustermodel.fit_predict(tsnevec)

#index = 0
#for index in range(len(modelresult)):
#    modelresult[index].append(tsnevec[index])
#    modelresult[index].append(tsnecluster[index])
#print(str(time.clock()-t0)+" seconds")


#cluster points together
print("generating clusters - Birch clustering")
t0=time.clock()
clustermodel = Birch(threshold=1, n_clusters=None)
tsnecluster = clustermodel.fit_predict(tsnevec)
#clustermodel = KMeans(n_clusters=12, random_state=0)
#tsnecluster = clustermodel.fit_predict(tsnevec)

index = 0
for index in range(len(modelresult)):
    modelresult[index].append(tsnevec[index])
    modelresult[index].append(tsnecluster[index])
print(str(time.clock()-t0)+" seconds")



#sort by cluster
t0=time.clock()
print('sorting by clusters')
modelresult.sort( key=lambda x: x[3] )
print(str(time.clock()-t0)+" seconds")


for mr in modelresult:
    print(str(mr[0])+"\t"+str(mr[3]))


