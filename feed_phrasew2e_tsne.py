import gensim
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

modelfilename = "c:\\users\\maschram\\downloads\\testmodel.bin"
testqueryfilename = "c:\\users\\maschram\\downloads\\testqueries.tsv"
outdim=3
stoplib=set(stopwords.words('english'))

traineddata = gensim.models.KeyedVectors.load_word2vec_format(modelfilename, binary=True)

modelresult = [];

#convert the list of queries into word vectors
print("generating word vectors")
counter=0
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
        if(counter>50):
           break

wordveclist = [x[1] for x in modelresult]

#create the t-sne model and fit points
print("generating tsne")
tsnemodel = TSNE(n_components = outdim, random_state = 0)
tsnevec = tsnemodel.fit_transform(wordveclist)

#cluster points together
print("generating clusters")
clustermodel = KMeans(n_clusters=12, random_state=0)
tsnecluster = clustermodel.fit_predict(tsnevec)

index = 0
for index in range(len(modelresult)):
    modelresult[index].append(tsnevec[index])
    modelresult[index].append(tsnecluster[index])


#sort by cluster
print('sorting by clusters')
modelresult.sort( key=lambda x: x[3] )


for mr in modelresult:
    print(str(mr[0])+"\t"+str(mr[3]))


