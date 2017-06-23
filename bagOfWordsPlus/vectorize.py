# filter with porter stemming and stop words,
# then vectorize into bag of words
#
# python vectorize.py cleaned.txt

import sys
import string
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy

stemmer = PorterStemmer()

queryData = []
verticalData = []
countryData = []
browserData = []
osData = []

dataLimit = 2000

def isEnglish(s):
    try:
        s.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
	
def getVector(input):

	#vectorizer = CountVectorizer(stop_words = 'english')
	vectorizer = TfidfVectorizer(stop_words = 'english', norm = 'l2')
	return vectorizer.fit_transform(input).toarray()
	
inputFile = open("session_data_all_2017-06-19.ss_TOP_10000.csv",'r')	
labelFile = open("labels.txt",'w')
labelFile2 = open("labels_query.txt",'w')

i = 0
for line in inputFile.readlines():

	if i == 0:
		labelFile.write(line.strip().replace(',','\t') + '\n')
		i = i + 1
		continue

	tokens = line.strip().split(',')
	if len(tokens) != 6:
		continue
	
	queries = tokens[0].strip().replace('|',' ')
	
	if not isEnglish(queries):
		continue
	
	queries = stemmer.stem(queries)
	
	# filter out anything < length 3
	if len(queries) < 3:
		continue	
	
	queryData.append(queries)
	verticalData.append(tokens[1].strip())
	countryData.append(tokens[2].strip())
	browserData.append(tokens[3].strip())
	osData.append(tokens[4].strip())
	labelFile.write("\t".join([queries, tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]]))
	labelFile.write('\n')
	labelFile2.write(queries)
	labelFile2.write('\n')
	
	i = i + 1
	if i > dataLimit:
		break	

labelFile.close()

print "output sizes: "
queryData = getVector(queryData)
output = queryData
verticalData = getVector(verticalData)
output = numpy.append(output, verticalData, axis=1)
countryData = getVector(countryData)
output = numpy.append(output, countryData, axis=1)
browserData = getVector(browserData)
output = numpy.append(output, browserData, axis=1)
osData = getVector(osData)
output = numpy.append(output, osData, axis=1)

print queryData.shape
print verticalData.shape
print countryData.shape
print browserData.shape
print osData.shape
print output.shape

with open("data_query.tsv",'w') as outputFile:
	for vec in queryData:
		isFirst = True
		for item in vec:
			if isFirst:
				isFirst = False
			else:
				outputFile.write('\t')
			outputFile.write(str(item))
		outputFile.write('\n')

with open("data.tsv",'w') as outputFile:
	for vec in output:
		isFirst = True
		for item in vec:
			if isFirst:
				isFirst = False
			else:
				outputFile.write('\t')
			outputFile.write(str(item))
		outputFile.write('\n')
