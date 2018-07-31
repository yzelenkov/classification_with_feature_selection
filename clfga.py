import csv
import numpy as np
import sklearn.tree as tree
import sklearn.model_selection as cross_validation
import sklearn.svm as svm
import sklearn.neighbors as knn
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb
import sklearn.ensemble as ens
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import sklearn.discriminant_analysis as da
import random
import copy
import pandas as pd



def version():
    return "0.9"

def dataRead(filename, norm_distr = False):
# Read source data and:
# if norm_dist = True - scale it to zero mean and unit variance (standard normally distributed data)
# else - scale features to range {-1;1}
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        feature_names = next(reader)
        for row in reader:
            data.append([])
            for index,item in enumerate(row):
                data[-1].append(float(item.replace(',','.')))
    data1 = np.array(data)
    if norm_distr:
        data = preprocessing.scale(data1)
    else:
        max_abs_scaler = preprocessing.MaxAbsScaler()
        data = max_abs_scaler.fit_transform(data1)
    return np.array(feature_names), data

class Classifier:
    def __init__(self,type,number_of_features):
        self.type = type
        self.accuracy = 0.0
        self.std = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.genome = []
        self.rank = []
        self.n_features = 0
        self.clf = None
        self.changed = True
        for i in range(0,number_of_features):
            k = 1
            self.genome.append(k)
            if k > 0:
                self.n_features += 1

    def calc_fitness(self,data, target):
        if self.changed:
            nfolds = 4
            scores = np.zeros(nfolds)
            precision = np.zeros(nfolds)
            recall = np.zeros(nfolds)
            X = np.copy(data)
            for i in range(0,len(self.genome)):
                if self.genome[len(self.genome)-1-i] == 0:
                    X = np.delete(X,len(self.genome)-1-i,1)
            i = 0
            skf = cross_validation.StratifiedKFold(n_splits=nfolds)
            for train, test in skf.split(X,target):
                if self.type == 'dt':
                    self.clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random').fit(X[train],target[train])
                elif self.type == 'svm':
                    self.clf = svm.SVC(kernel='linear').fit(X[train],target[train])
                elif self.type == 'knn':
                    self.clf = knn.KNeighborsClassifier().fit(X[train],target[train])
                elif self.type == 'lr':
                    self.clf = lm.LogisticRegression().fit(X[train],target[train])
                elif self.type == 'nb':
                    self.clf = nb.GaussianNB().fit(X[train],target[train])
                elif self.type == 'rf':
                    self.clf = ens.RandomForestClassifier().fit(X[train],target[train])
                elif self.type == 'et':
                    self.clf = ens.ExtraTreesClassifier().fit(X[train],target[train])
                elif self.type == 'mlp':
                    self.clf = nn.MLPClassifier(hidden_layer_sizes=(40,5)).fit(X[train],target[train])
                elif self.type == 'lda':
                    self.clf = da.LinearDiscriminantAnalysis().fit(X[train],target[train])
                elif self.type == 'qda':
                    self.clf = da.QuadraticDiscriminantAnalysis().fit(X[train],target[train])
                else:
                    self.clf = None
                p = self.clf.predict(X[test])
                scores[i] = metrics.accuracy_score(target[test],p)
                precision[i] = metrics.precision_score(target[test],p)
                recall[i]    = metrics.recall_score(target[test],p)
                i += 1
            self.accuracy = scores.mean()
            self.std = scores.std()
            self.precision = precision.mean()
            self.recall = recall.mean()
            self.changed = False

    def mutate(self):
        mutated_genes = random.randint(1,len(self.genome))
        for j in range(0,mutated_genes):
            i = random.randint(0,len(self.genome)-1)
            if self.genome[i] == 0:
                self.genome[i] = 1
                self.n_features +=1
            else:
                self.genome[i] = 0
                self.n_features -=1
        self.changed = True

    def print(self):
        print('--------------')
        print('item: features = %3d  accuracy = %5.3f (+/- %5.3f) prec = %5.3f rec = %5.3f' % (self.n_features, self.accuracy, self.std * 2, self.precision, self.recall))
        print(self.genome)
        print('--------------')

    def crossover(self,item):
        start = random.randint(0,self.n_features-2)
        end = random.randint(start+1,self.n_features-1)
        for i in range(start,end):
            g = self.genome[i]
            self.genome[i] = item.genome[i]
            item.genome[i] = g
        self.n_features = 0
        item.n_features = 0
        for i in range(0,len(self.genome)):
            if self.genome[i] == 1:
                self.n_features +=1
            if item.genome[i] == 1:
                item.n_features +=1
        self.changed = True
        item.changed = True
        return self, item

class Population:
    def __init__(self,type,pop_size,number_of_features):
        self.size = pop_size
        self.items = []
        for i in range(0,self.size):
            self.items.append(Classifier(type, number_of_features))

    def print(self):
        for i in range(0,self.size):
            print('item %3d : features = %3d  accuracy = %5.3f' % (i, self.items[i].n_features, self.items[i].accuracy ))

    def calc_fitness(self,data,target):
        rank = 0.0
        for i in range(0,self.size):
            self.items[i].calc_fitness(data, target)
            self.items[i].rank = rank + self.items[i].accuracy
            rank = self.items[i].rank

    def select(self):
        r = random.uniform(0.0,self.items[self.size-1].rank)
        for i in range(0,self.size):
            if r < self.items[i].rank:
                return self.items[i]

    def bestItem(self):
        f = 0.0
        for i in range(0,self.size):
            if f < self.items[i].accuracy:
                f = self.items[i].accuracy
                k = i
        return self.items[k]

class Ensemble:
    def __init__(self):
        self.items = []
        self.weights = []
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.std = 0.0
        self.rank = 0.0

    def add_clf(self,clf,w):
        self.items.append(copy.deepcopy(clf))
        self.weights.append(w)

    def calc_fitness(self,data,target,bagging = False):
        if bagging:
            nfolds = 4
            scores = np.zeros(nfolds)
            k = 0
            skf = cross_validation.StratifiedKFold(n_splits=nfolds)
            for train, test in skf.split(data,target):
                p = np.zeros(len(data[train]))
                for j in range(0,len(self.items)):
                    X = np.copy(data[train])
                    for i in range(0,len(self.items[j].genome)):
                        if self.items[j].genome[len(self.items[j].genome)-1-i] == 0:
                            X = np.delete(X,len(self.items[j].genome)-1-i,1)
                    predict = self.items[j].clf.predict(X)
                    for i in range(0,len(X)):
                        p[i] += predict[i] * self.weights[j]
                    for i in range(0,len(p)):
                        if p[i] < 0:
                            p[i] = -1
                        else:
                            p[i] = 1
                    scores[k] = metrics.accuracy_score(target[train],p)
                k += 1
            self.accuracy = scores.mean()
            self.std = scores.std()
        else:
            self.accuracy = 0
            p = np.zeros(len(data))
            for j in range(0,len(self.items)):
                X = np.copy(data)
                for i in range(0,len(self.items[j].genome)):
                    if self.items[j].genome[len(self.items[j].genome)-1-i] == 0:
                        X = np.delete(X,len(self.items[j].genome)-1-i,1)
                predict = self.items[j].clf.predict(X)
                for i in range(0,len(X)):
                    p[i] += predict[i] * self.weights[j]
            for i in range(len(p)):
                if p[i] < 0:
                    p[i] = -1
                else:
                    p[i] = 1
            self.accuracy = metrics.accuracy_score(target,p)
            self.precision = metrics.precision_score(target,p)
            self.recall = metrics.recall_score(target,p)
            self.std = 0.0

    def mutate(self, norm = False):
        for i in range(0,len(self.weights)):
            k = random.uniform(-0.1,0.1)
            self.weights[i] += k
            if self.weights[i] < 0.0:
                self.weights[i] = 0
            if norm:
                s = 0.0
                for i in range(0,len(self.weights)):
                    s += self.weights[i]
                for i in range(0,len(self.weights)):
                    self.weights[i] = self.weights[i] / s


    def summary(self):
        s = 'acc = ' + '{:5.3f}'.format(self.accuracy) + ' prec/rec = {:5.3f}/{:5.3f}'.format(self.precision,self.recall)
        for i in range(0,len(self.weights)):
            s = s + ' ' + self.items[i].type + ':' + '{:5.3f}'.format(self.weights[i])
        return s

    def crossover(self,item):
        start = random.randint(0,len(self.weights)-1)
        end = random.randint(start+1,len(self.weights))
        for i in range(start,end):
            g = self.weights[i]
            self.weights[i] = item.weights[i]
            item.weights[i] = g
        return self, item


class PopulationOfEnsembles:
    def __init__(self):
        self.size = 0
        self.items = []

    def addEnsemble(self,e):
        self.items.append(e)
        self.size += 1

    def calc_fitness(self,data,target, bagging):
        rank = 0.0
        for i in range(0,len(self.items)):
            self.items[i].calc_fitness(data,target, bagging)
            self.items[i].rank = rank + self.items[i].accuracy
            rank = self.items[i].rank

    def select(self):
        r = random.uniform(0.0,self.items[self.size-1].rank)
        for i in range(0,self.size):
            if r < self.items[i].rank:
                return self.items[i]

    def bestItem(self):
        f = 0.0
        for i in range(0,self.size):
            if f < self.items[i].accuracy:
                f = self.items[i].accuracy
                k = i
        return self.items[k]

def EvolveClf(clf_type,number_of_features,pop_size,generations,prob_of_crossover, data, target):
    pop = Population(clf_type,pop_size,number_of_features)
    fitness_story = []
    for g in range(0,generations):

        pop.calc_fitness(data, target)

        bi = copy.deepcopy(pop.bestItem())

        new_pop = Population(clf_type,pop_size,number_of_features)
        new_pop.items = []
        new_pop.items.append(bi)
        fitness_story.append(bi.accuracy)

        i = 1
        while i < pop_size:
            item = copy.deepcopy(pop.select())
            r = random.uniform(0.0,1.0)
            if r <= prob_of_crossover:
                item1, item2 = item.crossover(copy.deepcopy(pop.select()))
                new_pop.items.append(item1)
                new_pop.items.append(item2)
                i += 2
            else:
                item.mutate()
                new_pop.items.append(item)
                i += 1

        pop = copy.deepcopy(new_pop)

    return fitness_story,bi

def EvolveEnsemble(setOfClf, pop_size,generations,prob_of_crossover,data,target,bagging = False, norm = False):

    ensemble = Ensemble()

    for i in range(0,len(setOfClf)):
        ensemble.add_clf(setOfClf[i],1.0/len(setOfClf))

    pop = PopulationOfEnsembles()

    for i in range(0,pop_size):
        ensemble.mutate(norm)
        pop.addEnsemble(ensemble)

    fitness_story = []

    for g in range(0,generations):
        pop.calc_fitness(data,target,bagging)
        eb = copy.deepcopy(pop.bestItem())
        print('g = %2d  %s' % (g, eb.summary()))
        new_pop = PopulationOfEnsembles()
        new_pop.addEnsemble(eb)
        fitness_story.append(eb.accuracy)

        i = 1
        while i < pop_size:
            e = copy.deepcopy(pop.select())
            r = random.uniform(0.0,1.0)
            if r <= prob_of_crossover:
                e1, e2 = e.crossover(copy.deepcopy(pop.select()))
                new_pop.addEnsemble(e1)
                new_pop.addEnsemble(e2)
                i += 2
            else:
                e.mutate(norm)
                new_pop.addEnsemble(e)
            i += 1

        pop = copy.deepcopy(new_pop)

    return fitness_story,eb

if __name__ == '__main__':
    f = '../Trading/Nornikel.txt'

    df = pd.read_csv(f,sep=';', usecols=[2,3,4,5,6,7,8], index_col=[0], parse_dates=[[0, 1]])
    df['Target'] = [*map(lambda x : 1 if x < 10000 else -1, df['<CLOSE>'])]

    train = df.loc['2016-06-01' : '2016-12-31']
    test  = df.loc['2017-01-01' : '2017-12-31']

    data_train = np.array(train.iloc[:,0:5])
    target_train = np.array(train.iloc[:,5:6]).reshape((data_train.shape[0]))


    print(data_train)
    print(target_train)

    print("Train dataset: %5d rows and %d features" %(data_train.shape[0], data_train.shape[1]))
#    print("Test dataset:  %5d rows and %d features" %(target_train.shape[0], target_train.shape[1]))

    number_of_features = data_train.shape[1]

    pop_size = 5
    generations = 5
    prob_of_crossover = 0.5

    clf_set = ('knn','lr','nb','dt','mlp','lda','qda') #rf','et','mlp','svm')

    fitness = []
    setOfClf = []

    for clf_type in clf_set:
        fitness_story, bi = EvolveClf (clf_type,number_of_features,pop_size,generations,prob_of_crossover,data_train,target_train)
        print('%3s: accuracy start = %5.3f fin = %5.3f (+/- %5.3f) features = %2d prec/rec = %5.3f/%5.3f' %(clf_type,fitness_story[0],bi.accuracy,bi.std * 2,bi.n_features,bi.precision,bi.recall))
        fitness.append(fitness_story)
        setOfClf.append(bi)



















