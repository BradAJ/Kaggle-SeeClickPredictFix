__author__ = "Brad Jacobs"
__email__ = "bradaj@gmail.com"
__date__ = "Dec. 2013"

import numpy as np
import pandas as pd
import math
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import scipy.sparse
import sklearn.naive_bayes



def statement_stemmer(stmt):    
    """
    str -> [list of strs]

    Return list of stemmed words from a single statement string.
    """
    stmt_stems = []

    #only stem statements with strings (used in statement_corpus() below)
    if type(stmt) is float:
        return
    
    #stmt_text_str = rev_text.encode('utf-8')  ##deprecated.
    stmt_text_str = stmt.lower()
    
    
    sentence_list = nltk.sent_tokenize(stmt_text_str)
    
    for sent in sentence_list:
        word_list = nltk.word_tokenize(sent)
        for word in word_list:  #compare with WordNet Corpus
            wn_word = wordnet.morphy(word)
            if wn_word is not None:
                wn_stem = PorterStemmer().stem_word(wn_word)
                stmt_stems.append(wn_stem)
    return stmt_stems

def statement_corpus(stmt_list):
    """
    [list of strs] -> set(strs)
    
    Given a list of strings, stem and aggregate them to build a corpus
    of words from the strings.
    """

    corpus = set()
    
    for stmt in stmt_list:
        stmt_stemlist = statement_stemmer(stmt)
        if stmt_stemlist is not None:
            for stem in stmt_stemlist:
                corpus.add(stem)
    return corpus

def corp_table(corpus, stmt_list, stem = True, return_corp_dict = False):
    """
    set(strs), [list of strs] -> scipy.sparse.lil_matrix()
    OR
    set(strs), [list of strs] -> scipy.sparse.lil_matrix(), dict(str: total count of appearances)

    Given a corpus of words and a list of statements, build
    a sparse matrix of dimensions #statements x #words with 
    counts of the number of times a word appears in a statement.

    Optionally perform word stemming (default True)
    Optionally return a dict with each word and its number of appearances in the stmt_list (default False).
    """
    corp_list = list(corpus)
    corp_tab = scipy.sparse.lil_matrix((len(stmt_list), len(corp_list)))
    counter = 0

    if return_corp_dict:
        corp_dict = {word: 0 for word in corp_list}
    

    for stmt in stmt_list:
        if stem:
            stmt_stemlist = statement_stemmer(stmt)
        else:
            stmt_stemlist = stmt

        if stmt_stemlist is not None:
            stmt_stemset = set(stmt_stemlist)
            for stemmed in stmt_stemset:
                if stemmed in corpus:
                    corp_tab[counter, corp_list.index(stemmed)] = stmt_stemlist.count(stemmed)
                    if return_corp_dict:
                        corp_dict[stemmed] += 1
        counter += 1

    if return_corp_dict:
        return corp_tab.tocsr(), corp_dict
    else:
        return corp_tab.tocsr()

def rmsle(pred, ans):
    """
    [list of ints], [list of ints] -> float
    Calculate the RMS Log Error between a set of predictions, and their correponding answers.
    """
    no_samps = float(len(pred))
    err = math.sqrt( 1.0/no_samps * np.sum((np.log1p(np.float64(pred)) - np.log1p(np.float64(ans)))**2.0))
    return err

def source_numerator(train_df, joined_table = True):
    """
    Data in the SeeClickFix set comes from the sources and id_tags listed below.  

    Return a table with both ids and sources in a single table 
    (each row will have exactly two non-zero entries).

    Assign these a numeric category.
    
    """
    #replacing "NaN" for "NaN_str" below.
    sources = ['NaN_str',
    'Map Widget',
    'Mobile Site',
    'New Map Widget',
    'android',
    'city_initiated',
    'iphone',
    'remote_api_created',
    'web']

    
    id_tags = ['NaN_str',
    'abandoned_vehicle',
    'abandoned_vehicles',
    'animal_problem',
    'bad_driving',
    'bench',
    'bike_concern',
    'blighted_property',
    'bridge',
    'crosswalk',
    'drain_problem',
    'drug_dealing',
    'flood',
    'graffiti',
    'heat',
    'homeless',
    'hydrant',
    'illegal_idling',
    'lost_and_found',
    'noise_complaint',
    'odor',
    'other',
    'overgrowth',
    'parking_meter',
    'pedestrian_light',
    'pothole',
    'prostitution',
    'public_art',
    'public_concern',
    'road_safety',
    'roadkill',
    'robbery',
    'rodents',
    'sidewalk',
    'signs',
    'snow',
    'street_light',
    'street_signal',
    'test',
    'traffic',
    'trash',
    'tree',
    'zoning']


    src_list = train_df.source.fillna(value = "NaN_str")
    id_list = train_df.tag_type.fillna(value = "NaN_str")
    
    sources_ids = sources + id_tags
    src_id_tab = scipy.sparse.lil_matrix((len(src_list), len(sources_ids)))
    
    for i in xrange(len(src_list)):
        if src_list[i] in sources_ids:
            src_id_tab[i, sources_ids.index(src_list[i])] = 1
        if id_list[i] in sources_ids:
            src_id_tab[i, sources_ids.index(id_list[i])] = 1
    return src_id_tab.tocsr()




def four_cities_table(train_df):
    """
    Given the longitude of an entry classify it as one of 4 cities.
    0 = Oakland
    1 = Chicago
    2 = Richmond
    3 = New Haven

    Return a 4 column CSR sparse matrix where each row has exactly one entry of 1
    matching the columns numbers above.
    """

    lons = train_df.longitude

    city_class = scipy.sparse.lil_matrix((len(lons), 4))

    for i in xrange(len(lons)):
        if lons[i] < -100.0:
            city_class[i, 0] = 1
        elif lons[i] < -82.0: # -88 < chicago < -86 
            city_class[i, 1] = 1
        elif lons[i] < -75.0:
            city_class[i, 2] = 1 # -78 < richmond < -76 
        else:
            city_class[i, 3] = 1 # -73 < new haven < -71

    return city_class.tocsr()
        

def table_collector(train_df, desc_summ_corpus_tuple = None):
    """
    pd.DataFrame("seeclickfix train/test data") --> scipy.sparse.csrmatrix()

    Build a sparse CSR matrix with one row for each entry in the input DataFrame,
    that contains city, source, id_tag, description word counts, and summary word counts.

    There are 4 city columns, 9 source columns, and 43 id_tags in the training set.
    The word stemming done in the statement_corpus() function generates 8200 word 
    columns from the description field in the training set, and 3797 word columns
    from the summary dataset.  For a total of 12073 columns returned.
    """
    if desc_summ_corpus_tuple is None:
        desc_corpus = statement_corpus(train_df.description)
        summ_corpus = statement_corpus(train_df.summary)
    else:
        desc_corpus = desc_summ_corpus_tuple[0]
        summ_corpus = desc_summ_corpus_tuple[1]

    desc_tab = corp_table(desc_corpus, train_df.description, stem = True)
    summ_tab = corp_table(summ_corpus, train_df.summary, stem = True)
    
    src_tag_tab = source_numerator(train_df, joined_table = True)

    cities_tab = four_cities_table(train_df)

    return scipy.sparse.hstack([cities_tab, src_tag_tab, desc_tab, summ_tab]).tocsr()


def main():
    """
    Run in same directory as the train.csv and test.csv files.
    Returns: test_preds.csv

    For use with command line.  Load data and train a Naive Bayes Classifier and
    return predictions for number of views, votes and comments a 311 issue will
    receive in the test set.  

    Note1: There is a dramatic increase in the number of automatically generated 
    issues in the training set around the beginning of November 2012.  This skews
    the data to a degree such that I found that the simplest solution is to exclude
    these first 40000 issues from training.

    Note2: The time component to the data makes validation a challenge the best results
    I've found come from simply training on the early 70% of the data (starting with
    the 40000th entry) and validating on the last 30% of the training data.  For the 
    Naive Bayes model presented here the only parameter to adjust is the learning rate 
    alpha.  I found the best results with alpha = 0.6, 0.4, 23.0 for views, votes, and 
    comments, respectively, using this split.  For use with the test set the code below
    trains on 40000:end of the training set.


    """
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    desc_corp = statement_corpus(train_df.description)
    summ_corp = statement_corpus(train_df.summary)

    train_tab = table_collector(train_df, desc_summ_corpus_tuple = (desc_corp, summ_corp))
    test_tab = table_collector(test_df, desc_summ_corpus_tuple = (desc_corp, summ_corp))

    nb_views = sklearn.naive_bayes.MultinomialNB(alpha = 0.6)
    nb_votes = sklearn.naive_bayes.MultinomialNB(alpha = 0.4)
    nb_comments = sklearn.naive_bayes.MultinomialNB(alpha = 23.0)
    
    nb_views.fit(train_tab[40000:, :], train_df.num_views.loc[40000:])
    nb_votes.fit(train_tab[40000:, :], train_df.num_votes.loc[40000:])
    nb_comments.fit(train_tab[40000:, :], train_df.num_comments.loc[40000:])

    test_df['num_views'] = nb_views.predict(test_tab)
    test_df['num_votes'] = nb_votes.predict(test_tab)
    test_df['num_comments'] = nb_comments.predict(test_tab)

    test_df[['id', 'num_views', 'num_votes', 'num_comments']].to_csv('test_preds.csv', index = False)

#if __name__ == "__main__":
#    main()
    
