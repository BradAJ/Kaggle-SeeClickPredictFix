import numpy as np
import pandas as pd
import math
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import scipy.sparse
import sklearn.cross_validation
import random


def statement_stemmer(rev):    
    """
    str -> [list of strs]

    Return list of stemmed words from a single statement string.
    """
    rev_stems = []
    if type(rev) is float:
        return

    rev_text_str = rev.lower()
    #rev_text_str = rev_text.encode('utf-8')
    
    sentence_list = nltk.sent_tokenize(rev_text_str)
    
    for sent in sentence_list:
        word_list = nltk.word_tokenize(sent)
        for word in word_list:
            wn_word = wordnet.morphy(word)
            if wn_word is not None:
                wn_stem = PorterStemmer().stem_word(wn_word)
                rev_stems.append(wn_stem)
    return rev_stems

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

def corp_table(corpus, stmt_list, stem = True):
    """
    set(strs), [list of strs] -> scipy.sparse.lil_matrix()

    Given a corpus of words and a list of statements, build
    a sparse matrix of dimensions #statements x #words with 
    counts of the number of times a word appears in a statement.
    """
    corp_list = list(corpus)
    corp_tab = scipy.sparse.lil_matrix((len(stmt_list), len(corp_list)))
    counter = 0
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
        counter += 1

    return corp_tab.tocsr()

def rmsle(pred, ans):
    """
    [list of ints], [list of ints] -> float
    Calculate the RMS Log Error between a set of predictions, and their correponding answers.
    """
    no_samps = float(len(pred))
    err = math.sqrt( 1.0/no_samps * np.sum((np.log(np.float64(pred) + 1.0) - np.log(np.float64(ans) + 1.0))**2.0))
    return err

def source_numerator(train_df, joined_table = True):
    """
    Data in the SeeClickFix set comes from the following sources and id_tags.  If joined_table, then return 
    a table with both ids and sources in a single table (each row will have exactly two non-zero entries)
    else return separate tables.

    Assign these a numeric category.
    
    """
    #replace "NaN" for "NaN_str" below.
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
    if joined_table:
        sources_ids = sources + id_tags
        src_id_tab = scipy.sparse.lil_matrix((len(src_list), len(sources_ids)))
        
        for i in xrange(len(src_list)):
            if src_list[i] in sources_ids:
                src_id_tab[i, sources_ids.index(src_list[i])] = 1
            if id_list[i] in sources_ids:
                src_id_tab[i, sources_ids.index(id_list[i])] = 1
        return src_id_tab.tocsr()
    else:
        
        #sources_tab = scipy.sparse.lil_matrix((len(src_list), len(sources)))
        #ids_tab = scipy.sparse.lil_matrix((len(id_list), len(id_tags)))
        sources_tab = []
        ids_tab = []
        #src_counter = 0
        for src in src_list:
            if src in sources:
                sources_tab.append(sources.index(src))
            else:
                sources_tab.append(len(sources)) #unknown source: max(source index)+1
            #src_counter += 1

        #id_counter = 0
        for idtag in id_list:
            if idtag in id_tags:
                ids_tab.append(id_tags.index(idtag))
            else:
                ids_tab.append(len(id_tags)) #unknown idtag: max(id_tag index)+1
            #id_counter += 1
        
        return sources_tab, ids_tab


def four_cities(train_df):
    """
    Given the lat/lon of an entry classify it as one of 4 cities.
    0 = Oakland
    1 = Chicago
    2 = Richmond
    3 = New Haven?

    """

    lons = train_df.longitude

    city_class = np.zeros([len(lons), 1])

    for i in xrange(len(lons)):
        if lons[i] < -100.0:
            continue
        elif lons[i] < -82.0: # -88 < chicago < -86 
            city_class[i] = 1
        elif lons[i] < -75.0:
            city_class[i] = 2 # -78 < richmond < -76 
        else:
            city_class[i] = 3 # -73 < new haven < -71

    return city_class

def four_cities_table(train_df):
    """
    Given the lat/lon of an entry classify it as one of 4 cities.
    0 = Oakland
    1 = Chicago
    2 = Richmond
    3 = New Haven?

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
    Load with train_df = pd.read_csv("train.csv", quotechar = '"')
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

def cols_for_rfr(train_df):
    """
    rfcols1 = 
    ['comments_desc',
    'votes_tabcollect',
    'views_tabcollect',
    'city_class',
    'source_class',
    'tag_class',
    'secs_since_20120101']
    
    The prediction columns are assumed to be present.  Add the remaining for columns.
    Models are saved as:

    """
    train_df['city_class'] = four_cities(train_df)
    trainsrc, traintag = source_numerator(train_df, joined_table = False)

    train_df['tag_class'] = traintag
    train_df['source_class'] = trainsrc
    
    secs = []
    for dt in train_df.created_time:
        secs.append((pd.to_datetime(dt) - pd.datetime(2012,1,1,0,0,0)).total_seconds())
    
    train_df['secs_since_20120101'] = secs


def ten_fold_cross_val_mess(train_arr, targets, model, eval_measure, shuffle = False):
    
    
    
    err_meas = []
    train_rows = np.shape(train_arr)[0]
    kfold = sklearn.cross_validation.KFold(train_rows, n_folds = 10)

    if shuffle:
        rand_inds = range(train_rows)
        random.shuffle(rand_inds)
        train_arr = train_arr[rand_inds, :]
        targets = targets[rand_inds]

    for train_inds, cv_inds in kfold:
        model.fit(train_arr[train_inds, :], np.round(np.log1p(targets[train_inds])))
        err_meas.append(eval_measure(np.expm1(model.predict(train_arr[cv_inds])), targets[cv_inds]))

    err_meas2 = []
    for train_inds, cv_inds in kfold:
        model.fit(train_arr[train_inds, :], targets[train_inds])
        err_meas2.append(eval_measure(model.predict(train_arr[cv_inds]), targets[cv_inds]))


    print "Err on ln(x+1), Err on x:", sum(err_meas) / 10.0, sum(err_meas2) / 10.0

        
