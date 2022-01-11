import warnings

import pandas as pd
import numpy as np
import os
import gzip
import datetime
import re
import conllu
import gensim
import pickle

from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit

# COMMENTARE

# *************************** FUNZIONI ***************************


# Funzione che analizza i token file per file, e ne estrae un unica token list
from sklearn.preprocessing import StandardScaler


def create_token_list(cig_list, folder_path):
    cig_dict = {}
    # Cambio la cartella corrente con quella passata alla funzione
    os.chdir(folder_path)
    # Creo una variabile che contiene l'estensione ".gz"
    extension = ".gz"
    # Ottengo una lista con i file contenuti in una cartella
    folder_files = os.listdir(folder_path)
    # Lista che conterrà i token di tutti i documenti
    list_of_token = []
    # Per ogni file all'interno della cartella
    for file in folder_files:

        start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

        # Prendo il path del file
        file_path = os.path.relpath(file)
        # Controllo se il file ha l'estensione ".gz"
        if file.endswith(extension):
            # Prende i documenti all'interno dell'archivio e li analizza
            with gzip.open(file_path) as unzip_file:
                print("Parsing", file, "...")
                # Creo una lista vuota temporanea
                # Riempio la lista con i token all'interno di file
                cig_c = fill_token_list(cig_list, cig_dict, unzip_file)
                # Per ogni documento nella lista dei token del file
                # for li in tmp_list:
                #     # Aggiungo alla lista da restituire la lista dei token di un singolo file documento per documento
                #     list_of_token.append(li)

                end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
                print("File:", file, "parsed in", ((end_time - start_time).total_seconds() / 60),
                      "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    return cig_c


# Funzione che rimuove le stopwords dalla lista dei token della frase passata
def stop_word_remover(tokens):
    # Lista delle stopwords
    stop_words = stopwords.words('italian')
    # Lista stop words extra
    extra_stop_words = ["l’", "l'", "dell’", "dell'", "all’", "all'", "L’", "L'", "dall’", "dall'", "sull’", "sull'",
                        "un'", "un’", "il", "nell’", "nell'", "d’", "d'"]

    # Per ogni token della lista della frase analizzata
    for x in tokens.copy():
        # essendo x un oggetto di tipo classe, lo trasformo in stringa per confrontarlo con le stopword
        str_x = x.__str__()
        # rendo minuscola la stringa
        str_x = str_x.lower()
        # funzione per controllare se i token sono delle parole, utilizzando le regex
        match_w = re.match("^[a-z]+$", str_x)
        # se il token non è una parola allora lo rimuovo dalla token list
        if not match_w:
            tokens.remove(x)
        # se il token in questione è una parola e fa parte delle stopwords, oppure è un link, allora viene rimosso
        # dalla lista dei Token
        elif (str_x in stop_words) or (str_x in extra_stop_words) or (re.match("^http", str_x)) or \
             (re.match("^mailto", str_x)) or (re.match("^www", str_x)) or (len(str_x) < 3) or (re.match("_", str_x)):
            tokens.remove(x)


# Funzione che analizza una stringa in formato CoNLL-U, e ne aggiunge i token alla lista dei token di un documento
def token_parser(conll_string, docu):
    # Analizzo una stringa in formato CoNLL-U
    parsed_conll = conllu.parse(conll_string)
    # Per ogni lista di token dalla stringa analizzata
    for token_l in parsed_conll:
        # Ne rimuovo le stopwords
        stop_word_remover(token_l)
        # Per ogni token nella lista
        for token in token_l:
            # Ne estraggo l'attributo 'form' dal formato CoNLL-U
            y = token['lemma']
            # Lo rendo minuscolo
            y = y.lower()
            # Lo aggiungo alla lista dei token di un documento
            docu.append(y)


# Funzione che carica il file json e ne crea la token list
def fill_token_list(cig_list, cig_dictionary, unzipped_file):
    # Carico il dataframe in chunks, per evitare che la memoria si sovraccarichi
    chunks = pd.read_json(unzipped_file, lines=True, chunksize=10000)
    # Immetto in una variabile la grandezza del chunk
    chunk_size = chunks.chunksize
    count = 0
    # Per ogni chunk
    for c in chunks:
        # Imposto l'indice di un chunk in un range che va da 0 alla dimensione del chunk
        #c.index = pd.RangeIndex(len(c))
        c = c.groupby('title')
        for c_group in c:
            # Creo una lista vuota, che conterrà i token per un cig
            path = c_group[0]
            conll_lines = c_group[1]['conll'].values.tolist()
            # Normalizzo il path per os
            norm_path = os.path.normpath(path)
            path_elements = norm_path.split(os.sep)
            cig = path_elements[7]
            # for element in path_elements:
            #     if re.match("^[0-9]{7}[A-Z0-9]*$", element):
            #         cig = element
            if cig in cig_list:
                # Se la chiave è già nel dizionario
                if cig in cig_dictionary:
                    for document in conll_lines:
                        cig_tokens = []
                        token_parser(document, cig_tokens)
                        # Se la lista dei token non è vuota (possono succedere casi in cui è vuota perchè la frase contiene soltanto termini da eliminare)
                        if cig_tokens:
                            cig_dictionary[cig].extend(cig_tokens)
                else:
                    # aggiungo una nuova chiave alla dict con valore una lista vuota
                    cig_dictionary[cig] = []
                    for document in conll_lines:
                        cig_tokens = []
                        token_parser(document, cig_tokens)
                        if cig_tokens:
                            # creo la chiave contenente il nuovo cig, e aggiungo i token di questo chunk
                            cig_dictionary[cig].extend(cig_tokens)

    # Restituirà una lista contenente la lista dei token di un file documento per documento
    return cig_dictionary


def bigrams(words, bi_min=15, tri_min=10):
    # Costruzione bigram model
    bi_gram = gensim.models.Phrases(words, min_count=bi_min)  # Gensim’s Phrases class allows you to group related phrases into one token for LDA.
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bi_gram)
    return bigram_mod


# *************************** MAIN ***************************

if __name__ == '__main__':
    print("ESPERIMENTO CON DIVISIONE SEMPLICE DEL DATASET")

    data_df = pd.read_csv('Dataset/gare-empulia-cpv.csv')
    labels = data_df.COD_CPV
    data_df = data_df.drop(columns=['DESCRIZIONE_CPV', 'OGGETTO_LOTTO'])
    train, test = train_test_split(data_df, test_size=0.3, random_state=0)
    cig_train_list = train['CIG'].values.tolist()
    print(cig_train_list)
    print("______CIG_TRAIN_LIST:", len(cig_train_list))

    data_folder = "D:/Marilisa/Tesi/Dataset/process_bandi_cpv/udpipe"  # "D:/PycharmProject/pythonproject/provaset"

    # Chiamo la funzione create_token_list passandogli il parametro che inizia tutto il processo che restituirà
    # la token list di tutti i file documento per documento
    token_list = create_token_list(cig_train_list, data_folder)

    print("______NUM_CIG:", len(token_list))

    train_label = []
    a = train.CIG.values.tolist()
    b = train.COD_CPV.values.tolist()
    for key in token_list.keys():
        print("CIG:", key)
        for i in range(len(train)):
            if a[i] == key:
                print(b[i])
                train_label.append(b[i])
    print(train_label)

    tmp_list = []
    for ccc in token_list:
        tmp_list.append(token_list.get(ccc))

    bigram = bigrams(tmp_list)
    bigram = [bigram[cig] for cig in tmp_list]

    dictionary = gensim.corpora.Dictionary(bigram)
    # Filtro i termini poco frequenti o troppo frequenti
    dictionary.filter_extremes(no_above=0.35, no_below=2)
    # Funzione che pulisce gli id dei token rimossi
    dictionary.compactify()
    print(dictionary)

    bow_corpus = [dictionary.doc2bow(text) for text in bigram]

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

    # Creo una cartella in cui salverò i dati
    # Se la cartella non esiste la creo
    if not os.path.isdir("Dictionary_Corpus_Bigrams"):
        os.makedirs("Dictionary_Corpus_Bigrams")

    # Cambio la cartella corrente
    os.chdir('Dictionary_Corpus_Bigrams')

    # Memorizzo il corpus, il dizionario e i bigrammi
    with open('train_corpus.pkl', 'wb') as f:
        pickle.dump(bow_corpus, f)
    with open('train_dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    with open('bigram_train.pkl', 'wb') as f:
        pickle.dump(bigram, f)

    # Esco dalla cartella
    os.chdir("../")

    end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    print("DATI SCRITTI IN:", ((end_time - start_time).total_seconds() / 60), "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

    print("Training LDA Model...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamulticore.LdaMulticore(
            corpus=bow_corpus,  # The corpus or the document-term matrix to be passed to the model
            num_topics=20,  # num_topics is the number of topics we want to extract from the corpus
            id2word=dictionary,  # It is the mapping from word indices to words. Each of the words has an index that is present in the dictionary.
            chunksize=100,  # It is the number of documents to be used in each training chunk. The chunksize controls how many documents can be processed at one time in the training algorithm.
            workers=7,  # Num. Processing Cores - 1
            passes=50,
            eval_every=1,  # a flag that allows to process the corpus in chunks
            per_word_topics=True)

        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("LDA addestrato in", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

        start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        # Creo una cartella in cui salverò il modello LDA
        # Se la cartella non esiste la creo
        if not os.path.isdir("Lda_model"):
            os.makedirs("Lda_model")

        # Cambio la cartella corrente
        os.chdir('Lda_model')

        lda_train.save('lda_train.model')

        # Esco dalla cartella
        os.chdir("../")
        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("MODELLO SCRITTO IN:", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    # Per stampare i topic con le parole. For each topic, we will explore the words occuring in that topic and its relative weight.
    for idx, topic in lda_train.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    # Compute Perplexity, [Perplexity is the measure of how well a model predicts a sample.]
    print('\nPerplexity: ', lda_train.log_perplexity(bow_corpus))  # a measure of how good the model is. lower the better.
    end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    print("PERPLEXITY CALCOLATA IN:", ((end_time - start_time).total_seconds() / 60),
          "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    coherence_model_lda = CoherenceModel(model=lda_train, texts=bigram, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print("Coherence:", coherence_lda)
    end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    print("COHERENCE CALCOLATA IN:", ((end_time - start_time).total_seconds() / 60),
          "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

    # Feature engineering involves extracting features and properties from our data.
    # Features are the independent units that are used for predictive analysis to influence the output.
    # feature extraction
    train_vecs = []
    for i in range(len(token_list)):
        # Get the topic distribution for the given document.
        top_topics = lda_train.get_document_topics(bow_corpus[i],
                                                   minimum_probability=0.0)  # The key bit is using minimum_probability=0.0 in line 3. This ensures that we’ll capture the instances where a review is presented with 0% in some topics, and the representation for each review will add up to 100%.
        topic_vec = [top_topics[i][1] for i in range(20)]
        train_vecs.append(topic_vec)

    end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
    print("TRAIN_VECS:", ((end_time - start_time).total_seconds() / 60), "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    X = np.array(train_vecs)
    y = np.array(train_label)

    #
    # ______________________________________________________________________________________
    # X = np.array(data_df)
    # y = np.array(labels)


    # count = 0
    # labels = labels.apply(str)
    # a = labels.value_counts()
    # for i in a:
    #     if i == 1:
    #         count += 1
    #
    # print(count)


    # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    # for train_index, test_index in sss.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]



    # train, test = train_test_split(data_df, test_size=0.3, random_state=4, stratify=labels)



    # test_ratio = 0.3
    #
    # def split_train_test(data: np.ndarray, distribution: list, test_ratio: float):
    #     skf = StratifiedKFold(n_splits=int(test_ratio * 100), random_state=1374, shuffle=True)
    #     return next(skf.split(data, distribution))
    #
    #
    # split_train_test(data_df, labels, test_ratio)


