import sys
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
from sklearn import svm
from sklearn import metrics
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit

# *************************** FUNZIONI ***************************


# Funzione che analizza i token file per file, e restituisce un dict contenente i cig e i token ad essi associati
def create_token_list(cig_list, folder_path):
    cig_dict = {}
    # Cambio la cartella corrente con quella passata alla funzione
    os.chdir(folder_path)
    # Creo una variabile che contiene l'estensione ".gz"
    extension = ".gz"
    # Ottengo una lista con i file contenuti in una cartella
    folder_files = os.listdir(folder_path)
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
                # Analizzo i token all'interno del file corrente
                cig_d = parse_token_file(cig_list, cig_dict, unzip_file)

                end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
                print("File:", file, "parsed in", ((end_time - start_time).total_seconds() / 60),
                      "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    return cig_d


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


# Funzione che carica il file json e un dict con chiave il CIG e valore la token list per quel CIG
def parse_token_file(cig_list, cig_dictionary, unzipped_file):
    # Carico il dataframe in chunks, per evitare che la memoria si sovraccarichi
    chunks = pd.read_json(unzipped_file, lines=True, chunksize=10000)
    # Per ogni chunk
    for c in chunks:
        # Raggruppo gli elementi in un chunk per 'title' che contiene il path di un documento (in cui è presente il CIG)
        c = c.groupby('title')
        # Per ogni gruppo nel chunk
        for c_group in c:
            # Prendo il path dal gruppo
            path = c_group[0]
            # Prendo l'output in formato CoNLL-U all'interno del gruppo
            conll_lines = c_group[1]['conll'].values.tolist()
            # Normalizzo il path per os
            norm_path = os.path.normpath(path)
            # Splitto il path
            path_elements = norm_path.split(os.sep)

            # cig = path_elements[7]

            # Per ogni elemento nel path splittato
            for element in path_elements:
                # Se l'elemento è un CIG
                if re.match("^[0-9]{7}[A-Z0-9]*$", element):
                    # Memorizzo il CIG
                    cig = element
            # Se il cig fa parte della lista dei cig del dataset (csv)
            if cig in cig_list:
                # Se il cig è già nel dizionario
                if cig in cig_dictionary:
                    for document in conll_lines:
                        cig_tokens = []
                        token_parser(document, cig_tokens)
                        # Se la lista dei token non è vuota (possono succedere casi in cui è vuota perchè la frase contiene soltanto termini da eliminare)
                        if cig_tokens:
                            cig_dictionary[cig].extend(cig_tokens)
                else: # Se il cig non è nel dizionario come chiave
                    # aggiungo una nuova chiave alla dict, che sarebbe il cig, con valore una lista vuota
                    cig_dictionary[cig] = []
                    # Per ogni documento nel formato conll
                    for document in conll_lines:
                        cig_tokens = []
                        # Analizzo i token del documento corrente
                        token_parser(document, cig_tokens)
                        # Se restituisce una lista di token non vuota
                        if cig_tokens:
                            # aggiungo i token del documento alla chiave corrispondente
                            cig_dictionary[cig].extend(cig_tokens)

    return cig_dictionary


# Funzione che crea un modello per creare bigrammi
def bigrams(words, bi_min=15, tri_min=10):
    # Costruzione bigram model
    bi_gram = gensim.models.Phrases(words, min_count=bi_min)  # La classe phrases consente di raggruppare frasi correlate in un token per il LDA
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bi_gram)
    return bigram_mod


# Funzione che estrae i token dal dataframe
def cig_token_list(df):
    tmp_list = []
    for line in df['TOKEN_CIG']:
        tmp_list.append(line)
    return tmp_list


# Funzione che dato un dataframe restituisce un lista di bigrammi, un dizionario e la bow
def create_dictionary_and_corpus(df):
    # Estraggo una lista di token per ogni CIG dal dataframe
    cigs_tokens = cig_token_list(df)

    # Creo un modello per i bigrammi
    bigram = bigrams(cigs_tokens)
    # Formo i bigrammi per ogni CIG
    bigram = [bigram[cig] for cig in cigs_tokens]

    # Creo il dizionario dei token, in cui viene assegnato un id ai termini
    dictionary = gensim.corpora.Dictionary(bigram)
    # Filtro i termini poco frequenti o troppo frequenti
    dictionary.filter_extremes(no_above=0.35, no_below=2)
    # Funzione che pulisce gli id dei token rimossi
    dictionary.compactify()

    # Creo la bow
    bow_corpus = [dictionary.doc2bow(text) for text in bigram]

    return bigram, dictionary, bow_corpus


# Funzione che restituisce i feature vectors che conterrano la distribuzione dei topic per ogni CIG
def extract_feature_vecs(cig_bigram, corpus):
    vecs = []
    for i in range(len(cig_bigram)):
        # Ottengo la distribuzione dei topic per il dato CIG
        top_topics = lda_train.get_document_topics(corpus[i], minimum_probability=0.0)  # The key bit is using minimum_probability=0.0 in line 3. This ensures that we’ll capture the instances where a review is presented with 0% in some topics, and the representation for each review will add up to 100%.
        topic_vec = [top_topics[i][1] for i in range(20)]
        vecs.append(topic_vec)
    return vecs


# *************************** MAIN ***************************

if __name__ == '__main__':
    # --------------------- PRIMO ESPERIMENTO ---------------------
    print("ESPERIMENTO CON DIVISIONE SEMPLICE DEL DATASET")

    # Leggo il csv contenente i dati e creo un dataframe
    data_df = pd.read_csv('Dataset/gare-empulia-cpv.csv')
    labels = data_df.COD_CPV
    # Rimuovo le colonne che non servono
    data_df = data_df.drop(columns=['DESCRIZIONE_CPV', 'OGGETTO_LOTTO'])

    # Creo una lista dei CIG
    cigs_list = data_df.CIG.values.tolist()
    print("______CIG_TRAIN_LIST:", len(cigs_list))

    data_folder = "D:/Marilisa/Tesi/Dataset/process_bandi_cpv/udpipe"  # "D:/PycharmProject/pythonproject/provaset"

    # Creo un dict che avrà come chiave il CIG, e come valore una lista che conterrà i token rispettivi
    cig_token_dictionary = create_token_list(cigs_list, data_folder)

    # Estraggo dal dict i cig, e i rispettivi token e ne creo delle Series per un nuovo dataframe
    s1 = pd.Series(cig_token_dictionary.keys(), name='CIG')
    s2 = pd.Series(list(cig_token_dictionary.values()), name='TOKEN_CIG')

    print("______NUM_CIG:", len(cig_token_dictionary))

    # Dal dataframe estraggo il CPV corrispondente al cig e ne creo un altra Series
    df_labels = []
    a = data_df.CIG.values.tolist()
    b = data_df.COD_CPV.values.tolist()
    for key in cig_token_dictionary.keys():
        for i in range(len(data_df)):
            if a[i] == key:
                df_labels.append(b[i])

    s3 = pd.Series(df_labels, name='CPV')

    # Creo un nuovo dataset con cig che hanno del testo, con i rispettivi token e con il CPV associato
    new_df = pd.concat([s1, s2, s3], axis=1)
    new_df = new_df.dropna()

    # Divisione semplice del nuovo dataframe in train e test
    train, test = train_test_split(new_df, test_size=0.2)

    # Dal train e dal test estraggo i token per ogni cig, i bigrammi, il dizionario e la bow
    train_bigram, train_dictionary, train_corpus = create_dictionary_and_corpus(train)
    test_bigram, test_dictionary, test_corpus = create_dictionary_and_corpus(test)

    # ************************ Creo una cartella in cui salverò i dati di training

    # Se la cartella non esiste la creo
    if not os.path.isdir("Dictionary_Corpus_Bigrams"):
        os.makedirs("Dictionary_Corpus_Bigrams")

    # Cambio la cartella corrente
    os.chdir('Dictionary_Corpus_Bigrams')

    # Memorizzo il corpus, il dizionario e i bigrammi
    with open('train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, f)
    with open('train_dictionary.pkl', 'wb') as f:
        pickle.dump(train_dictionary, f)
    with open('bigram_train.pkl', 'wb') as f:
        pickle.dump(train_bigram, f)

    # Esco dalla cartella
    os.chdir("../")

    # ******************************************************

    # TESTARE HDP, per vedere se i topic sono sempre 20

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

    print("Training LDA Model...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamulticore.LdaMulticore(
            corpus=train_corpus,
            num_topics=20,  # è il numero di topic che vogliamo estrarre dal corpus, ottenuto tramite lo HDP (Hierarchical Dirichlet Process)
            id2word=train_dictionary,
            chunksize=100,  # definisce quanti documenti vengono processati allo stesso tempo
            workers=7,  # Num. Processing Cores - 1
            passes=50,
            eval_every=1,  # un flag che permette di processare il corpus in chunks
            per_word_topics=True)

        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("LDA addestrato in", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

        # Creo una cartella in cui salverò il modello LDA
        # Se la cartella non esiste la creo
        if not os.path.isdir("Lda_model"):
            os.makedirs("Lda_model")

        # Cambio la cartella corrente
        os.chdir('Lda_model')

        lda_train.save('lda_train.model')

        # Esco dalla cartella
        os.chdir("../")

        # Per ogni topic, mostrerà le parole che occorrono in quel topic con il relativo peso
        # for idx, topic in lda_train.print_topics(-1):
            # print('Topic: {} \nWords: {}'.format(idx, topic))

        start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        coherence_model_lda = CoherenceModel(model=lda_train, texts=train_bigram, dictionary=train_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print("Coherence:", coherence_lda)
        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("COHERENCE CALCOLATA IN:", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    # Applico il topic model addestrato al train e test set, estraendono i feature vectors che conterrano la distribuzione di probabilità dei topic per ogni CIG
    train_vecs = extract_feature_vecs(train_bigram, train_corpus)
    print(train_vecs)
    test_vecs = extract_feature_vecs(test_bigram, test_corpus)
    print(test_vecs)

    X_train = np.array(train_vecs)
    y_train = np.array(train.CPV.values.tolist())

    X_test = np.array(test_vecs)
    y_test = np.array(test.CPV.values.tolist())

    # SUPPORT VECTOR MACHINE

    # TRAIN

    # Creo un modello per la SVM utilizzando l'implementazione svc
    svc = svm.SVC(probability=True)  # Probability, permette di calcolare le probabilità. Cosa che svm non fa normalmente
    # Addestro il modello utilizzando il training set
    svc.fit(train_vecs, y_train)

    # TEST

    # Il modello prevede la risposta per il test set
    y_pred_svc = svc.predict(test_vecs)

    # Calcola la distr. di probabilità dei CPV per ogni CIG del test set
    # prova = svc.predict_proba(test_vecs)

    # VALUTAZIONE

    # Model Accuracy: how often is the classifier correct?
    print("Support vector machine accuracy:", metrics.accuracy_score(y_test, y_pred_svc))

    # Model Precision: what percentage of positive tuples are labeled as such?
    # print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))

    # Model Recall: what percentage of positive tuples are labelled as such?
    # print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))

    # RANDOM FOREST

    # TRAIN

    # Creo un modello per RF
    rf = RandomForestClassifier(n_estimators=100)

    # Addestro il classificatore con i dati di train
    rf.fit(X_train, y_train)

    # TEST

    y_pred_rf = rf.predict(X_test)

    # VALUTAZIONE

    print("Random forest accuracy:", metrics.accuracy_score(y_test, y_pred_rf))















    #
    # ______________________________________________________________________________________

    # --------------------- SECONDO ESPERIMENTO ---------------------

    print("ESPERIMENTO CON DIVISIONE DEL DATASET MANTENENDO LA STESSA DISTRIBUZIONE DELLE LABEL")

    # Vengono contate le occorrenze dei CPV, sui CIG che contengono il testo
    cpv_occurences = new_df.CPV.value_counts()

    # Seleziono i CPV che hanno un occorrenza < 2
    to_remove = cpv_occurences[cpv_occurences < 2].index

    # Mantengo nel dataframe i CIG che hanno più di una occorrenza
    df_without_one = new_df[~new_df.CPV.isin(to_remove)]

    # Divido il dataframe mantenendo la stessa distrubuzione delle label nel train e test set
    train, test = train_test_split(df_without_one, test_size=0.3, stratify=df_without_one.CPV)

    # Dal train e dal test estraggo i token per ogni cig, i bigrammi, il dizionario e la bow
    train_bigram, train_dictionary, train_corpus = create_dictionary_and_corpus(train)
    test_bigram, test_dictionary, test_corpus = create_dictionary_and_corpus(test)

    # ************************ Creo una cartella in cui salverò i dati di training

    # Se la cartella non esiste la creo
    if not os.path.isdir("Dictionary_Corpus_Bigrams"):
        os.makedirs("Dictionary_Corpus_Bigrams")

    # Cambio la cartella corrente
    os.chdir('Dictionary_Corpus_Bigrams')

    # Memorizzo il corpus, il dizionario e i bigrammi
    with open('train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corpus, f)
    with open('train_dictionary.pkl', 'wb') as f:
        pickle.dump(train_dictionary, f)
    with open('bigram_train.pkl', 'wb') as f:
        pickle.dump(train_bigram, f)

    # Esco dalla cartella
    os.chdir("../")

    # ******************************************************

    # TESTARE HDP, per vedere se i topic sono sempre 20

    start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE

    print("Training LDA Model...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamulticore.LdaMulticore(
            corpus=train_corpus,
            num_topics=20,
            # è il numero di topic che vogliamo estrarre dal corpus, ottenuto tramite lo HDP (Hierarchical Dirichlet Process)
            id2word=train_dictionary,
            chunksize=100,  # definisce quanti documenti vengono processati allo stesso tempo
            workers=7,  # Num. Processing Cores - 1
            passes=50,
            eval_every=1,  # un flag che permette di processare il corpus in chunks
            per_word_topics=True)

        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("LDA addestrato in", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

        # Creo una cartella in cui salverò il modello LDA
        # Se la cartella non esiste la creo
        if not os.path.isdir("Lda_model"):
            os.makedirs("Lda_model")

        # Cambio la cartella corrente
        os.chdir('Lda_model')

        lda_train.save('lda_train.model')

        # Esco dalla cartella
        os.chdir("../")

        # Per ogni topic, mostrerà le parole che occorrono in quel topic con il relativo peso
        # for idx, topic in lda_train.print_topics(-1):
        # print('Topic: {} \nWords: {}'.format(idx, topic))

        start_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        coherence_model_lda = CoherenceModel(model=lda_train, texts=train_bigram, dictionary=train_dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print("Coherence:", coherence_lda)
        end_time = datetime.datetime.now()  # DA RIMUOVERE, SOLO PER PROVE
        print("COHERENCE CALCOLATA IN:", ((end_time - start_time).total_seconds() / 60),
              "minutes.")  # DA RIMUOVERE, SOLO PER PROVE

    # Applico il topic model addestrato al train e test set, estraendono i feature vectors che conterrano la distribuzione di probabilità dei topic per ogni CIG
    train_vecs = extract_feature_vecs(train_bigram, train_corpus)
    print(train_vecs)
    test_vecs = extract_feature_vecs(test_bigram, test_corpus)
    print(test_vecs)

    X_train = np.array(train_vecs)
    y_train = np.array(train.CPV.values.tolist())

    X_test = np.array(test_vecs)
    y_test = np.array(test.CPV.values.tolist())

    # SUPPORT VECTOR MACHINE

    # TRAIN

    # Creo un modello per la SVM utilizzando l'implementazione svc
    svc = svm.SVC(
        probability=True)  # Probability, permette di calcolare le probabilità. Cosa che svm non fa normalmente
    # Addestro il modello utilizzando il training set
    svc.fit(train_vecs, y_train)

    # TEST

    # Il modello prevede la risposta per il test set
    y_pred_svc = svc.predict(test_vecs)

    # Calcola la distr. di probabilità dei CPV per ogni CIG del test set
    # prova = svc.predict_proba(test_vecs)

    # VALUTAZIONE

    # Model Accuracy: how often is the classifier correct?
    print("Support vector machine accuracy:", metrics.accuracy_score(y_test, y_pred_svc))

    # Model Precision: what percentage of positive tuples are labeled as such?
    # print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))

    # Model Recall: what percentage of positive tuples are labelled as such?
    # print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))

    # RANDOM FOREST

    # TRAIN

    # Creo un modello per RF
    rf = RandomForestClassifier(n_estimators=100)

    # Addestro il classificatore con i dati di train
    rf.fit(X_train, y_train)

    # TEST

    y_pred_rf = rf.predict(X_test)

    # VALUTAZIONE

    print("Random forest accuracy:", metrics.accuracy_score(y_test, y_pred_rf))

