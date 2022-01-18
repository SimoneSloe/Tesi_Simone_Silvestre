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
from gensim.models import CoherenceModel, TfidfModel
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

        # Prendo il path del file
        file_path = os.path.relpath(file)
        # Controllo se il file ha l'estensione ".gz"
        if file.endswith(extension):
            # Prende i documenti all'interno dell'archivio e li analizza
            with gzip.open(file_path) as unzip_file:
                print("Parsing", file, "...")
                # Analizzo i token all'interno del file corrente
                cig_d = parse_token_file(cig_list, cig_dict, unzip_file)

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
            cig = ""

            # Per ogni elemento nel path splittato
            for element in path_elements:
                # Se l'elemento è un CIG
                if re.match("^[0-9]{7}[A-Z0-9]*$", element):
                    # Memorizzo il CIG
                    cig = element
                    break  # se esamina anche l'ultima parte del path ci possono essere casi in cui i file hanno nel
                    # nome un CPV, e può creare problemi. Quindi interrompo, appena trovo il CPV nel percorso

            # Se il cig fa parte della lista dei cig del dataset (csv)
            if cig in cig_list:
                # Se il cig è già nel dizionario
                if cig in cig_dictionary:
                    for document in conll_lines:
                        cig_tokens = []
                        token_parser(document, cig_tokens)
                        # Se la lista dei token non è vuota (possono succedere casi in cui è vuota perchè la frase
                        # contiene soltanto termini da eliminare)
                        if cig_tokens:
                            cig_dictionary[cig].extend(cig_tokens)
                else:  # Se il cig non è nel dizionario come chiave
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


# Funzione che estra i cpv dal file csv in base al cig
def cpv_extracter_from_csv():
    csv_labels = []
    a = data_df.CIG.values.tolist()
    b = data_df.COD_CPV.values.tolist()
    for key in cig_token_dictionary.keys():
        for index in range(len(data_df)):
            if a[index] == key:
                csv_labels.append(b[index])
    return csv_labels


# Funzione che crea un modello per creare bigrammi
def bigrams(words, bi_min=15):
    # Costruzione bigram model
    bi_gram = gensim.models.Phrases(words, min_count=bi_min)  # La classe phrases consente di raggruppare frasi correlate in un token per il LDA
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bi_gram)
    return bigram_mod


# Funzione che estrae i token dal dataframe
def cig_token_list(df):
    tmp_list = []
    for line in df.TOKEN_CIG:
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


# Funzione che salva i dati di training del topic model per un eventuale futuro utilizzo
def save_data(folder_name, train_big, train_dict, train_corp):
    # Creo una cartella in cui salverò i dati di training

    # Se la cartella non esiste la creo
    if not os.path.isdir("Dictionary_Corpus_Bigrams"):
        os.makedirs("Dictionary_Corpus_Bigrams")

    # Cambio la cartella corrente
    os.chdir('Dictionary_Corpus_Bigrams')

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    os.chdir(folder_name)

    # Memorizzo il corpus, il dizionario e i bigrammi
    with open('train_corpus.pkl', 'wb') as f:
        pickle.dump(train_corp, f)
    with open('train_dictionary.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
    with open('bigram_train.pkl', 'wb') as f:
        pickle.dump(train_big, f)

    # Esco dalla cartella
    os.chdir("../")
    os.chdir("../")


# Funzione che salva il topic model addestrato per un eventuale futuro utilizzo
def save_topic_model(folder_name, topic_model):
    # Creo una cartella in cui salverò il modello LDA

    # Se la cartella non esiste la creo
    if not os.path.isdir("Lda_model"):
        os.makedirs("Lda_model")

    # Cambio la cartella corrente
    os.chdir('Lda_model')

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    os.chdir(folder_name)

    topic_model.save('lda_train.model')

    # Esco dalla cartella
    os.chdir("../")
    os.chdir("../")


# Funzione che permette di addestrare il topic model
def training_topic_model(train_corp, train_dict, folder_name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=train_corp,
            num_topics=20,  # è il numero di topic che vogliamo estrarre dal corpus, ottenuto tramite lo HDP (Hierarchical Dirichlet Process)
            id2word=train_dict,
            chunksize=100,  # definisce quanti documenti vengono processati allo stesso tempo
            workers=7,  # Num. Processing Cores - 1
            passes=50,
            eval_every=1,  # un flag che permette di processare il corpus in chunks
            per_word_topics=True)

        save_topic_model(folder_name, lda_model)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=train_bigram, dictionary=train_dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print("Coherence:", coherence_lda)

        return lda_train


# Funzione che restituisce i feature vectors dal topic model che conterrano la distribuzione dei topic per ogni CIG
def feature_extraction_topic(trained_lda, cig_bigram, corpus):
    vecs = []
    for index in range(len(cig_bigram)):
        # Ottengo la distribuzione dei topic per il dato CIG
        top_topics = trained_lda.get_document_topics(corpus[i], minimum_probability=0.0)  # The key bit is using minimum_probability=0.0 in line 3. This ensures that we’ll capture the instances where a review is presented with 0% in some topics, and the representation for each review will add up to 100%.
        topic_vec = [top_topics[index][1] for index in range(20)]
        vecs.append(topic_vec)
    return vecs


# Funzione che estrae le feature dalla bow
def feature_extraction_bow(train_tokens, test_tokens):
    # Per applicare il countvectorizer su una lista di parole, bisogna disabilitare l'analyzer
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    x_tr = vectorizer.fit_transform(train_tokens).toarray()
    x_te = vectorizer.transform(test_tokens).toarray()
    return x_tr, x_te


# Funzione che ridimensiona le feature (sottraendo la media e dividendo per la deviazione standard)
def feature_scaler(train_feature, test_feature):
    scaler_bow = StandardScaler()
    train_feature_scaled = scaler_bow.fit_transform(train_feature)
    test_feature_scaled = scaler_bow.transform(test_feature)
    return train_feature_scaled, test_feature_scaled


# Funzione che permette di addestrare una Support Vector Machine per la classificazione
def training_svc(train_feature, train_label):
    # Creo un modello per la SVM utilizzando l'implementazione svc
    svc = svm.SVC()  # VEDERE SE RIMANE "svc" SOTTOLINEATO
    # Addestro il modello utilizzando il training set
    svc.fit(train_feature, train_label)
    return svc


# Funzione che permette di effettuare predizioni con la SVM addestrata
def svc_predict(test_feature, svc_clf):
    y_pred = svc_clf.predict(test_feature)
    return y_pred


def svm_classifier(train_tm_feat, test_tm_feat, train_bow_feat, test_bow_feat, train_label):

    # FEATURE VECTOR DA TOPIC MODEL

    # Addestro la SVM sulle feature estratte dal topic model
    trained_svc_tm = training_svc(train_tm_feat, train_label)
    # Il modello prevede la risposta per il test set
    y_pred_svc_tm = svc_predict(test_tm_feat, trained_svc_tm)

    # FEATURE VECTOR DALLA BOW

    # Addestro la SVM sulle feature estratte dalla bow
    trained_svc_bow = training_svc(train_bow_feat, train_label)
    # Il modello prevede la risposta per il test set
    y_pred_svc_bow = svc_predict(test_bow_feat, trained_svc_bow)

    return y_pred_svc_tm, y_pred_svc_bow


# Funzione che permette di addestrare una Random Forest per la classificazione
def training_rf(train_feature, train_label):
    # Creo un modello per RF
    rf = RandomForestClassifier()  # VEDERE SE RIMANE "svc" SOTTOLINEATO
    # Addestro il modello utilizzando il training set
    rf.fit(train_feature, train_label)
    return rf


# Funzione che permette di effettuare predizioni con la RF addestrata
def rf_predict(test_feature, rf_clf):
    y_pred = rf_clf.predict(test_feature)
    return y_pred


# Funziona che crea un classificatore ed effettua delle predizioni
def rf_classifier(train_tm_feat, test_tm_feat, train_bow_feat, test_bow_feat, train_label):

    # FEATURE VECTOR DA TOPIC MODEL

    # Addestro la SVM sulle feature estratte dal topic model
    trained_rf_tm = training_rf(train_tm_feat, train_label)
    # Il modello prevede la risposta per il test set
    y_pred_ranfo_tm = rf_predict(test_tm_feat, trained_rf_tm)

    # FEATURE VECTOR DALLA BOW

    # Addestro la SVM sulle feature estratte dalla bow
    trained_rf_bow = training_rf(train_bow_feat, train_label)
    # Il modello prevede la risposta per il test set
    y_pred_ranfo_bow = rf_predict(test_bow_feat, trained_rf_bow)

    return y_pred_ranfo_tm, y_pred_ranfo_bow


# Funzione che calcola l'accuracy tenendo conto delle divisioni dei CPV
def new_accuracy(test_labels, pred_labels):
    num_samples = len(test_labels)
    matches = 0

    for index in range(len(test_labels)):
        # Se il cpv predetto è della stessa divisione del cpv di test
        if pred_labels[index][:2] == test_labels[index][:2]:
            # Lo considero corretto
            matches += 1

    acc = matches / num_samples

    return acc


# *************************** MAIN ***************************

if __name__ == '__main__':

    # --------------------- PRIMO ESPERIMENTO ---------------------

    cpv_division = ['03000000-0', '09000000-0', '14000000-0', '15000000-0', '16000000-0', '18000000-0', '19000000-0',
                    '22000000-0', '24000000-0', '30000000-0', '31000000-0', '32000000-0', '33000000-0', '34000000-0',
                    '35000000-0', '37000000-0', '38000000-0', '39000000-0', '41000000-0', '42000000-0', '43000000-0',
                    '44000000-0', '45000000-0', '48000000-0', '50000000-0', '51000000-0', '55000000-0', '60000000-0',
                    '63000000-0', '64000000-0', '65000000-0', '66000000-0', '70000000-0', '71000000-0', '72000000-0',
                    '73000000-0', '75000000-0', '76000000-0', '77000000-0', '79000000-0', '80000000-0', '85000000-0',
                    '90000000-0', '92000000-0', '98000000-0']

    # Leggo il csv contenente i dati e creo un dataframe
    data_df = pd.read_csv('Dataset/gare-empulia-cpv.csv')
    # Rimuovo le colonne che non servono
    data_df = data_df.drop(columns=['DESCRIZIONE_CPV', 'OGGETTO_LOTTO'])

    # Creo una lista dei CIG
    cigs_list = data_df.CIG.values.tolist()

    # Assegno a una variabile il percorso passato dal terminale
    data_folder = sys.argv[1]  # (1) "D:/Marilisa/Tesi/Dataset/process_bandi_cpv/udpipe",
    # (2) "D:/PycharmProject/pythonproject/provaset"

    # Creo un dict che avrà come chiave il CIG, e come valore una lista che conterrà i token rispettivi
    cig_token_dictionary = create_token_list(cigs_list, data_folder)

    print("\nESPERIMENTO CON DIVISIONE SEMPLICE DEL DATASET")

    # Estraggo dal dict i cig, e i rispettivi token e ne creo delle Series per un nuovo dataframe
    s1 = pd.Series(cig_token_dictionary.keys(), name='CIG')
    s2 = pd.Series(list(cig_token_dictionary.values()), name='TOKEN_CIG')

    # Dal dataframe estraggo il CPV corrispondente al cig e ne creo un altra Series
    df_labels = cpv_extracter_from_csv()

    s3 = pd.Series(df_labels, name='CPV')

    # Lista che conterrà i cpv trasformati nella rispettiva divisione
    transformed_df_labels = []

    for i in range(len(df_labels)):
        for j in range(len(cpv_division)):
            if df_labels[i][:2] == cpv_division[j][:2]:
                transformed_df_labels.append(cpv_division[j])
                break

    s3_transformed = pd.Series(transformed_df_labels, name='CPV_T')

    # Creo un nuovo dataset con cig che hanno del testo, con i rispettivi token e con il CPV associato
    new_df = pd.concat([s1, s2, s3, s3_transformed], axis=1)
    new_df = new_df.dropna()

    # Divisione semplice del nuovo dataframe in train e test
    train, test = train_test_split(new_df, test_size=0.2)

    # Dal train e dal test estraggo i token per ogni cig, i bigrammi, il dizionario e la bow
    train_bigram, train_dictionary, train_corpus = create_dictionary_and_corpus(train)
    test_bigram, test_dictionary, test_corpus = create_dictionary_and_corpus(test)

    # Salvo i dati di training del topic model
    save_data("1_Divisione_semplice_dataset", train_bigram, train_dictionary, train_corpus)

    # Addestro il topic model
    print("\n Training LDA Model...")
    lda_train = training_topic_model(train_corpus, train_dictionary, "1_Divisione_semplice_dataset")

    print("\nEstrazione feature dal topic model..")
    # Applico il topic model addestrato al train e test set, estraendono i feature vectors che conterrano
    # la distribuzione dei topic per ogni CIG
    train_vecs = feature_extraction_topic(lda_train, train_bigram, train_corpus)
    test_vecs = feature_extraction_topic(lda_train, test_bigram, test_corpus)

    X_train_tm = np.array(train_vecs)
    X_test_tm = np.array(test_vecs)

    y_train = np.array(train.CPV.values.tolist())
    y_test = np.array(test.CPV.values.tolist())

    y_train_tr = np.array(train.CPV_T.values.tolist())
    y_test_tr = np.array(test.CPV_T.values.tolist())

    print("Estrazione feature dalla bow...\n")
    # Estraggo le feature anche dalla bow
    X_train_bow, X_test_bow = feature_extraction_bow(train.TOKEN_CIG, test.TOKEN_CIG)  # (train_bigram, test_bigram)

    print("Esperimento classificazione con CPV trasformate nella rispettiva divisione\n")

    # SUPPORT VECTOR MACHINE

    y_pred_svm_tm, y_pred_svm_bow = svm_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train_tr)

    accuracy_tm = metrics.accuracy_score(y_test_tr, y_pred_svm_tm)
    print("Support vector machine accuracy (topic model):", accuracy_tm)

    accuracy_bow = metrics.accuracy_score(y_test_tr, y_pred_svm_bow)
    print("Support vector machine accuracy (bow):", accuracy_bow)

    # RANDOM FOREST

    y_pred_rf_tm, y_pred_rf_bow = rf_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train_tr)

    accuracy_tm = metrics.accuracy_score(y_test_tr, y_pred_rf_tm)
    print("Random forest (topic model):", accuracy_tm)

    accuracy_bow = metrics.accuracy_score(y_test_tr, y_pred_rf_bow)
    print("Random forest accuracy (bow):", accuracy_bow)

    print("\n Esperimento classificazione con modifica alla valutazione dell'accuracy\n")

    # SUPPORT VECTOR MACHINE

    y_pred_svm_tm, y_pred_svm_bow = svm_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train)

    accuracy_tm = new_accuracy(y_test, y_pred_svm_tm)
    print("Support vector machine accuracy (topic model):", accuracy_tm)

    accuracy_bow = new_accuracy(y_test, y_pred_svm_bow)
    print("Support vector machine accuracy (bow):", accuracy_bow)

    # RANDOM FOREST

    y_pred_rf_tm, y_pred_rf_bow = rf_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train)

    accuracy_tm = new_accuracy(y_test, y_pred_rf_tm)
    print("Random forest (topic model):", accuracy_tm)

    accuracy_bow = new_accuracy(y_test, y_pred_rf_bow)
    print("Random forest accuracy (bow):", accuracy_bow)

    # --------------------- SECONDO ESPERIMENTO ---------------------

    print("\n \n ESPERIMENTO CON DIVISIONE DEL DATASET MANTENENDO LA STESSA DISTRIBUZIONE DELLE LABEL")

    # Vengono contate le occorrenze dei CPV, sui CIG che contengono il testo
    cpv_occurences = new_df.CPV.value_counts()

    # Seleziono i CPV che hanno un occorrenza < 2
    to_remove = cpv_occurences[cpv_occurences < 2].index

    # Mantengo nel dataframe i CIG che hanno più di una occorrenza
    df_without_one = new_df[~new_df.CPV.isin(to_remove)]

    label_list = df_without_one.CPV

    # Divido il dataframe mantenendo la stessa distrubuzione delle label nel train e test set
    train, test = train_test_split(df_without_one, test_size=0.3, stratify=label_list)

    # Dal train e dal test estraggo i token per ogni cig, i bigrammi, il dizionario e la bow
    train_bigram, train_dictionary, train_corpus = create_dictionary_and_corpus(train)
    test_bigram, test_dictionary, test_corpus = create_dictionary_and_corpus(test)

    # Salvo i dati di training del topic model
    save_data("2_Divisione_mantenendo_distribuzione", train_bigram, train_dictionary, train_corpus)

    # Addestro il topic model
    print("\n Training LDA Model...")
    lda_train = training_topic_model(train_corpus, train_dictionary, "2_Divisione_mantenendo_distribuzione")

    print("\nEstrazione feature dal topic model..")
    # Applico il topic model addestrato al train e test set, estraendono i feature vectors che conterrano
    # la distribuzione dei topic per ogni CIG
    train_vecs = feature_extraction_topic(lda_train, train_bigram, train_corpus)
    test_vecs = feature_extraction_topic(lda_train, test_bigram, test_corpus)

    X_train_tm = np.array(train_vecs)
    X_test_tm = np.array(test_vecs)

    y_train = np.array(train.CPV.values.tolist())
    y_test = np.array(test.CPV.values.tolist())

    y_train_tr = np.array(train.CPV_T.values.tolist())
    y_test_tr = np.array(test.CPV_T.values.tolist())

    print("Estrazione feature dalla bow...\n")
    # Estraggo le feature anche dalla bow
    X_train_bow, X_test_bow = feature_extraction_bow(train.TOKEN_CIG, test.TOKEN_CIG)  # (train_bigram, test_bigram)

    # Scalo le feature per entrambi gli esperimenti
    # X_train, X_test = feature_scaler(X_train, X_test)
    # X_train_bow, X_test_bow = feature_scaler(X_train_bow, X_test_bow)

    print(" Esperimento classificazione con CPV trasformate nella rispettiva divisione\n")

    # SUPPORT VECTOR MACHINE

    y_pred_svm_tm, y_pred_svm_bow = svm_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train_tr)

    accuracy_tm = metrics.accuracy_score(y_test_tr, y_pred_svm_tm)
    print("Support vector machine accuracy (topic model):", accuracy_tm)

    accuracy_bow = metrics.accuracy_score(y_test_tr, y_pred_svm_bow)
    print("Support vector machine accuracy (bow):", accuracy_bow)

    # RANDOM FOREST

    y_pred_rf_tm, y_pred_rf_bow = rf_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train_tr)

    accuracy_tm = metrics.accuracy_score(y_test_tr, y_pred_rf_tm)
    print("Random forest (topic model):", accuracy_tm)

    accuracy_bow = metrics.accuracy_score(y_test_tr, y_pred_rf_bow)
    print("Random forest accuracy (bow):", accuracy_bow)

    print("\n Esperimento classificazione con modifica alla valutazione dell'accuracy\n")

    # SUPPORT VECTOR MACHINE

    y_pred_svm_tm, y_pred_svm_bow = svm_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train)

    accuracy_tm = new_accuracy(y_test, y_pred_svm_tm)
    print("Support vector machine accuracy (topic model):", accuracy_tm)

    accuracy_bow = new_accuracy(y_test, y_pred_svm_bow)
    print("Support vector machine accuracy (bow):", accuracy_bow)

    # RANDOM FOREST

    y_pred_rf_tm, y_pred_rf_bow = rf_classifier(X_train_tm, X_test_tm, X_train_bow, X_test_bow, y_train)

    accuracy_tm = new_accuracy(y_test, y_pred_rf_tm)
    print("Random forest (topic model):", accuracy_tm)

    accuracy_bow = new_accuracy(y_test, y_pred_rf_bow)
    print("Random forest accuracy (bow):", accuracy_bow)
