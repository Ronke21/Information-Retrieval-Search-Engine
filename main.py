import math
import os
import pickle
import re
import warnings

import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from googletrans import Translator
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

from yap_api_lemmas import heb_yap_process

nltk.download('punkt', quiet=True)  #
nltk.download('stopwords', quiet=True)

HEB_FILE_SYSTEM = r'\hebrew_files_medium'
ENG_FILE_SYSTEM = r'\english_files_medium'
# HEB_FILE_SYSTEM = r'\hebrew_files_short'
# ENG_FILE_SYSTEM = r'\english_files_short'

EXISTING_FILE_SYSTEM = "1"
UPDATE_FILE_SYSTEM = "2"

HEB_STOP_WORDS = [
    'ה',
    'ל',
    'ב',
    'כ',
    'מ',
    'אני',
    'את',
    'אתה',
    'אנחנו',
    'אתן',
    'אתם',
    'הם',
    'הן',
    'היא',
    'הוא',
    'שלי',
    'שלו',
    'שלך',
    'שלה',
    'שלנו',
    'שלכם',
    'שלכן',
    'שלהם',
    'שלהן',
    'לי',
    'לו',
    'לה',
    'לנו',
    'לכם',
    'לכן',
    'להם',
    'להן',
    'אותה',
    'אותו',
    'זה',
    'זאת',
    'אלה',
    'אלו',
    'תחת',
    'מתחת',
    'מעל',
    'בין',
    'עם',
    'עד',
    'נגר',
    'על',
    'אל',
    'מול',
    'של',
    'אצל',
    'כמו',
    'אחר',
    'אותו',
    'בלי',
    'לפני',
    'אחרי',
    'מאחורי',
    'עלי',
    'עליו',
    'עליה',
    'עליך',
    'עלינו',
    'עליכם',
    'לעיכן',
    'עליהם',
    'עליהן',
    'כל',
    'כולם',
    'כולן',
    'כך',
    'ככה',
    'כזה',
    'זה',
    'זות',
    'אותי',
    'אותה',
    'אותם',
    'אותך',
    'אותו',
    'אותן',
    'אותנו',
    'ואת',
    'את',
    'אתכם',
    'אתכן',
    'איתי',
    'איתו',
    'איתך',
    'איתה',
    'איתם',
    'איתן',
    'איתנו',
    'איתכם',
    'איתכן',
    'יהיה',
    'תהיה',
    'היתי',
    'היתה',
    'היה',
    'להיות',
    'עצמי',
    'עצמו',
    'עצמה',
    'עצמם',
    'עצמן',
    'עצמנו',
    'עצמהם',
    'עצמהן',
    'מי',
    'מה',
    'איפה',
    'היכן',
    'במקום שבו',
    'אם',
    'לאן',
    'למקום שבו',
    'מקום בו',
    'איזה',
    'מהיכן',
    'איך',
    'כיצד',
    'באיזו מידה',
    'מתי',
    'בשעה ש',
    'כאשר',
    'כש',
    'למרות',
    'לפני',
    'אחרי',
    'מאיזו סיבה',
    'הסיבה שבגללה',
    'למה',
    'מדוע',
    'לאיזו תכלית',
    'כי',
    'יש',
    'אין',
    'אך',
    'מנין',
    'מאין',
    'מאיפה',
    'יכל',
    'יכלה',
    'יכלו',
    'יכול',
    'יכולה',
    'יכולים',
    'יכולות',
    'יוכלו',
    'יוכל',
    'מסוגל',
    'לא',
    'רק',
    'אולי',
    'אין',
    'לאו',
    'אי',
    'כלל',
    'נגד',
    'אם',
    'עם',
    'אל',
    'אלה',
    'אלו',
    'אף',
    'על',
    'מעל',
    'מתחת',
    'מצד',
    'בשביל',
    'לבין',
    'באמצע',
    'בתוך',
    'דרך',
    'מבעד',
    'באמצעות',
    'למעלה',
    'למטה',
    'מחוץ',
    'מן',
    'לעבר',
    'מכאן',
    'כאן',
    'הנה',
    'הרי',
    'פה',
    'שם',
    'אך',
    'ברם',
    'שוב',
    'אבל',
    'מבלי',
    'בלי',
    'מלבד',
    'רק',
    'בגלל',
    'מכיוון',
    'עד',
    'אשר',
    'ואילו',
    'למרות',
    'אס',
    'כמו',
    'כפי',
    'אז',
    'אחרי',
    'כן',
    'לכן',
    'לפיכך',
    'מאד',
    'עז',
    'מעט',
    'מעטים',
    'במידה',
    'שוב',
    'יותר',
    'מדי',
    'גם',
    'כן',
    'נו',
    'אחר',
    'אחרת',
    'אחרים',
    'אחרות',
    'אשר',
    'או']
ENG_STOP_WORDS = stopwords.words('english')

DOC_NUMBER_FOR_RANKING = 30
LANGUAGE_RECOGNITION_LIMIT = 0.7

lemmatizer = WordNetLemmatizer()
translator = Translator()


class DocSimilarityObject:
    def __init__(self, doc):
        self.doc = doc
        self.bow_similarity_score = 0
        self.jaccard_similarity_score = 0
        self.doc2vec_similarity_score = 0
        self.tfidf_similarity_score = 0
        self.total_similarity_score = 0


class Document:
    def __init__(self, name, language, text):
        self.name = name
        self.language = language
        self.text = text
        self.processed_text = self.process_text()
        self.tokenized_text = self.tokenize_text()
        self.lemmatized_text = self.lemmatize_text()
        self.bow = []
        self.doc2vec_vector = []

    # process the text
    def process_text(self):
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', self.text)

        if self.language == 'HEB':
            # remove not hebrew letters
            my_word = re.sub(r'[^אבגדהוזחטיכלמנסעפצקרשתםןץףך]', ' ', text)
            my_word2 = re.sub(r'(\s)([אבגדהוזחטיכלמנסעפצקרשתםןץףך])(\s)', ' ', my_word)

        else:
            # remove all non-english characters
            my_word = re.sub(r'[^a-zA-Z]', ' ', text)
            my_word2 = re.sub(r'(\s)([a-zA-Z])(\s)', ' ', my_word)

        self.processed_text = my_word2
        return self.processed_text

    # tokenize the text
    def tokenize_text(self):
        processed_text = self.processed_text
        if self.language == 'HEB':
            # remove stop words
            self.tokenized_text = [word for word in processed_text.split() if word not in HEB_STOP_WORDS]
        else:
            # remove stop words
            self.tokenized_text = [word for word in processed_text.split() if word not in ENG_STOP_WORDS]
        return self.tokenized_text

    def lemmatize_text(self):
        if self.language == 'ENG':
            self.lemmatized_text = [lemmatizer.lemmatize(word) for word in self.tokenized_text]
        elif self.language == 'HEB':
            lemmas = heb_yap_process(' '.join(self.tokenized_text))
            lemmatized_text = lemmas.split()
            self.lemmatized_text = [word for word in lemmatized_text if word not in HEB_STOP_WORDS]

        return self.lemmatized_text


def train_doc2vec_model(file_system):
    print('Training doc2vec model...')
    # get all texts from the file system
    clean_texts = []
    for doc in file_system:
        clean_texts.append(doc.lemmatized_text)

    tagged_texts = []
    # tokenize clean texts and convert into gensim formatted tagged data
    for index, text in enumerate(clean_texts):
        tagged_texts.append(TaggedDocument(words=text, tags=[index]))

    # create doc2vec model with training data and validation data

    # create doc2vec model
    model = Doc2Vec(vector_size=300,  # size of vector for each document
                    min_count=1,  # Ignore words with total frequency lower than this. (rare words can harm the model)
                    epochs=20,  # Number of iterations (epochs) over the corpus
                    alpha=0.05,  # The initial learning rate.
                    dm=1,  # training algo. 1 for ‘distributed memory’ (PV-DM). else, distributed bag of words (PV-DBOW)
                    workers=8
                    # Use these many worker threads to train the model (=faster training with multicore machines).
                    )

    # Build vocabulary from a sequence of documents
    model.build_vocab(tagged_texts)
    # print('Model word count: ', model.corpus_count)

    # train model - update the model’s neural weights.
    model.train(tagged_texts, total_examples=model.corpus_count, epochs=model.epochs)

    # add to every document the doc2vec vector
    for doc in file_system:
        doc.doc2vec_vector = model.infer_vector(doc.tokenized_text)

    return model, file_system


def train_bow(file_system):
    print('Training BOW model...')
    # create vocabulary set from all lemmatized text in all documents
    all_words = []
    for doc in file_system:
        all_words += doc.lemmatized_text
    vocabulary_set = list(set(all_words))

    # create a bag of words for each document and do it with parallel processing (multiprocessing)
    # create a dictionary for each document with the words as keys and the number of times the word appears as value
    for doc in file_system:
        doc.bow = dict([(word, count) for word, count in zip(vocabulary_set, doc.lemmatized_text)])

    # create a bag of words for each document
    for doc in file_system:
        doc.bow = [0] * len(vocabulary_set)
        for word in doc.lemmatized_text:
            index = vocabulary_set.index(word)
            doc.bow[index] += 1
    return vocabulary_set, file_system


def infer_bow_vector(lemmatized_text, vocabulary_set):
    # create a bag of words for the query document
    bow = [0] * len(vocabulary_set)
    for word in lemmatized_text:
        try:
            # if word in query not in vocabulary set, ignore it (don't add score to bow)
            index = vocabulary_set.index(word)
            bow[index] += 1
        except ValueError:
            pass
    return bow


# function that get path to dir and read all files in it, put texts to list and return
def get_files(path):
    files = os.listdir(path)
    texts = []
    for file in files:
        with open(path + file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


# create function that get all texts and return dataset of tfidf of all the words
def get_tfidf(texts):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    return tfidf_matrix


def get_file_system(language):
    path = os.getcwd()
    if language == "ENG":
        path += ENG_FILE_SYSTEM
    else:
        path += HEB_FILE_SYSTEM
    # ask the user if to use the saved file system or to update it
    while True:
        print('\nDo you want to use the saved file system or to update it?\n'
              '1. Use saved file system\n'
              '2. Update file system\n')
        choice = input('Enter your choice: ')
        if choice == EXISTING_FILE_SYSTEM:
            # if file exist read it, else print error message
            if os.path.exists(path + '\\processed_file_system.pickle'):
                file_system = pickle.load(open(path + '\\processed_file_system.pickle', 'rb'))
                doc2vec_model = pickle.load(open(path + '\\doc2vec_model.pickle', 'rb'))
                bow_model = pickle.load(open(path + '\\bow_model.pickle', 'rb'))
                fs_num = len(file_system)
                dir_num = count_documents_in_dir(path)
                if dir_num > fs_num:
                    print('Amount of documents in the file system: ', fs_num)
                    print('Amount of documents in the dir: ', dir_num)
                    print('The file system is not up to date.')
                    # get input from user if want to use the saved file system or to update it
                    input2 = input('Type 1 to continue with the saved file system, else anything: ')
                    if input2 == '1':
                        break
                    else:
                        continue
                else:
                    break
            else:
                print('File system not found')
        elif choice == UPDATE_FILE_SYSTEM:
            file_system = process_file_system(language)
            doc2vec_model = None
            bow_model = None
            break
        else:
            print('Invalid choice\n')

    return file_system, doc2vec_model, bow_model


def count_documents_in_dir(path):
    # return number of files ending with .txt in the path
    return len([name for name in os.listdir(path) if name.endswith('.txt')])


def save_file_system(file_system, doc2vec_model, bow_model, language):
    path = os.getcwd()
    if language == "ENG":
        path += ENG_FILE_SYSTEM
    else:
        path += HEB_FILE_SYSTEM
    pickle.dump(file_system, open(path + '\\processed_file_system.pickle', 'wb'))
    pickle.dump(doc2vec_model, open(path + '\\doc2vec_model.pickle', 'wb'))
    pickle.dump(bow_model, open(path + '\\bow_model.pickle', 'wb'))


def process_file_system(language):
    path = os.getcwd()
    if language == "ENG":
        path += ENG_FILE_SYSTEM
    else:
        path += HEB_FILE_SYSTEM
    # read all txt files in FILE_SYSTEM and get every one to Document class
    documents_lst = []

    # get all files in the directory and sub directories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    document = Document(file, language, f.read())
                    documents_lst.append(document)
                    if len(documents_lst) % 25 == 0:
                        print(str(len(documents_lst)) + ' files processed as documents')

    return documents_lst


def get_query():
    language = ""
    # get query from user
    while True:
        query = input("\nEnter query: ")
        if query == "":
            print("query can not be empty")
            continue
        if len(query) <= 1:
            print("query can not be one char")
            continue

        # remove all punctuation marks from query
        query = re.sub(r'[^\w\s]', '', query)
        # replace all " " with ""
        query_without_spaces = re.sub(r'\s', '', query)

        query_len = len(query_without_spaces)
        eng_query_len = len(re.findall(r'[a-zA-Z]', query_without_spaces))
        heb_query_len = len(re.findall(r'[אבגדהוזחטיכלמנסעפצקרשתםןץףך]', query_without_spaces))

        # if 75% of the query is english letters, language="ENG"
        if eng_query_len / query_len > LANGUAGE_RECOGNITION_LIMIT:
            language = "ENG"
            break
        # else if 75% of the query is hebrew letters, language="HEB"
        elif heb_query_len / query_len > LANGUAGE_RECOGNITION_LIMIT:
            language = "HEB"
            break
        else:
            print("Please enter a query in english or hebrew")

    return query, language


def get_language():
    # get language from user, only options are 0 for exit, 1 for english or 2 for hebrew, else ask again
    while True:
        num = input("Enter language for file system (1 for english, 2 for hebrew, 0 for exit): ")
        if num == "1":
            language = "ENG"
            break
        elif num == "2":
            language = "HEB"
            break
        elif num == "0":
            return
        else:
            print("Please enter a valid number between 0-2\n\n")
    return language


def calc_jaccard_similarity_score(bow, bow1):
    intersection = (set(bow)).intersection(set(bow1))
    union = (set(bow)).union(set(bow1))
    return len(intersection) / len(union)


def calculate_documents_similarity_list(query_doc, file_system):
    """
    # higher bow_similarity is for better similarity
    # higher jaccard_similarity is for better similarity
    # lower doc2vec_similarity is for better similarity
    # higher tfidf_similarity is for better similarity
    :param query_doc:
    :param file_system:
    :return:
    """
    print('\nCalculating similarities...\n')
    tokenized_corpus = [doc.lemmatized_text for doc in file_system]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_bm25_scores = bm25.get_scores(query_doc.lemmatized_text)

    document_similarity_list = []

    for doc in file_system:
        doc_sim = DocSimilarityObject(doc)

        doc_sim.doc2vec_similarity_score = spatial.distance.cosine(query_doc.doc2vec_vector, doc.doc2vec_vector)

        doc_sim.bow_similarity_score = np.dot(query_doc.bow, doc.bow)

        doc_sim.jaccard_similarity_score = calc_jaccard_similarity_score(query_doc.bow, doc.bow)

        doc_sim.tfidf_similarity_score = doc_bm25_scores[file_system.index(doc)]

        document_similarity_list.append(doc_sim)

    # normalize all similarity scores from 0 to 1
    max_doc2vec_score = max(document_similarity_list, key=lambda x: x.doc2vec_similarity_score).doc2vec_similarity_score
    max_bow_score = max(document_similarity_list, key=lambda x: x.bow_similarity_score).bow_similarity_score
    max_jaccard_score = max(document_similarity_list, key=lambda x: x.jaccard_similarity_score).jaccard_similarity_score
    max_tfidf_score = max(document_similarity_list, key=lambda x: x.tfidf_similarity_score).tfidf_similarity_score
    for doc in document_similarity_list:
        try:
            doc.bow_similarity_score /= max_bow_score
        except Exception:
            if doc.bow_similarity_score == 0:
                doc.bow_similarity_score = 0.0
        try:
            doc.jaccard_similarity_score /= max_jaccard_score
        except Exception:
            if doc.jaccard_similarity_score == 0:
                doc.jaccard_similarity_score = 0.0
        try:
            doc.tfidf_similarity_score /= max_tfidf_score
        except Exception:
            if doc.tfidf_similarity_score == 0:
                doc.tfidf_similarity_score = 0.0
        try:
            doc.doc2vec_similarity_score /= max_doc2vec_score
        except Exception:
            if doc.doc2vec_similarity_score == 0:
                doc.doc2vec_similarity_score = 0.0
        # normalize doc2vec similarity score so the most similar doc is the highest score
        doc.doc2vec_similarity_score = 1 - doc.doc2vec_similarity_score

    # if one of the fields is nan, set it to 0
    for doc in document_similarity_list:
        if math.isnan(doc.doc2vec_similarity_score):
            doc.doc2vec_similarity_score = 0
        if math.isnan(doc.bow_similarity_score):
            doc.bow_similarity_score = 0
        if math.isnan(doc.jaccard_similarity_score):
            doc.jaccard_similarity_score = 0
        if math.isnan(doc.tfidf_similarity_score):
            doc.tfidf_similarity_score = 0

    # calculate total similarity score for each document
    for doc in document_similarity_list:
        doc.total_similarity_score = (doc.doc2vec_similarity_score +
                                      doc.bow_similarity_score +
                                      doc.jaccard_similarity_score +
                                      doc.tfidf_similarity_score) / 4

    return document_similarity_list


def get_synonym_word_eng(word):
    syns = wordnet.synsets(word)
    syn_words = []
    for i in range(len(syns)):
        for j in syns[i].lemma_names():
            syn_words.append(j)
    # return first synonym word that is not the same as the original word if exists, else return original word
    for syn in syn_words:
        if syn != word:
            return syn
    return word


def get_synonym_word_heb(word):
    """
    trans_word = translator.translate(word, src='he', dst='eng')
    syn_trans_word = get_synonym_word_eng(trans_word)
    syn_word = translator.translate(syn_trans_word, src='eng', dst='he')
    return syn_word
    """
    return word


def get_synonym_sentence(language, sentence):
    syn_sentence = []
    if language == "ENG":
        for word in sentence:
            syn_sentence.append(get_synonym_word_eng(word))
    else:
        for word in sentence:
            syn_sentence.append(get_synonym_word_heb(word))
    return syn_sentence


def get_syn_query(language, query_doc):
    try:
        syn_query = get_synonym_sentence(language, query_doc.lemmatized_text)
    except Exception:
        syn_query = query_doc.lemmatized_text
    # return syn_query as one string
    return " ".join(syn_query)


def get_similarity_kind():
    # ask user which similarity to use: 1 for bow, 2 for jaccard, 3 for bm25, 4 for doc2vec, 5 for all
    while True:
        print("\nPlease choose a similarity measure to use:")
        print("1. Bow")
        print("2. Jaccard")
        print("3. BM25")
        print("4. Doc2vec")
        print("5. All")
        num = input("Enter your choice: ")
        if num == "1":
            return "bow"
        elif num == "2":
            return "jaccard"
        elif num == "3":
            return "bm25"
        elif num == "4":
            return "doc2vec"
        elif num == "5":
            return "all"
        else:
            print("Please enter a valid number between 1-5\n\n")


def find_and_print_most_similar_docs(query_doc, file_system, include_query, forbidden_words, measure):
    document_similarity_list = calculate_documents_similarity_list(query_doc, file_system)

    if include_query is False:
        # remove from the list documents that contain the query words
        for doc in document_similarity_list:
            for word in forbidden_words:
                if word in doc.doc.lemmatized_text:
                    document_similarity_list.remove(doc)
                    break  # document is removed - finish inner loop and go to next document

    doc_len = min(DOC_NUMBER_FOR_RANKING, len(document_similarity_list))
    print("\nTotal amount of documents in the system: " + str(len(document_similarity_list)))
    print("Top", doc_len, "documents with the highest total similarity score, ", end='')
    if include_query is True:
        print("including the query words:")
        print("(Query is: " + ' '.join(query_doc.lemmatized_text) + ")")
    else:
        print("excluding the query words:")
        print("(Substitutional query is: " + ' '.join(query_doc.lemmatized_text) + ")")
    print()

    if measure == "all":
        # sort the list by total similarity score
        document_similarity_list.sort(key=lambda x: x.total_similarity_score, reverse=True)
        for i in range(doc_len):
            sim = document_similarity_list[i].total_similarity_score
            print(str(i + 1) + ")\t\tFile", document_similarity_list[i].doc.name, "\t similarity:", round(sim, 5))
    elif measure == "bow":
        # sort the list by bow similarity score
        document_similarity_list.sort(key=lambda x: x.bow_similarity_score, reverse=True)
        for i in range(doc_len):
            sim = document_similarity_list[i].bow_similarity_score
            print(str(i + 1) + ")\t\tFile", document_similarity_list[i].doc.name, "\t similarity:", round(sim, 5))
    elif measure == "jaccard":
        # sort the list by jaccard similarity score
        document_similarity_list.sort(key=lambda x: x.jaccard_similarity_score, reverse=True)
        for i in range(doc_len):
            sim = document_similarity_list[i].jaccard_similarity_score
            print(str(i + 1) + ")\t\tFile", document_similarity_list[i].doc.name, "\t similarity:", round(sim, 5))
    elif measure == "bm25":
        # sort the list by bm25 similarity score
        document_similarity_list.sort(key=lambda x: x.tfidf_similarity_score, reverse=True)
        for i in range(doc_len):
            sim = document_similarity_list[i].tfidf_similarity_score
            print(str(i + 1) + ")\t\tFile", document_similarity_list[i].doc.name, "\t similarity:", round(sim, 5))
    elif measure == "doc2vec":
        # sort the list by doc2vec similarity score
        document_similarity_list.sort(key=lambda x: x.doc2vec_similarity_score, reverse=True)
        for i in range(doc_len):
            sim = document_similarity_list[i].doc2vec_similarity_score
            print(str(i + 1) + ")\t\tFile", document_similarity_list[i].doc.name, "\t similarity:", round(sim, 5))


def run_query():
    # query, language = "what is the 5 matrixדג project january", "ENG"
    # query, language = "delete words", "ENG"
    # query, language = "eBook samuel johnson new york", "ENG"
    # query, language = "מה זה4 המטריקס שלA חנות", "HEB"
    query, language = get_query()
    query_doc = Document("query", language, query)
    file_system, doc2vec_model, bow_model = get_file_system(language)

    if doc2vec_model is None:
        doc2vec_model, file_system = train_doc2vec_model(file_system)
    query_doc.doc2vec_vector = doc2vec_model.infer_vector(query.split())

    if bow_model is None:
        bow_model, file_system = train_bow(file_system)
    query_doc.bow = infer_bow_vector(query_doc.lemmatized_text, bow_model)

    save_file_system(file_system, doc2vec_model, bow_model, language)

    measure = get_similarity_kind()
    include_query = True
    find_and_print_most_similar_docs(query_doc, file_system, include_query, None, measure)

    syn_query = get_syn_query(language, query_doc)
    syn_query_doc = Document("syn query", language, syn_query)
    syn_query_doc.doc2vec_vector = doc2vec_model.infer_vector(syn_query.split())
    syn_query_doc.bow = infer_bow_vector(syn_query_doc.lemmatized_text, bow_model)
    include_query = False
    find_and_print_most_similar_docs(syn_query_doc, file_system, include_query, query_doc.lemmatized_text, False)


def main():
    # eBook sam vaknin new york
    # נשיא ארהב
    while True:
        run_query()
        ans = input("\nDo you want to run another query? (1 for yes, else anything) ")
        if ans != "1":
            print("\nGoodbye!")
            break


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
