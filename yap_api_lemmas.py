# Author: Amit Shkolnik
# Python Version: 3.6


## Copyright 2019 Amit Shkolnik
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import json
import time

import pandas as pd
import numpy as np
import requests
import sys
import traceback
import csv
import re, string
from enums import *
from hebtokenizer import HebTokenizer


class YapApi(object):
    """
    Interface to Open University YAP (Yet another parser) https://github.com/OnlpLab/yap.
    This class is calling GO baesd server, and:
    1. Wrap YAP in Python.
    2. Add tokenizer. Credit: Prof' Yoav Goldberg.
    3. Turn output CONLLU format to Dataframe & JSON.
    """

    def __init__(self):
        pass

    def run(self, text: str, ip: str):
        """
        text: the text to be parsed.
        ip: YAP server IP, with port (default is 8000), if localy installed then 127.0.0.1:8000
        """
        try:
            # print('Start Yap call')
            # Keep alpha-numeric and punctuations only.
            alnum_text = self.clean_text(text)
            # Tokenize...
            tokenized_text = HebTokenizer().tokenize(alnum_text)
            tokenized_text = ' '.join([word for (part, word) in tokenized_text])
            # print("Tokens: {}".format(len(tokenized_text.split())))
            self.init_data_items()
            # Split to sentences for best performance.
            text_arr = self.split_text_to_sentences(tokenized_text)
            for i, sntnce_or_prgrph in enumerate(text_arr):
                # Actual call to YAP server
                rspns = self.call_yap(sntnce_or_prgrph)
                #print('End Yap call {} /{}'.format(i, len(text_arr) - 1))

                _dep_tree = self.parse_dep_tree(rspns['dep_tree']).fillna(-1)
                _segmented_text = ' '.join(_dep_tree[yap_cols.word.name])
                _lemmas = ' '.join(_dep_tree[yap_cols.lemma.name])
                self.append_prgrph_rslts(_segmented_text, _lemmas)
            return tokenized_text, self.segmented_text, self.lemmas

        except Exception as err:
            """
            print(sys.exc_info()[0])
            print(traceback.format_exc())
            print(str(err))
            """
        # print("Unexpected end of program")
            return None

    def split_text_to_sentences(self, tokenized_text):
        """
        YAP better perform on sentence-by-sentence.
        Also, dep_tree is limited to 256 nodes.
        """
        max_len = 150
        arr = tokenized_text.strip().split()
        sentences = []
        # Finding next sentence break.
        while (True):
            stop_points = [h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?]", e)]]
            if len(stop_points) > 0:
                stop_point = min(stop_points)
                # Keep several sentence breaker as 1 word, like "...." or "???!!!"
                while True:
                    stop_points.remove(stop_point)
                    if len(stop_points) > 1 and min(stop_points) == (stop_point + 1):
                        stop_point = stop_point + 1
                    else:
                        break
                # Case there is no sentence break, and this split > MAX LEN:
                sntnc = arr[:stop_point + 1]
                if len(sntnc) > max_len:
                    while (len(sntnc) > max_len):
                        sentences.append(" ".join(sntnc[:140]))
                        sntnc = sntnc[140:]
                    sentences.append(" ".join(sntnc))
                # Normal: sentence is less then 150 words...
                else:
                    sentences.append(" ".join(arr[:stop_point + 1]))
                arr = arr[stop_point + 1:]
            else:
                break
        if len(arr) > 0:
            sentences.append(" ".join(arr))
        return sentences

    def clean_text(self, text: str):
        text = text.replace('\n', ' ').replace('\r', ' ')
        pattern = re.compile(r'[^א-ת\s.,!?a-zA-Z]')
        alnum_text = pattern.sub(' ', text)
        while (alnum_text.find('  ') > -1):
            alnum_text = alnum_text.replace('  ', ' ')
        return alnum_text

    def init_data_items(self):
        self.segmented_text = ""
        self.lemmas = ""
        self.dep_tree = pd.DataFrame()

    def append_prgrph_rslts(self, _segmented_text: str, _lemmas: str):
        self.segmented_text = "{} {}".format(self.segmented_text, _segmented_text).strip()
        self.lemmas = "{} {}".format(self.lemmas, _lemmas).strip()

    def split_long_text(self, tokenized_text: str):
        # Max num of words YAP can handle at one call.
        max_len = 150
        arr = tokenized_text.split()
        rslt = []
        while len(arr) > max_len:
            # Finding next sentence break.
            try:
                stop_point = min([h for h in [i for i, e in enumerate(arr) if re.match(r"[!|.|?]", e)] if h > max_len])
            except Exception as err:
                if str(err) == "min() arg is an empty sequence":
                    stop_point = 150
                if len(arr) < stop_point:
                    stop_point = len(arr) - 1
            rslt.append(" ".join(arr[: (stop_point + 1)]))
            arr = arr[(stop_point + 1):]
        rslt.append(" ".join(arr))
        return rslt

    def call_yap(self, text: str):
        localhost_yap = "http://localhost:8000/yap/heb/joint"
        data = json.dumps({'text': "{}  ".format(text)})  # input string ends with two space characters
        headers = {'content-type': 'application/json'}
        response = requests.get(url=localhost_yap, data=data, headers=headers)
        if response.status_code != 200:
            print("Error: {}".format(response.status_code))
            print(response.text)
            return None
        json_response = response.json()
        return json_response

    def parse_dep_tree(self, v: str):
        data = [sub.split("\t") for item in str(v).split("\n\n") for sub in item.split("\n")]
        labels = [yap_cols.num.name, yap_cols.word.name, yap_cols.lemma.name, yap_cols.pos.name, yap_cols.pos_2.name,
                  yap_cols.empty.name, yap_cols.dependency_arc.name, yap_cols.dependency_part.name,
                  yap_cols.dependency_arc_2.name, yap_cols.dependency_part_2.name]
        # remove first line w=hich is empty
        data = [l for l in data if len(l) != 1]
        # remove stop char
        new_data = []
        for row in data:
            n_row = [word.replace("\r", "") for word in row]
            new_data.append(n_row)
        df = pd.DataFrame.from_records(new_data, columns=labels)
        # Case YAP find punctuation chars like ',', '.', YAP set no lemma for them.
        # That case set the word to be its own lemma
        df.loc[df[yap_cols.lemma.name] == '', [yap_cols.lemma.name]] = df[yap_cols.word.name]
        return df


def heb_yap_process(text: str):
    # IP of YAP server, if locally installed then '127.0.0.1'
    ip = '127.0.0.1:8000'
    yap = YapApi()
    try:
        tokenized_text, segmented_text, lemmas = yap.run(text, ip)
    except Exception as err:
        print("\n\nError: YAP server not running.\nPlease start YAP server in port 8000 and start again\nInstructions in https://github.com/OnlpLab/yap\n")
        sys.exit(1)
    return lemmas


if __name__ == '__main__':
    # CALC time for the whole process
    start_time = time.time()
    # The text to be processed.
    # read file hebrew_files_short/2125.txt coding = utf-8
    text = open("hebrew_files_short/2125.txt", "r", encoding="utf-8").read()

    tokenized_text, segmented_text, lemmas = heb_yap_process(text)

    print("Tokenized text:", tokenized_text)
    print("Segmented text:", segmented_text)
    print("Lemmas:", lemmas)
    print("\n\n")
    print('Program end')
    print("--- %s seconds ---" % (time.time() - start_time))
