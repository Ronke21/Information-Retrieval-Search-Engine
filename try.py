from googletrans import Translator

translator = Translator()

word = "למחוק"
# translate from hebrew to english
translated_word = translator.translate(word, src='he')
print(translated_word.text)