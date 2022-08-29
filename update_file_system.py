from project import *
import schedule
import time
def update_file_system():
    languages = ["HEB", "ENG"]
    for language in languages:
        file_system, doc2vec_model, bow_model = get_file_system(language)
        doc2vec_model, file_system = train_doc2vec_model(file_system)
        bow_model, file_system = train_bow(file_system)
        save_file_system(file_system, doc2vec_model, bow_model)

def main():
    schedule.every().day.at("01:00").do(update_file_system, 'It is 01:00')

    while True:
        schedule.run_pending()
        time.sleep(60)  # wait one minute


if __name__ == '__main__':
    main()

