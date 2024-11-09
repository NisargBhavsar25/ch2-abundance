import os
from multiprocessing import Pool


def run_classifier(arg):
    os.system(f"python ./SuryaDrishti/classifier.py {arg}")


if __name__ == '__main__':
    pool = Pool(8)
    pool.map(run_classifier, range(1, 58))
    pool.close()
    pool.join()
