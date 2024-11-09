import os


def run():
    os.system("source scripts/flare_catalog/pipeline.sh")
    os.system("python3 scripts/flare_catalog/classifier.py")


if __name__ == "__main__":
    run()
