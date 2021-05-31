from xtlearn.utils import make_directory, dump_pickle, load_pickle


def main() -> None:

    model = load_pickle("stages/model.pkl")


if __name__ == "__main__":
    main()
