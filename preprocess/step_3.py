import concurrent.futures
import pickle
from pathlib import Path

from windowing import generate_windows


def sample(data, thread):
    with open('data/var.pkl', 'rb') as file:
        _, target_var = pickle.load(file)
    samples, info = generate_windows(data, target_var)
    with open('data/first/samples_{}.pkl'.format(thread + 1), 'wb') as file:
        pickle.dump([samples, info], file)
    print('Thread_{} finished'.format(thread))


if __name__ == '__main__':
    with open('data/sets.pkl', 'rb') as file:
        sets = pickle.load(file)
    Path('data/first').mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Consume results so worker failures are not silently ignored.
        list(executor.map(sample, sets, range(len(sets))))
