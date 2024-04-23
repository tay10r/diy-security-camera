#!venv/bin/python3

from dataset import Dataset, load_labels

def get_summary(root: str) -> tuple:
    ds = Dataset(root, load_labels('data/labels.json'), None)
    num_positives = 0
    num_negatives = 0
    for data in ds:
        img, label = data
        if label == 0:
            num_negatives += 1
        else:
            num_positives += 1
    return num_negatives, num_positives

def main():
    train_summary = get_summary('data/train')
    val_summary = get_summary('data/validation')
    print('Summary:')
    print(f' Training Data')
    print(f'   Negatives: {train_summary[0]}')
    print(f'   Positives: {train_summary[1]}')
    print(f' Validation Data')
    print(f'   Negatives: {val_summary[0]}')
    print(f'   Positives: {val_summary[1]}')

main()
