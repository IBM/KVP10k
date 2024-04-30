import argparse
import pandas as pd
from utils.annotation_utils.split_statistics import split_statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_path", type=str, help='path to dataset root folder', required=True)
    parser.add_argument('--splits', nargs='+', type=str, default=["test", "train"], help='An array of split values')
    args = parser.parse_args()

    results = []

    for split in args.splits:
        cs = split_statistics(args.dataset_root_path, [split])

        split_data = {
            "total images": cs.count_images(),
            "total annotations": cs.count_annotation(),
            "total empty annotations": cs.count_empty_annotation(),
            "total flat_kvp" : cs.count_flat_kvp(),
            "total unkeyed_value": cs.count_unkeyed_value(),
            "total unvalued_key": cs.count_unvalued_key(),
        }

        results.append((split, split_data))

    df = pd.DataFrame(dict(results)).transpose()
    df.loc['All'] = df.sum()

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    from pandas.plotting import table

    # Plot table
    fig, ax = plt.subplots(figsize=(12, 2))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    the_table = table(ax, df, loc='center')

    plt.show()
