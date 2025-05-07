import csv
import datetime
import logging
import re
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
import os
import pandas as pd
logging.basicConfig(level='ERROR')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data


def get_metrics(scores, labels):
    labels = np.array(labels, dtype=int)  # Ensure labels are in binary format
    scores = np.array(scores)  # Ensure scores are in the correct format
    # Calculate ROC curve and AUROC
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    # Check if AUROC is below 0.5 and invert scores if necessary
    if auroc < 0.5:
        scores = -scores  # Invert scores
        fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr_list, tpr_list)
    # Calculate FPR at TPR >= 0.95
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]] if np.any(tpr_list >= 0.95) else np.nan
    # Calculate TPR at FPR <= 0.05
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]] if np.any(fpr_list <= 0.05) else np.nan

    # Calculate accuracy
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    return auroc, fpr95, tpr05, acc


def do_plot(prediction, answers, legend="", output_dir=None):
    fpr, tpr, auc_score, acc = get_metrics(prediction, answers)
    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
    print(f"Type of FPR: {type(fpr)}, Type of TPR: {type(tpr)}")
    if len(fpr) == 0 or len(tpr) == 0:
        print(f"No valid FPR/TPR values for {legend}")
        low = 0.0
    else:
        fpr_below_threshold = fpr[fpr < .05]
        if len(fpr_below_threshold) == 0:
            print(f"No FPR values below 0.05 for {legend}")
            low = 0.0
        else:
            low = tpr[np.where(fpr < .05)[0][-1]]
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n' % (legend, auc_score, acc, low))
    metric_text = 'auc=%.3f' % auc_score
    plt.plot(fpr, tpr, label=legend + metric_text)
    return legend, auc_score, acc, low


def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = None
    metric2predictions = defaultdict(list)

    # Collect labels and predictions
    for ex in all_output:
        if answers is None:
            answers = ex["label"]
            if not isinstance(answers, list):
                answers = [answers]  # Ensure labels are a list
            print(f"Collected {len(answers)} labels from one example.")
        for metric, scores in ex["pred"].items():
            if not isinstance(scores, list):
                scores = [scores]  # Ensure scores are a list
            print(f"Collected {len(scores)} scores for metric {metric} from one example.")
            metric2predictions[metric].extend(scores)

    print(f"Total collected labels: {len(answers)}")

    # Check if lengths of answers and each set of predictions match
    valid_metrics = {}
    for metric, predictions in metric2predictions.items():
        print(f"Checking lengths for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
        if len(predictions) != len(answers):
            print(f"Length mismatch for metric {metric}: {len(predictions)} predictions vs {len(answers)} answers")
            continue
        valid_metrics[metric] = predictions

    print(f"Valid metrics: {list(valid_metrics.keys())}")

    # Plotting and writing to file
    auc_file_path = f"{output_dir}/auc.txt"
    with open(auc_file_path, "w") as f:
        for metric, predictions in valid_metrics.items():
            print(f"Processing metric {metric} with {len(predictions)} predictions and {len(answers)} answers")
            plt.figure(figsize=(4, 3))
            legend, auc_score, acc, low = do_plot(predictions, answers, legend=metric, output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc_score, acc, low))

            plt.semilogx()
            plt.semilogy()
            plt.xlim(1e-5, 1)
            plt.ylim(1e-5, 1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.plot([0, 1], [0, 1], ls='--', color='gray')
            plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
            plt.legend(fontsize=8)
            plt.savefig(f"{output_dir}/{metric}_auc.png")
            plt.close()  # Close the plot to avoid overlap in subsequent iterations
            print(f"Saved plot for metric {metric} to {output_dir}/{metric}_auc.png")

    print(f"AUC results saved to {auc_file_path}")


def save_metrics_to_csv(results, output_dir, prefix):
    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame(results)

    # Specify the attacks to keep
    attacks_to_keep = [
        "ppl", "ppl_lowercase_ppl", "ppl_zlib",
        "Min_20.0% Prob", "Max_20.0% Prob",
        "MinK++_20.0% Prob", "sentence_entropy_log_likelihood",
        "sentence_entropy_log_likelihood_kpp","sentence_log_probs_one_token", "sentence_log_probs_one_token_kpp"
    ]

    # Filter the DataFrame to include only the specified attacks
    df_filtered = df[df['method'].isin(attacks_to_keep)]

    # Drop the fpr95 column
    df_filtered = df_filtered.drop(columns=['fpr95'])

    # Specify the CSV file path
    csv_file_name = f'metrics_results_{prefix}.csv'
    csv_file_path = os.path.join(output_dir, csv_file_name)

    # Save the DataFrame to CSV with the specified header
    df_filtered.to_csv(csv_file_path, index=False, header=["method", "auroc", "tpr05", "acc"])

    print(f"Results saved to {csv_file_path}")


def convert_data_format(data):
    """
    Convert data to the format required by evaluate_and_save_results.
    """
    scores_dict = defaultdict(list)

    for entry in data:
        label = entry['label']
        predictions = entry['pred']

        for method, score in predictions.items():
            scores_dict[method].append(score)

    return scores_dict


def write_to_csv_pred_min_k(data, filename):
    # data = convert_data_format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Check if data is not empty
    if data:
        # Determine the headers from the keys of the first dictionary
        headers = data[0].keys()

        # Create or overwrite the CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in data:
                # Filter the row to only include keys in the headers
                filtered_row = {key: row[key] for key in headers if key in row}
                writer.writerow(filtered_row)
                # writer.writerow(row)
    else:
        print("No data to write to CSV")


def evaluate_and_save_results(scores_dict, data, dataset, folder, prefix="", output_dir=None):
    # scores_dict = convert_data_format(scores_dict)
    labels = [d['label'] for d in data]  # Ensure labels are binary
    results = defaultdict(list)
    all_output = []
    roc_data = {}

    for method, scores in scores_dict.items():
        # Remove NaN and infinity values
        clean_scores = [score for score in scores if not (np.isnan(score) or np.isinf(score))]
        clean_labels = [labels[i] for i, score in enumerate(scores) if not (np.isnan(score) or np.isinf(score))]

        if len(clean_scores) != len(clean_labels):  # Ensure lengths match
            print(f"Length mismatch for method {method}: {len(clean_scores)} scores vs {len(clean_labels)} labels")
            continue

        if not clean_scores:  # Check if clean_scores is empty
            print(f"No valid scores for method: {method}")
            continue

        auroc, fpr95, tpr05, acc = get_metrics(clean_scores, clean_labels)
        fpr, tpr, _ = roc_curve(clean_labels, clean_scores)


        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")
        results['acc'].append(f"{acc:.1%}")

        # # Store ROC data for combined plot
        roc_data[method] = (fpr, tpr, auroc)

        all_output.append({
            "label": clean_labels,
            "pred": {method: clean_scores}
        })
    if output_dir:
        save_root = f"{output_dir}/results/{dataset}/{folder}"
    else:
        save_root = f"results/{dataset}/{folder}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Create and save the combined ROC curve plot
    create_combined_roc_curve_plot(roc_data, save_root, title=dataset, prefix=prefix)

    df = pd.DataFrame(results)
    print(df)

    # Save the results to CSV
    # save_metrics_to_csv(results, save_root, prefix)
    file_name = f"metrics_{prefix}.csv"
    print(f"Metrics file name: {save_root}/{file_name}")
    if os.path.isfile(os.path.join(save_root, file_name)):
        df.to_csv(os.path.join(save_root, file_name), index=False, mode='w', header=False)
    else:
        df.to_csv(os.path.join(save_root, file_name), index=False)

    # fig_fpr_tpr(all_output, save_root)


def create_combined_roc_curve_plot(roc_data, output_dir, title='ROC curve', prefix=""):
    """
    Create and save a combined ROC curve plot.
    """
    plt.figure(figsize=(10, 8))

    for method, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{method} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'roc_curve_{prefix}.png'))
    plt.close()


def evaluate_like_min_k(csv_path, kind=''):
    # Extract dataset and model information from the path
    parts = csv_path.split('/')
    # print(parts)
    dataset = parts[-3]
    folder = parts[-2]
    output_dir = '/'.join(parts[:-4])

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Exclude the 'FILE_PATH' and 'label' columns to get all other features dynamically
    attacks = [col for col in df.columns if col not in ['FILE_PATH', 'label']]

    # Extract labels
    labels = df['label'].tolist()

    # Prepare scores dictionary
    scores_dict = {attack: df[attack].tolist() for attack in attacks}

    # Evaluate and save results
    evaluate_and_save_results(scores_dict, [{'label': label} for label in labels], dataset, folder=folder, prefix=kind, output_dir=output_dir)