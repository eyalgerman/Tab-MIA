import logging
import datetime
import os
import numpy as np
from tqdm import tqdm
from MIA import eval_2
from MIA.eval_2 import load_jsonl
from MIA.create_map_entropy_both import create_entropy_map, create_line_to_top_words_map, bottom_k_entropy_words
logging.basicConfig(level='ERROR')
from pathlib import Path
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from options import Options
import torch.nn.functional as F
import spacy

MAX_LEN_LINE_GENERATE = 40
MIN_LEN_LINE_GENERATE = 7
MAX_LEN_LINE = 10000
TOP_K_ENTROPY = 2


def load_model(name1):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)
    return model1, tokenizer1


def load_local_model(name1):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(Path(name1), return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(Path(name1))

    return model1, tokenizer1


def calculate_perplexity(sentence, model, tokenizer, gpu):
    """
    Calculates the perplexity and log-probabilities for a given sentence.

    Args:
        sentence (str): Input sentence to evaluate.
        model (transformers.PreTrainedModel): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        gpu (torch.device): Device to run computation on.

    Returns:
        Tuple[float, List[float], float, torch.Tensor, torch.Tensor]:
        Perplexity, list of log probabilities, loss, logits, and processed input token IDs.

    Raises:
        RuntimeError: If inference fails due to input length or model errors.
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    try:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        # Move to CPU immediately
        logits = logits.detach().cpu()

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        all_prob = []
        input_ids_processed = input_ids[0][1:].cpu()
        for i, token_id in enumerate(input_ids_processed):
            probability = probabilities[0, i, token_id].item()
            all_prob.append(probability)
        return torch.exp(loss).item(), all_prob, loss.item(), logits, input_ids_processed
    except RuntimeError as e:
        # print(f"Error: {e} in sentence: {sentence}")
        print("Length of sentence: ", len(sentence))
        raise e


def inference(model1, tokenizer1, text, label, name, line_to_top_words_map, entropy_map, data_name=None):
    """
    Runs a detailed statistical analysis of a text input to assess MIA indicators.

    Args:
        model1 (transformers.PreTrainedModel): The loaded model for inference.
        tokenizer1 (transformers.PreTrainedTokenizer): Corresponding tokenizer.
        text (str): The input text to evaluate.
        label (int): Label associated with the input (e.g., member or non-member).
        name (str): Identifier or file name for the input.
        line_to_top_words_map (dict): Map of line number to top entropy words.
        entropy_map (dict): Map of token entropies across the dataset.
        data_name (str, optional): Optional name of the dataset.

    Returns:
        dict: Dictionary of computed metrics and probabilities for the input.
    """
    pred = {}
    pred["FILE_PATH"] = name
    pred["label"] = label
    if data_name:
        pred["data_name"] = data_name

    # clean up memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    with torch.no_grad():
        p1, all_prob, p1_likelihood, logits, input_ids_processed = calculate_perplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, all_prob_lower, p_lower_likelihood, logits_lower, input_ids_processed_lower = calculate_perplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

        # ppl
        pred["ppl"] = p1

        # Ratio of log ppl of lower-case and normal-case
        pred["ppl_lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
        # Ratio of log ppl of large and zlib
        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
        pred["ppl_zlib"] = np.log(p1) / zlib_entropy

        # min-k prob
        for ratio in [0.1, 0.2, 0.3]:
            k_length = int(len(all_prob) * ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            pred[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

        # max-k prob
        for ratio in [0.1, 0.2, 0.3]:
            k_length = int(len(all_prob) * ratio)
            topk_prob = np.sort(all_prob)[-k_length:]
            pred[f"Max_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

        # Min-K++
        input_ids = torch.tensor(tokenizer1.encode(text)).unsqueeze(0).to(model1.device)
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)

        input_ids = input_ids.cpu() # Now all on CPU
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        # move to cpu
        mu = mu.detach().cpu()
        sigma = sigma.detach().cpu()
        token_log_probs = token_log_probs.detach().cpu()
        ## mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.1, 0.2, 0.3]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            pred[f"MinK++_{ratio * 100}% Prob"] = np.mean(topk).item()

        tokens = tokenizer1.tokenize(text)
        concatenated_tokens = "".join(token for token in tokens)
        mink_plus = mink_plus.cpu()
        torch.cuda.empty_cache()

        # Define the values of k you want to iterate over
        k_values = range(1, 10) # all k from 1 to 9 inclusive

        # Create a list of bottom k words once
        all_bottom_k_words = {}

        for line_num, top_words in line_to_top_words_map.items():
            bottom_k_words = bottom_k_entropy_words(" ".join(top_words), entropy_map, max(k_values))
            all_bottom_k_words[line_num] = bottom_k_words

        # Intermediate storage for results
        intermediate_results = {
            "relevant_log_probs": [],
            "relevant_log_probs_zlib": [],
            "relevant_log_probs_kpp": [],
            "relevant_log_probs_one_token": [],
            "relevant_log_probs_one_token_kpp": [],
            "relevant_indexes": []
        }

        # Process bottom k words once, ensuring lowest to highest entropy processing
        for line_num, bottom_k_words in all_bottom_k_words.items():
            for i, word in enumerate(bottom_k_words):
                if word in concatenated_tokens:
                    start_index = concatenated_tokens.find(word)
                    end_index = start_index + len(word)
                    start_token_index = end_token_index = None
                    current_length = 0
                    for j, token in enumerate(tokens):
                        current_length += len(token)
                        if current_length > start_index and start_token_index is None:
                            start_token_index = j
                        if current_length >= end_index:
                            end_token_index = j
                            break
                    if start_token_index is not None and end_token_index is not None:
                        if start_token_index < len(all_prob):
                            intermediate_results["relevant_log_probs_one_token"].append((i, all_prob[start_token_index]))
                            intermediate_results["relevant_log_probs_one_token_kpp"].append((i, mink_plus[start_token_index]))
                        for idx in range(start_token_index, end_token_index + 1):
                            if idx < len(all_prob):
                                intermediate_results["relevant_log_probs"].append((i, all_prob[idx]))
                                if zlib_entropy != 0:
                                    intermediate_results["relevant_log_probs_zlib"].append(
                                        (i, np.log(abs(all_prob[idx])) / zlib_entropy))
                                intermediate_results["relevant_log_probs_kpp"].append((i, mink_plus[idx]))
                                intermediate_results["relevant_indexes"].append((i, idx))

        # Calculate and store results for each k value
        for k in k_values:
            relevant_log_probs = [val for i, val in intermediate_results["relevant_log_probs"] if i < k]
            relevant_log_probs_zlib = [val for i, val in intermediate_results["relevant_log_probs_zlib"] if i < k]
            relevant_log_probs_kpp = [val for i, val in intermediate_results["relevant_log_probs_kpp"] if i < k]
            relevant_log_probs_one_token = [val for i, val in intermediate_results["relevant_log_probs_one_token"] if i < k]
            relevant_log_probs_one_token_kpp = [val for i, val in intermediate_results["relevant_log_probs_one_token_kpp"]
                                                if i < k]

            if relevant_log_probs:
                sentence_log_likelihood = np.mean(relevant_log_probs)
                pred[f"sentence_entropy_log_likelihood_k={k}"] = sentence_log_likelihood

            if relevant_log_probs_zlib:
                sentence_log_likelihood_zlib = np.mean(relevant_log_probs_zlib)
                pred[f"sentence_entropy_log_likelihood_zlib_k={k}"] = sentence_log_likelihood_zlib

            if relevant_log_probs_kpp:
                sentence_log_likelihood_kpp = np.mean(relevant_log_probs_kpp)
                pred[f"sentence_entropy_log_likelihood_kpp_k={k}"] = sentence_log_likelihood_kpp

            if relevant_log_probs_one_token:
                sentence_log_probs_one_token = np.mean(relevant_log_probs_one_token)
                pred[f"sentence_log_probs_one_token_k={k}"] = sentence_log_probs_one_token

            if relevant_log_probs_one_token_kpp:
                sentence_log_probs_one_token_kpp = np.mean(relevant_log_probs_one_token_kpp)
                pred[f"sentence_log_probs_one_token_k={k}"] = sentence_log_probs_one_token_kpp

        # Process lower case tokens and top words for all_prob_lower
        tokens_lower = tokenizer1.tokenize(text.lower())
        concatenated_tokens_lower = "".join(token for token in tokens_lower)
        relevant_log_probs_lower = []
        relevant_log_probs_zlib_lower = []
        relevant_indexes_lower = []
        # Process top words from all lines
        for line_num, top_words in line_to_top_words_map.items():
            # print(top_words)
            for word in top_words:
                word = word.lower()
                if word in concatenated_tokens_lower:
                    start_index = concatenated_tokens_lower.find(word)
                    end_index = start_index + len(word)
                    start_token_index = end_token_index = None
                    current_length = 0
                    for i, token in enumerate(tokens_lower):
                        current_length += len(token)
                        if current_length > start_index and start_token_index is None:
                            start_token_index = i
                        if current_length >= end_index:
                            end_token_index = i
                            break
                    if start_token_index is not None and end_token_index is not None:
                        for idx in range(start_token_index, end_token_index + 1):
                            if idx < len(all_prob_lower):
                                relevant_log_probs_lower.append(all_prob_lower[idx])
                                if zlib != 0:
                                    relevant_log_probs_zlib_lower.append(np.log(abs(all_prob_lower[idx])) / zlib_entropy)
                                relevant_indexes_lower.append(idx)

        if relevant_log_probs_lower:
            sentence_log_likelihood_lower = np.mean(relevant_log_probs_lower)
            pred["sentence_entropy_log_likelihood_lower"] = sentence_log_likelihood_lower
        if relevant_log_probs_zlib_lower:
            sentence_log_likelihood_zlib_lower = np.mean(relevant_log_probs_zlib_lower)
            pred["sentence_entropy_log_likelihood_zlib_lower"] = sentence_log_likelihood_zlib_lower

    return pred


def evaluate_data(test_data, model1, tokenizer1, col_name, modelname1, mode):
    """
    Evaluates a dataset for watermark or MIA indicators using the specified model.

    Args:
        test_data (List[dict]): List of input records (e.g., from a `.jsonl` file).
        model1 (transformers.PreTrainedModel): Model for inference.
        tokenizer1 (transformers.PreTrainedTokenizer): Tokenizer for the model.
        col_name (str): Key in the input dicts for text input (e.g., "input").
        modelname1 (str): Name or path of the model used.
        mode (str): Mode identifier (e.g., for entropy mapping).

    Returns:
        List[dict]: List of dictionaries with computed MIA metrics per input.
    """
    print(f"All data size: {len(test_data)}")
    print(f"mode: {mode}")
    all_output = []
    test_data = test_data
    nlp_spacy = spacy.load("en_core_web_sm")
    entropy_map = create_entropy_map(test_data, mode=mode)
    # print("test_data:", test_data)
    num_records_too_long = 0
    for ex in tqdm(test_data):
        text = ex[col_name]
        data_name = ex.get("data_name", None)
        # print("text: ", text)
        # if len(text.split()) < MIN_LEN_LINE_GENERATE or len(text.split()) > MAX_LEN_LINE_GENERATE:
        #     continue
        if len(text) > MAX_LEN_LINE:
            print(f"Text too long: {len(text)} make it shorter")
            text = text[:MAX_LEN_LINE]
            num_records_too_long += 1
            # continue
        line_to_top_words_map, sentences = create_line_to_top_words_map(
            text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY, nlp_spacy
        )
        new_ex = inference(model1, tokenizer1, text, ex['label'], modelname1, line_to_top_words_map, entropy_map, data_name)
        all_output.append(new_ex)
    print(f"Number of records too long: {num_records_too_long} out of {len(test_data)}, {num_records_too_long / len(test_data) * 100}%")
    print(f"Max length of line: {MAX_LEN_LINE}")
    return all_output


def main(model_path, data_path, suffix = "", output_dir=""):
    """
    Main function to run the watermark detection pipeline on a dataset.

    Loads model and data, computes metrics for MIA detection, and saves the results to disk.

    Args:
        model_path (str): Path to the fine-tuned or base model.
        data_path (str): Path to the input `.jsonl` file.
        suffix (str, optional): Optional suffix for result file naming.
        output_dir (str, optional): Directory to save the output results.

    Returns:
        None
    """
    print("Start watermark detection")
    print(f"Target model: {model_path}")
    print(f"Data: {data_path}")
    # load model and data
    model, tokenizer = load_local_model(model_path)
    if "jsonl" in data_path:
        data = load_jsonl(f"{data_path}")
    else:  # load data from huggingface
        data = None
        print("Data is not jsonl file")
        return None
    all_output = evaluate_data(data, model, tokenizer, "input", model_path, data_path)
    # dataset = data_path.rstrip('/').split('/')[-2]
    dataset = os.path.basename(data_path).split(".")[0]
    print(f"Dataset: {dataset}")
    model = model_path.rstrip('/').split('/')[-1]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = f"E=MIA_detection{suffix}"
    # folder = f"M={model}_{current_time}"
    file_preds = f"{output_dir}/results/{model}/{dataset}{suffix}_{current_time}/preds_{kind}.csv"
    eval_2.write_to_csv_pred_min_k(all_output, file_preds)
    print(f"Results preds saved to {file_preds}")
    eval_2.evaluate_like_min_k(file_preds, kind=kind)


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    main(args)
