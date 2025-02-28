#!/usr/bin/env python


# Importing libraries
import argparse
import json
import logging
import requests
from config import CropWizardConfig, LangchainConfig, OpenAIConfig, OllamaConfig
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from os import environ, getenv
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate as ragas_eval
from ragas import metrics, EvaluationDataset
from time import sleep


# Error log config
logging.basicConfig(filename="cropwizard_rag_eval_error_log.txt", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logging.error("-" * 80)  # Separator line
logging.error("NEW EVALUATION RUN STARTED - %(asctime)s")
logging.error("-" * 80)

# Load environment variables
load_dotenv()


# Initialize CropWizard specific variables
config = CropWizardConfig()


# Initialize Langchain specific environment variables
LangchainConfig()

# Initialize LLM specific environment variables
openaiconfig = OpenAIConfig()
ollamaconfig = OllamaConfig()


# Defining methods
def get_prompt_tokens(prompt:str, 
                      url:str =config.prompt_endpoint,
                      db:str=config.db_version, 
                      groups:list=config.cw_groups, 
                      limit:int=config.token_limit, 
                      verbose:bool=False,
                      log:bool=True) -> str:

    """
    Posts a prompt to CropWizard, and returns the token vector as a JSON.
    Arguments:
    url -- Address of CropWizard instance being prompted.
    prompt -- A string representing the prompt submitted to CropWizard.
    db -- A string representing the name of the queried database. | Default: cropwizard-1.5
    groups -- a list containing all databases to be queried. | Default: ["All Documents"]
    limit -- An integer representing the token limit for the query. | Default: 128000
    
    Returns:
    A dictionary of tokens representing the fragments, retrieved from the submitted prompt.
    """

    payload:dict = {
    "course_name": db,
    "doc_groups": groups,
    "search_query": prompt,
    "token_limit": limit
    }

    if verbose:
        print(payload)

    response = requests.post(url, json=payload)

    # Error handling
    assert response.status_code == 200, f"Failed to retrieve data for get_prompt_tokens (error_code: {response.status_code})"
    if "ERROR: In /getTopContexts" in response.json():
        for attempt in range(3):
            if log:
                logging.error(f"Error ({attempt + 1}) in get_prompt_tokens() for question {prompt}: {response.json()}")
            sleep(0.25)
            response = requests.post(url, json=payload)
            if "ERROR: In /getTopContexts" not in response.json():
                break
            elif attempt == 3:
                if log:
                    logging.error(f"Max retries reached for get_prompt_tokens(). Failed request to obtain chunks for question: {prompt}.")
    
    fragments = response.json()
    
    return fragments


def query_cropwizard(prompt:str,
                     model:str=config.model,
                     url:str=config.answer_endpoint, 
                     course:str=config.db_version, 
                     group:list=config.cw_groups, 
                     limit:int=config.token_limit,
                     log:bool=True) -> dict:
    """
    Function to send a prompt to CropWizard and get the response.
    """
    payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": config.cropwiz_sys_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.1,
    "course_name": course,
    "doc_groups": group,
    "search_query": prompt,
    "token_limit": limit,
    "stream": True,
    "api_key": config.cropwiz_api_key,
}
    
    response = requests.post(url, json=payload)
    # Error handling
    assert response.status_code == 200, f"failed to retrieve data for query_cropwizard (error_code: {response.status_code})"

    if "Error processing streaming response" in response.text:
        for attempt in range(3):
            if log:
                logging.error(f"Error ({attempt + 1}) in query_cropwizard() for question {prompt}: {response.text}")
            sleep(0.25)
            response = requests.post(url, json=payload)
            if "Error processing streaming response" not in response.text:
                break
            elif attempt == 3:
                if log:
                    logging.error(f"Max retries reached for query_cropwizard(). Failed request to obtain streamed response for question: {prompt}.")

    if "The CropWizard database doesn't have anything covering this exact question" in response.text:
        if log:
            logging.error(f"Vector search mismatch for question: {prompt} - not found in database")
    
    return response.text


def create_test_cases(question_answer_pairs:dict) -> dict:
    """
    Creates a test case dictionary from a question-answer dictionary.

    Args:
        question_answer_pairs (dict): Dictionary with keys representing questions and values representing expert answers.

    Returns:
        test_cases (dict): Dictionary with keys "question", "answer", "retrieved_contexts", and "ground_truth".
    """

    test_cases = {"question":[], "answer":[], "retrieved_contexts":[], "ground_truth":[]}
    
    for key,value in question_answer_pairs.items():
        sleep(0.25)          # Added sleep to avoid issues on the server side
        test_cases["question"].append(key)
        test_cases["answer"].append(query_cropwizard(key))
        test_cases["retrieved_contexts"].append(get_prompt_tokens(key))
        test_cases["ground_truth"].append(value)

    return test_cases


def preprocess_test_cases(test_cases:dict) -> dict:
    """
    Extracts text from the "retrieved_contexts" key from a test_cases dictionary.

    Args:
        test_cases (dict): Dictionary with keys "question", "answer", "retrieved_contexts", and "ground_truth".

    Returns:
        dict: Dictionary with keys "question", "answer", "retrieved_contexts", and "ground_truth", where "retrieved_contexts" 
        now only contains the contents of its "text" key
    """

    return {
        "question": test_cases["question"],
        "answer": test_cases["answer"],
        "retrieved_contexts": [
            [entry["text"] for entry in inner_list] if isinstance(inner_list, list) else inner_list
            for inner_list in test_cases["retrieved_contexts"] if isinstance(test_cases["retrieved_contexts"], list)
        ],
        "ground_truth": test_cases["ground_truth"],
    }


def create_dataset(data:dict):
    """
    Cleans the input dictionary by removing entries where `retrieved_contexts` is a string instead of a list.

    Args:
        data (dict): Dictionary with keys "question", "answer", "retrieved_contexts", and "ground_truths".

    Returns:
        cleaned_data (dict): Cleaned dictionary with valid entries.
        removed_entries (list): List of dictionaries containing removed entries for review.
    """
    removed_entries = []  # To store removed tuples for review

    # Ensure all lists have the same length
    keys = ["question", "answer", "retrieved_contexts", "ground_truth"]
    assert all(len(data[key]) == len(data[keys[0]]) for key in keys), "All lists must have the same length."

    # Iterate over retrieved_contexts and remove invalid entries
    valid_indices = []
    for i, retrieved_context in enumerate(data["retrieved_contexts"]):
        if isinstance(retrieved_context, list):
            valid_indices.append(i)  # Keep valid entries
        elif isinstance(retrieved_context, str) and "error" in retrieved_context.lower():
            # Add invalid entries to removed_entries
            removed_entries.append((
                data["question"][i],
                data["answer"][i],
                data["retrieved_contexts"][i],
                data["ground_truth"][i],
            ))

    # 

    # Filter the dictionary to keep only valid entries
    cleaned_data = {
        key: [data[key][i] for i in valid_indices] for key in keys
    }

    return cleaned_data, removed_entries


def convert_dict_to_list(data:dict) -> list:
    """
    Converts a dictionary with keys 'question', 'answer', 'retrieved_contexts', and 'ground_truth'
    into a list of dictionaries with the desired structure.

    Args:
        data (dict): Input dictionary with keys as lists of matching indexes.

    Returns:
        list: A list of dictionaries following the specified layout.
    """
    dataset = []
    for i in range(len(data["question"])):
        dataset.append({
            "user_input": data["question"][i],
            "retrieved_contexts": data["retrieved_contexts"][i],
            "response": data["answer"][i],
            "reference": data["ground_truth"][i],
        })
    return dataset


def single_judge_evaluation(question_answer_pairs:dict,
                            judge:str="gpt-4o-mini",
                            log:bool=True,
                            ) -> str:
    """
    Evaluates RAG performance for a set of question-answer pairs using a specified LLM judge.

    Args:
        question_answer_pairs (dict): A dictionary containing question-answer pairs for evaluation.
        judge (str, optional): A string representing the choice of LLM model to use for evaluation. Defaults to "gpt-4o-mini".
        log (bool, optional): Whether to log errors. Defaults to True.

    Returns:
        results (str): Average result for each metric from the Ragas framework.
    """
    test_cases = preprocess_test_cases(create_test_cases(question_answer_pairs))
    evaluation_dict, errors = create_dataset(test_cases)
    if errors != []:
        if log:
            logging.error(f"errors in dataset creation: {errors}")
    # evaluation_list = convert_dict_to_list(evaluation_dict)
    
    # Convert dataset to LangSmith format
    langsmith_ragas_eval = EvaluationDataset.from_list(convert_dict_to_list(evaluation_dict))
    
    # Initialize Langchain LLM wrapper
    llm_options = {
    # OpenAI models
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=openaiconfig.temperature),
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=openaiconfig.temperature),
    
    # Ollama models
    "deepseek-r1": ChatOllama(model="deepseek-r1", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "llama3": ChatOllama(model="llama3", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "mistral": ChatOllama(model="mistral", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "mixtral": ChatOllama(model="mixtral", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    
    # Commented out models that could be added in the future
    # "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.1),
    # "command-r-plus": ChatCohere(model="command-r-plus", temperature=0.1),
    # "gemini-2-flash": ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.1),
    # "llama3-70b": ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.1),
    }

    # Check if the judge model is in the available options
    if judge in llm_options:
        evaluator_llm = LangchainLLMWrapper(llm_options[judge])
    else:
        if log:
            logging.error(f"Model '{judge}' not found in available models. Reverting to default model (gpt-4o-mini).")
        evaluator_llm = LangchainLLMWrapper(llm_options["gpt-4o-mini"])

    # Run evaluation
    results = ragas_eval(
        dataset=langsmith_ragas_eval,
        metrics=[
            metrics.ContextPrecision(),     # 
            metrics.ContextRecall(),        # 
            metrics.AnswerRelevancy(),      # 
            metrics.Faithfulness(),         # 
            metrics.FactualCorrectness(),   # 
        ],
        llm=evaluator_llm,
    )

    return results


# def multi_judge_evaluation(question_answer_pairs:dict,
#                             list_of_judges:list=[0,1,2],
#                             ) -> dict:
#     """
#     Evaluates RAG performance for a set of question-answer pairs using multiple LLM judges.

#     Args:
#         question_answer_pairs (dict): A dictionary containing question-answer pairs for evaluation.
#         judge (int, optional): A list representing the choices of LLM model to use for evaluation. Defaults to [0, 1, 2].

#     Returns:
#         results (dict): A dictionary containing the model name and average results for each metric from the Ragas framework.
#     """
#     test_cases = preprocess_test_cases(create_test_cases(question_answer_pairs))
#     evaluation_dataset, errors = create_dataset(test_cases)
#     if errors != []:
#         logging.error(f"errors in dataset creation: {errors}")
#     evaluation_dataset = convert_dict_to_list(evaluation_dataset)
    
#     # Convert dataset to LangSmith format
#     langsmith_ragas_eval = EvaluationDataset.from_list(convert_dict_to_list(evaluation_dataset))
    
#     # Initialize Langchain LLM wrapper
#     llm_options = {
#     0: ChatOpenAI(model="gpt-4o-mini"),
#     1: ChatOpenAI(model="gpt-4o"),
#     2: ChatOpenAI(model="gpt-3.5-turbo"),
# #     3: ChatAnthropic(model="claude-3-5-sonnet-latest")),  # Anthropic Claude 3.5 Sonnet
# #     4: ChatCohere(model="command-r-plus")),  # Cohere Command-R-plus
# #     5: ChatGoogleGenerativeAI(model="gemini-2.0-flash-001"),   # Google Vertex AI Gemini 2.0 Flash
# #     6: ChatNVIDIA(model="meta/llama3-70b-instruct"),   # NVIDIA LLaMA 3-70B
#     }

#     results = {}
#     for llm_index in list_of_judges:
#         evaluator_llm = LangchainLLMWrapper(llm_options.get(llm_index))

#         # Run evaluation
#         result = ragas_eval(
#             dataset=langsmith_ragas_eval,
#             metrics=[
#                 metrics.ContextPrecision(),
#                 metrics.ContextRecall(),
#                 metrics.AnswerRelevancy(),
#                 metrics.Faithfulness(),
#                 metrics.FactualCorrectness(),
#             ],
#             llm=evaluator_llm,
#         )
#         results[llm_options.get(llm_index)] = result

#     return results


def main(question_answer_pair:json, test_judge:list=["gpt-4o-mini"], ) -> str:
    with open(question_answer_pair, "r") as imported_json:
        imported_dataset = json.load(imported_json)
    list_of_judge_tests=["gpt-4o-mini", "gpt-4o", "deepseek-r1", "llama3", "mistral"]
    if all(item in list_of_judge_tests for item in test_judge):
        if len(test_judge) == 1:
            return single_judge_evaluation(imported_dataset, test_judge[0])
        # elif len(test_judge) > 1:
        #     return multi_judge_evaluation(imported_dataset, test_judge)
    else:
        raise ValueError(f"One or more invalid values for test_judge: {test_judge}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--qa_file', type=str, required=True,
                        help='Path to the JSON file containing question-answer pairs')
    
    args = parser.parse_args()
    
    main(args.qa_file)
