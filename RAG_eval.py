#!/usr/bin/env python

# !pip install -r requirements.txt

# Importing libraries
import argparse
import json
import logging
import requests
import datetime
import os
from collections import Counter
from typing import Optional #, Union, Dict, List, 
from config import CropWizardConfig, LangchainConfig, OpenAIConfig, OllamaConfig
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
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


class ReportGenerator:
    """
    A class to generate and manage markdown reports for RAG evaluation.
    This class handles creating, updating, and saving reports with various sections.
    """
    
    def __init__(self, report_dir="evaluation_reports"):
        """
        Initialize a new report with timestamp.
        
        Args:
            report_dir (str): Directory to store reports
        """
        self.timestamp = datetime.datetime.now()
        self.formatted_time = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = report_dir
        self.report_path = f"{report_dir}/rag_evaluation_{self.formatted_time}.md"
        self.content = []
        self.errors = Counter()
        self.metadata = {}
        self.error_messages = []
        
        # Create report directory if it doesn't exist
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # Initialize report with header
        self.add_header(f"CropWizard RAG Evaluation Report")
        self.add_text(f"**Date and Time:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def add_header(self, text, level=1):
        """Add a header to the report"""
        self.content.append(f"{'#' * level} {text}\n")
    
    def add_text(self, text):
        """Add plain text to the report"""
        self.content.append(f"{text}\n")
    
    def add_metadata(self, section, data):
        """
        Add metadata to the report
        
        Args:
            section (str): Section name for the metadata
            data (dict): Dictionary of metadata key-value pairs
        """
        if section not in self.metadata:
            self.metadata[section] = {}
        
        self.metadata[section].update(data)
    
    def add_error(self, error_type, message=None):
        """
        Log an error in the report
        
        Args:
            error_type (str): Type of error
            message (str, optional): Error message
        """
        self.errors[error_type] += 1
        if message:
            self.error_messages.append((error_type, message))
    
    def add_metrics(self, metrics_dict, prefix=""):
        """
        Add metrics to the report
        
        Args:
            metrics_dict (dict): Dictionary of metrics
            prefix (str, optional): Prefix for section name
        """
        self.add_header(f"{prefix}Metrics", level=2)
        
        for metric, value in metrics_dict.items():
            if isinstance(value, dict) and 'mean' in value:
                self.add_text(f"- **{metric}:** {value['mean']:.4f}")
            else:
                self.add_text(f"- **{metric}:** {value}")
        
        self.add_text("")
    
    def add_question_result(self, index, question, metrics, had_error=False, error_message=""):
        """
        Add individual question result
        
        Args:
            index (int): Question index
            question (str): The question text
            metrics (dict): Metrics for this question
            had_error (bool): Whether this question had an error
            error_message (str): Error message if applicable
        """
        self.add_header(f"Question {index+1}", level=3)
        self.add_text(f"**Question:** {question}\n")
        
        if had_error:
            self.add_text(f"**ERROR:** {error_message}\n")
        
        for metric, value in metrics.items():
            self.add_text(f"- **{metric}:** {value:.4f if isinstance(value, float) else value}")
        
        self.add_text("")
    
    def generate_report(self):
        """Generate the complete report content"""
        report_content = []
        
        # Add all content up to this point
        report_content.extend(self.content)
        
        # Add metadata section
        report_content.append("## Metadata\n")
        for section, data in self.metadata.items():
            report_content.append(f"### {section}\n")
            for key, value in data.items():
                report_content.append(f"- **{key}:** {value}\n")
            report_content.append("\n")
        
        # Add error statistics
        report_content.append("## Error Statistics\n")
        report_content.append(f"- **Total Errors:** {sum(self.errors.values())}\n")
        for error_type, count in self.errors.items():
            report_content.append(f"- **{error_type}:** {count}\n")
        report_content.append("\n")
        
        return "".join(report_content)
    
    def save(self):
        """Save the report to a file"""
        with open(self.report_path, "w") as f:
            f.write(self.generate_report())
        return self.report_path


def initialize_report():
    """Initialize a new report for this evaluation run"""
    return ReportGenerator()

# Defining methods
def get_prompt_tokens(prompt:str,
                      url:str =config.prompt_endpoint,
                      db:str=config.db_version,
                      groups:list=config.cw_groups,
                      limit:int=config.token_limit,
                      verbose:bool=False,
                      log:bool=True,
                      report:Optional[ReportGenerator]=None) -> str:

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
                     log:bool=True,
                     report:Optional[ReportGenerator]=None) -> dict:
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

    # Add metadata to report if provided
    if report:
        report.add_metadata("CropWizard API", {
            "model": model,
            "course_name": course,
            "doc_groups": group,
            "token_limit": limit,
            "temperature": 0.1,
            "system_prompt": config.cropwiz_sys_prompt[:100] + "..." if len(config.cropwiz_sys_prompt) > 100 else config.cropwiz_sys_prompt
        })

    response = requests.post(url, json=payload)
    # Error handling
    assert response.status_code == 200, f"failed to retrieve data for query_cropwizard (error_code: {response.status_code})"

    if "Error processing streaming response" in response.text:
        for attempt in range(3):
            if log:
                logging.error(f"Error ({attempt + 1}) in query_cropwizard() for question {prompt}: {response.text}")
                if report:
                    report.add_error("streaming_error", f"Attempt {attempt + 1}: {response.text}")
            sleep(0.25)
            response = requests.post(url, json=payload)
            if "Error processing streaming response" not in response.text:
                break
            elif attempt == 3:
                if log:
                    logging.error(f"Max retries reached for query_cropwizard(). Failed request to obtain streamed response for question: {prompt}.")
                    if report:
                        report.add_error("max_retries_reached", f"Failed to obtain streamed response for question: {prompt}")

    if "The CropWizard database doesn't have anything covering this exact question" in response.text:
        if log:
            logging.error(f"Vector search mismatch for question: {prompt} - not found in database")
            if report:
                report.add_error("vector_search_mismatch", f"Question not found in database: {prompt}")

    return response.text


def create_test_cases(question_answer_pairs:dict, report:Optional[ReportGenerator]=None) -> dict:
    """
    Creates a test case dictionary from a question-answer dictionary.

    Args:
        question_answer_pairs (dict): Dictionary with keys representing questions and values representing expert answers.
        report (Optional[ReportGenerator]): Report generator instance for logging.

    Returns:
        test_cases (dict): Dictionary with keys "question", "answer", "retrieved_contexts", and "ground_truth".
    """
    if report:
        report.add_metadata("Test Dataset", {
            "number_of_questions": len(question_answer_pairs),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    test_cases = {"question":[], "answer":[], "retrieved_contexts":[], "ground_truth":[]}

    for key,value in question_answer_pairs.items():
        sleep(0.25)          # Added sleep to avoid issues on the server side
        test_cases["question"].append(key)
        test_cases["answer"].append(query_cropwizard(key, report=report))
        test_cases["retrieved_contexts"].append(get_prompt_tokens(key, report=report))
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
                            ) -> dict:
    """
    Evaluates RAG performance for a set of question-answer pairs using a specified LLM judge.

    Args:
        question_answer_pairs (dict): A dictionary containing question-answer pairs for evaluation.
        judge (str, optional): A string representing the choice of LLM model to use for evaluation. Defaults to "gpt-4o-mini".
        log (bool, optional): Whether to log errors. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluation results and the path to the markdown report.
    """
    # Initialize report
    report = initialize_report()
    
    # Create test cases and preprocess them
    test_cases = create_test_cases(question_answer_pairs, report=report)
    processed_test_cases = preprocess_test_cases(test_cases)
    evaluation_dict, errors = create_dataset(processed_test_cases)
    
    # Log errors
    if errors:
        if log:
            logging.error(f"errors in dataset creation: {errors}")
        for error_entry in errors:
            if isinstance(error_entry[2], str) and "error" in error_entry[2].lower():
                error_type = "unknown_error"
                if "Error processing streaming response" in error_entry[2]:
                    error_type = "streaming_error"
                elif "Vector search mismatch" in error_entry[2]:
                    error_type = "vector_search_mismatch"
                elif "ERROR: In /getTopContexts" in error_entry[2]:
                    error_type = "context_retrieval_error"
                report.add_error(error_type)
    
    # Convert dataset to LangSmith format
    langsmith_ragas_eval = EvaluationDataset.from_list(convert_dict_to_list(evaluation_dict))

    # Initialize Langchain LLM wrapper
    llm_options = {
    # OpenAI models
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=openaiconfig.temperature),
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=openaiconfig.temperature),

    # Ollama models
    "llama3.1:8b-instruct-fp16": ChatOllama(model="llama3.1:8b-instruct-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "llama3.2:1b-instruct-fp16": ChatOllama(model="llama3.2:1b-instruct-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "llama3.2:3b-instruct-fp16": ChatOllama(model="llama3.2:3b-instruct-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "deepseek-r1:14b-qwen-distill-fp16": ChatOllama(model="deepseek-r1:14b-qwen-distill-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "qwen2.5:14b-instruct-fp16": ChatOllama(model="qwen2.5:14b-instruct-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),
    "qwen2.5:7b-instruct-fp16": ChatOllama(model="qwen2.5:7b-instruct-fp16", base_url=ollamaconfig.base_url, temperature=ollamaconfig.temperature),

    # Commented out models that could be added in the future
    # "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.1),
    # "command-r-plus": ChatCohere(model="command-r-plus", temperature=0.1),
    # "gemini-2-flash": ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.1),
    # "llama3-70b": ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.1),
    }

    # Add judge metadata to report
    report.add_metadata("Judge Model", {
        "model": judge,
        "temperature": openaiconfig.temperature if "gpt" in judge else ollamaconfig.temperature,
        "version": "latest"  # Assuming latest version
    })

    # Check if the judge model is in the available options
    if judge in llm_options:
        evaluator_llm = LangchainLLMWrapper(llm_options[judge])
    else:
        if log:
            logging.error(f"Model '{judge}' not found in available models. Reverting to default model (gpt-4o-mini).")
            if report:
                report.add_error("model_not_found", f"Model '{judge}' not found, using gpt-4o-mini instead")
        evaluator_llm = LangchainLLMWrapper(llm_options["gpt-4o-mini"])

    # Run evaluation
    results = ragas_eval(
        dataset=langsmith_ragas_eval,
        metrics=[
            metrics.ContextPrecision(),
            metrics.ContextRecall(),
            metrics.AnswerRelevancy(),
            metrics.Faithfulness(),
            metrics.FactualCorrectness(),
        ],
        llm=evaluator_llm,
    )

    # Add overall metrics to report
    overall_metrics = {
        "Context Precision": results.get("context_precision", {}).get("mean", "N/A"),
        "Context Recall": results.get("context_recall", {}).get("mean", "N/A"),
        "Answer Relevancy": results.get("answer_relevancy", {}).get("mean", "N/A"),
        "Faithfulness": results.get("faithfulness", {}).get("mean", "N/A"),
        "Factual Correctness": results.get("factual_correctness", {}).get("mean", "N/A")
    }
    report.add_metrics(overall_metrics, prefix="Overall ")
    
    # Add divider
    report.add_text("---\n")
    report.add_header("Individual Question Metrics", level=2)
    
    # Add individual question results
    for i in range(len(evaluation_dict["question"])):
        # Check if this question had an error
        had_error = False
        error_message = ""
        for error_entry in errors:
            if error_entry[0] == evaluation_dict['question'][i]:
                had_error = True
                error_message = error_entry[2]
                break
        
        # Individual metrics for this question
        metrics_for_question = {
            "Context Precision": results.get("context_precision", [])[i] if i < len(results.get("context_precision", [])) else "N/A",
            "Context Recall": results.get("context_recall", [])[i] if i < len(results.get("context_recall", [])) else "N/A",
            "Answer Relevancy": results.get("answer_relevancy", [])[i] if i < len(results.get("answer_relevancy", [])) else "N/A",
            "Faithfulness": results.get("faithfulness", [])[i] if i < len(results.get("faithfulness", [])) else "N/A",
            "Factual Correctness": results.get("factual_correctness", [])[i] if i < len(results.get("factual_correctness", [])) else "N/A"
        }
        
        report.add_question_result(i, evaluation_dict['question'][i], metrics_for_question, had_error, error_message)
    
    # Save report
    report_path = report.save()
    
    # Return both results and report path
    return {
        "results": results,
        "report_path": report_path
    }


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


def main(question_answer_pair:json, test_judge:list=["gpt-4o-mini"], ) -> dict:
    with open(question_answer_pair, "r") as imported_json:
        imported_dataset = json.load(imported_json)
    list_of_judge_tests=["gpt-4o-mini", "gpt-4o", "deepseek-r1", "llama3", "mistral"]
    if all(item in list_of_judge_tests for item in test_judge):
        if len(test_judge) == 1:
            result = single_judge_evaluation(imported_dataset, test_judge[0])
            print(f"Evaluation complete. Report saved to: {result['report_path']}")
            return result
        # elif len(test_judge) > 1:
        #     return multi_judge_evaluation(imported_dataset, test_judge)
    else:
        raise ValueError(f"One or more invalid values for test_judge: {test_judge}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--qa_file', type=str, required=True,
                        help='Path to the JSON file containing question-answer pairs')
    parser.add_argument('--judge', type=str, default="gpt-4o-mini",
                        help='Judge model to use for evaluation')

    args = parser.parse_args()

    result = main(args.qa_file, [args.judge])
    # print(f"Overall metrics:")
    # for metric, value in result['results'].items():
    #     if isinstance(value, dict) and 'mean' in value:
    #         print(f"- {metric}: {value['mean']:.4f}")
