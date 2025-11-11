'''
Author: Ignaz Reicht
Copyright: Ignaz Reicht (2025)

NOTE: If you change the location of this file, 
make sure to adapt the variable current_file_path and its references to other paths
'''

import subprocess
import datetime
from igi_helper import write_log, get_variable_value, format_duration, set_benchmark_session_id_mod, set_benchmark_base_path, set_benchmark_log_file_path, set_benchmark_crew_iteration, append_finished_crew_iteration, reset_benchmark_tmp_file, get_finished_crew_iterations, get_benchmark_task_details, get_benchmark_logs_dir_path, get_benchmark_base_path
from collections import Counter
import re
import os
import filecmp

### START THIS SCRIPT in PARENT DIR of src/...

benchmark_compiled_results_filepath=os.path.join(os.path.expanduser('~'), 'Nextcloud','public','LLM_benchmark','Benchmark_Overview.md' )
# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Extract the directory name from the file path
current_dir_path = os.path.dirname(current_file_path)

# ToDo CONFIG: setup working dir and adjust paths accordingly
working_dir = current_dir_path

do_only_call_summarize = False

# Define the number of iterations
iterations = 10

timestamp = "04-08-2025_17-14-59" #set for debugging purpose, is ignored when do_only_call_summarize=False
timestamp = "05-08-2025_14-27-46" #set for debugging purpose, is ignored when do_only_call_summarize=False
timestamp = "08-08-2025_17-23-39" #set for debugging purpose, is ignored when do_only_call_summarize=False
if not do_only_call_summarize:
    reset_benchmark_tmp_file()
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

print(f"Timestamp Start-Session_ID: {timestamp}")
set_benchmark_session_id_mod(f"{timestamp}")





benchmark_results_dir = os.path.join(working_dir,"benchmark_results")
os.makedirs(benchmark_results_dir,exist_ok=True)
set_benchmark_base_path(f"{benchmark_results_dir}")
benchmark_logs_dir_path = get_benchmark_logs_dir_path()
os.makedirs(benchmark_logs_dir_path, exist_ok=True)

logfile_name = f"benchmark_{timestamp}.log"
logfile_path = os.path.join(get_benchmark_logs_dir_path(),logfile_name)
set_benchmark_log_file_path(f"{logfile_path}")

# self._remember_format_after_usages: int = 20
toolUsage_file_path = os.path.join(current_dir_path,'crewai/tools/tool_usage.py')
tool_format_remember_iteration = get_variable_value(toolUsage_file_path,'self._remember_format_after_usages: int')
# Generate a timestamp in the desired format: DD-MM-YYYY_HH-MM-SS

bench_start = datetime.datetime.now()
# Loop to call main.py 'iterations' times
for i in range(iterations):
    if do_only_call_summarize:
        break

    # Start time for each iteration
    it_log_cnt = i+1
    start_time = datetime.datetime.now()
    try:
        # Run the Python script
        write_log(f"{logfile_path}",f"Iteration Status: {it_log_cnt} / {iterations}")
        set_benchmark_crew_iteration(f"{it_log_cnt}")
        main_exec_file = os.path.join(current_dir_path,'main.py')
        result = subprocess.run(['python', main_exec_file], check=True, capture_output=True)
        
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        formatted_duration = format_duration(duration)

        # when using exit(#)
        if result.returncode == 0:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated. Duration {formatted_duration}")
            append_finished_crew_iteration(it_log_cnt)
        elif result.returncode == 1:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated due to timeout. Duration {formatted_duration}")
        elif result.returncode == 2:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated due to empty response. Duration {formatted_duration}")
    except subprocess.CalledProcessError as e:
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        # Format the duration in HH:MM:SS format
     
        formatted_duration = format_duration(duration)
        # when using sys.exit(#)
        if e.returncode == 0:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated (sys). Duration {formatted_duration}")
        elif e.returncode == 1:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated (sys) due to timeout. Duration {formatted_duration}")
        elif e.returncode == 2:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated (sys) due to empty response. Duration {formatted_duration}")
        else:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} duration {formatted_duration} failed with unknown (sys) error code {e.returncode}")
    
bench_end = datetime.datetime.now()
duration = bench_end - bench_start
formatted_duration = format_duration(duration)
print(f"TOTAL Duration: {formatted_duration}")
write_log(f"{logfile_path}",f"+++ TOTAL Duration (HH:MM:SS): {formatted_duration} ++++")

set_benchmark_session_id_mod(f"''")
set_benchmark_log_file_path(f"''")

def summarize_logfile(logfile_path):
    summary = {}

    # Regular expression to match lines containing 'bnchmrk'
    pattern = re.compile(r'\b(bnchmrk_[^:]+)')

    # Initialize a counter to count occurrences of each term starting with 'bnchmrk'
    bnchmrk_counter = Counter()

    # Open and read the logfile
    with open(logfile_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                term = match.group(0)
                bnchmrk_counter[term] += 1

    total_iterations = sum(bnchmrk_counter.values())
    summary["crew_sum_bnchmrk_iterations"] = {f"{timestamp}":f"{total_iterations}"}

    # Print the summary
    print("==== SUMMARY ====")
    print(f"Iterations Total: {total_iterations} (100%)")

    '''Sample
    sum_bnchmrk_llm_empty_response = 1 (33%)
    sum_bnchmrk_successfully_finished = 2 (67%)
    '''
    for term, count in bnchmrk_counter.items():
        percentage = (count / total_iterations) * 100
        print(f"sum_{term} = {count} ({percentage:.0f}%)")
        summary[f"crew_sum_{term}#"] = {f"{timestamp}":f"{count}"}
        summary[f"crew_sum_{term}%"] = {f"{timestamp}":f"{percentage:.0f}%"}

    # user_input = input(f"Write Summary to {logfile_path}?[(y)es]:")
    user_input = "y"
    if user_input == "y":
        write_log(f"{logfile_path}", "==== SUMMARY ====")
        for term, count in bnchmrk_counter.items():
            percentage = (count / total_iterations) * 100
            write_log(f"{logfile_path}", f"sum_{term} = {count} ({percentage:.0f}%)")
    
    return summary

# return the logfiles of named task
def list_finished_crew_files(directory, session_id, process_finished_calls_only=True, task_name=""):
    files = []
    # Get finished_iterations from benchmak_tmp.json file only if needed.
    '''
    Sample
       "finished_crew_iteration": [
        1,
        2
    ]'''
    finished_iterations = get_finished_crew_iterations() if process_finished_calls_only else None

    # Construct regex pattern to match filenames like:
    # "benchmark_01010101_action_call_it_<number>.log"
    pattern = re.compile(r"^benchmark_" + re.escape(session_id) + re.escape(f"_TASK_NAME_{task_name}") + r"_action_call_it_(\d+)\.log$")

    for filename in os.listdir(directory):
        # Use regex matching to check the filename and extract the number.
        match = pattern.match(filename)
        if match:
            iteration = int(match.group(1))
            if process_finished_calls_only:
                if iteration in finished_iterations:
                    files.append(filename)
            else:
                files.append(filename)
    return files

def list_finished_task_result_files(directory, session_id, process_finished_calls_only=True, task_name=""):
    files = []
    # Get finished_iterations from benchmak_tmp.json file only if needed.
    '''
    Sample
       "finished_crew_iteration": [
        1,
        2
    ]'''
    finished_iterations = get_finished_crew_iterations() if process_finished_calls_only else None

    # Construct regex pattern to match filenames like:
    # "benchmark_01010101_action_call_it_<number>.log"
    pattern = re.compile(r"^benchmark_" + re.escape(session_id) + re.escape(f"_TASK_NAME_{task_name}") + r"_action_call_it_(\d+)\.log$")

    for filename in os.listdir(directory):
        # Use regex matching to check the filename and extract the number.
        match = pattern.match(filename)
        if match:
            iteration = int(match.group(1))
            if process_finished_calls_only:
                if iteration in finished_iterations:
                    files.append(filename)
            else:
                files.append(filename)
    return files

def summarize_tool_usage(file_path):
    # Initialize dictionaries to store counts
    tool_counts = {}
    input_counts = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            if "AGENT_USED_TOOL:" in line:
                tool = line.split("AGENT_USED_TOOL:")[1].strip()
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            elif "AGENT_USED_TOOL_INPUT:" in line:
                input_query = line.split("AGENT_USED_TOOL_INPUT:")[1].strip()
                input_counts[input_query] = input_counts.get(input_query, 0) + 1
    
    # Print results for AGENT_USED_TOOL
    print("Tools used and their counts:")
    total_tool_count = 0
    for tool, count in tool_counts.items():
        print(f"{tool}: {count}")
        total_tool_count += count
    print(f"Total tools listed (AGENT_USED_TOOL): {total_tool_count}\n")
    
    # Print results for AGENT_USED_TOOL_INPUT
    print("Queries used and their counts:")
    total_input_count = 0
    for query, count in input_counts.items():
        print(f"{query}: {count}")
        total_input_count += count
    print(f"Total queries listed (AGENT_USED_TOOL_INPUT): {total_input_count}\n")
    
    # Return the counts for further processing if needed
    return tool_counts, input_counts, total_tool_count, total_input_count

# Logfile specific function
def parse_tool_use_log_file(file_path):
    """
    Parses a log file and returns a list of tuples (query, tool_used).

    The log file is expected to have blocks like:
        [TIMESTAMP]
        AGENT_USED_TOOL:<tool>
        AGENT_USED_TOOL_INPUT:{"query": "<query_text>"}
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex explanation:
    tools = re.findall(r'AGENT_USED_TOOL:(.*?)\n', content)
    inputs = re.findall(r'AGENT_USED_TOOL_INPUT:(.*)', content)
    
    result = []
    for tool, input in zip(tools, inputs):
        result.append((tool.strip(),input))
        
    return result

# This function is specific to the ToolUsage Outputformat of the logfiles
def crosscheck_tool_use_benchmark(expected_file="expected_output_benchmark_A.txt",
                         benchmark_file="benchmark_ts_action_call_it3.log"):
    """
    Cross-checks the expected benchmark log against the actual benchmark output.

    It checks:
      - That each expected query is present in the benchmark output.
      - In case of duplicates, counts extra occurrences.
      - In case of missing queries, counts how many are missing.
      - That each query uses the correct AGENT_USED_TOOL as expected.
    
    Parameters:
      expected_file (str): Filename for the expected output.
      benchmark_file (str): Filename for the actual benchmark output.
      
    Returns:
      dict: A dictionary with the counts of missing queries, duplicate queries, 
            occurrences where the wrong tool was used, and any unexpected queries.
    """
    # Parse both files
    expected_logs = parse_tool_use_log_file(expected_file)
    benchmark_logs = parse_tool_use_log_file(benchmark_file)
    
    # Build dictionary for expected queries: query -> dict(tool, count)
    expected_dict = {}
    for tool, query in expected_logs:
        if query not in expected_dict:
            expected_dict[query] = {"tool": tool, "count": 0}
        expected_dict[query]["count"] += 1
    
    # Build dictionary for benchmark queries: query -> list of tools used
    benchmark_dict = {}
    for tool, query in benchmark_logs:
        if query not in benchmark_dict:
            benchmark_dict[query] = []
        benchmark_dict[query].append(tool)
    
    # Initialize result counters
    missing_queries = {}
    duplicate_queries = {}
    wrong_tool_usage = {}

    # Check expected queries against the benchmark log
    for query, exp in expected_dict.items():
        expected_count = exp["count"]
        expected_tool = exp["tool"]
        actual_tools = benchmark_dict.get(query, [])
        actual_count = len(actual_tools)
        
        # Count missing occurrences if actual frequency is less than expected
        if actual_count < expected_count:
            missing_queries[query] = expected_count - actual_count
        
        # Count duplicates if actual frequency is more than expected
        if actual_count > expected_count:
            duplicate_queries[query] = actual_count - expected_count
        
        # Check for wrong tool usage and record the wrong tool names with counts
        wrong_tools = {}
        for tool in actual_tools:
            if tool != expected_tool:
                wrong_tools[tool] = wrong_tools.get(tool, 0) + 1
        if wrong_tools:
            wrong_tool_usage[query] = wrong_tools

    # Optionally, check for extra (unexpected) queries present in the benchmark log
    unexpected_queries = {}
    for query, tools in benchmark_dict.items():
        if query not in expected_dict:
            unexpected_queries[query] = len(tools)
    
    # Produce the final result report as a dictionary
    result = {
        "missing_queries": missing_queries,
        "duplicate_queries": duplicate_queries,
        "wrong_tool_usage": wrong_tool_usage,
        "unexpected_queries": unexpected_queries
    }
    
    return result




'''
Logfile specific function

Sample returned datastructure. 
results:
{
  "missing_queries": {
    "{\"query\": \"History of LLMs from 2010 until 2024\"}": 1,
    "{\"query\": \"Latest jailbreak of LLM 2024\"}": 1
  },
  "duplicate_queries": {},
  "wrong_tool_usage": {},
  "unexpected_queries": {
    "{\"query\": \"History of LLMs from 2010 until 2025\"}": 1,
    "{\"query\": \"Latest jailbreak of LLM 2025\"}": 1
  }
}
'''
def tool_usage_details(benchmark_logs_dir, session_id, process_finished_calls_only=True, task_name=""):
    task_logfiles = list_finished_crew_files(benchmark_logs_dir, session_id, process_finished_calls_only=process_finished_calls_only, task_name=task_name)
    
    # print(f"list of toolfiles of finished crew runs: {tool_files_finished}")
    results = []
    for task_log in task_logfiles:
        task_log_path = os.path.join(benchmark_logs_dir, task_log)
        try:
            expected_bench_file = os.path.join(get_benchmark_base_path(),"expected_outputs", f"benchmark_expected_output_TASK_NAME_{task_name}_actionCall.log")
            result = crosscheck_tool_use_benchmark(expected_file=expected_bench_file,benchmark_file=task_log_path)
            results.append(result)
        except Exception as e:
            print(f"TASK: {task_name}: No expected output found. Skipping comparison of retrieved output and expected output. Error: {e}")

    return results

'''
returns a list of true or false values
True: result matches expected output
False: result does not match expected output
'''
def task_result_details(task_result_info, session_id=timestamp,process_finished_calls_only=True, task_name=""):
    task_resultfiles = []
    finished_iterations = get_finished_crew_iterations() if process_finished_calls_only else None
    
    # Filter and print elements with the desired timestamp
    filtered_elements = [entry for entry in task_result_info if entry["timestamp"] == session_id]

    for task_element in filtered_elements:
        task_sessionid = task_element.get("timestamp") #just get the sessionid so we don't forget it is there
        iteration = task_element.get("iteration")
        filename = task_element.get("filepath")
        if process_finished_calls_only:
            if iteration in finished_iterations:
                task_resultfiles.append(filename)
        else:
            task_resultfiles.append(filename)
    
    results = []
    for task_output_file in task_resultfiles:
        
        try:
            expected_bench_file = os.path.join(get_benchmark_base_path(),"expected_outputs", f"benchmark_expected_output_TASK_NAME_{task_name}_result.md")
            has_passed = filecmp.cmp(expected_bench_file, task_output_file)
            results.append(has_passed)
        except Exception as e:
            print(f"TASK: {task_name}: No expected output found. Skipping comparison of retrieved output and expected output. Error: {e}")

    return results


def merge_results(results_list):
    # Initialize the merged dictionary with empty dictionaries for each key.
    merged = {
        "missing_queries": {},
        "duplicate_queries": {},
        "wrong_tool_usage": {},
        "unexpected_queries": {}
    }
    
    for res in results_list:
        # Merge missing_queries, duplicate_queries, and unexpected_queries
        for key in ["missing_queries", "duplicate_queries", "unexpected_queries"]:
            for query, count in res.get(key, {}).items():
                merged[key][query] = merged[key].get(query, 0) + count
        
        # Merge wrong_tool_usage (nested dictionary)
        for query, tool_dict in res.get("wrong_tool_usage", {}).items():
            if query not in merged["wrong_tool_usage"]:
                merged["wrong_tool_usage"][query] = {}
            for tool, count in tool_dict.items():
                merged["wrong_tool_usage"][query][tool] = (
                    merged["wrong_tool_usage"][query].get(tool, 0) + count
                )
    return merged

def make_stats_of_results(tool_results, description_str="ANALYSIS of TOOL RESULTS"):
    summary = {}
   
    finished_total = len(tool_results)
    summary["tool_total_calls"] = finished_total

    if finished_total <= 0:
        print(f"\nNo results to evaluate. Num_ool_results: {finished_total}")
        return summary
    
    finished_wrong = 0
    for result in tool_results:
        for key, value in result.items():
            if value != {}:
                finished_wrong += 1
                break
    
    print(f"===== {description_str} ========")
    print(f"Number of considered attempts: ({finished_total})\n ")
    finished_correct = finished_total-finished_wrong
    print(f"PASSED: {finished_correct} ({((finished_correct / finished_total)*100):.0f}%) ")
    print(f"FAILED: {finished_wrong} ({((finished_wrong / finished_total)*100):.0f}%) ")

    write_log(f"{logfile_path}",f"--- {description_str} ---")
    write_log(f"{logfile_path}",f"Number of considered attempts: ({finished_total}) ")
    write_log(f"{logfile_path}",f"PASSED: {finished_correct} ({((finished_correct / finished_total)*100):.0f}%) ")
    write_log(f"{logfile_path}",f"FAILED: {finished_wrong} ({((finished_wrong / finished_total)*100):.0f}%) ")

    merged_tool_results = merge_results(tool_results)

    print("\n=============\nResult details:")

    for key, errors in merged_tool_results.items():
        print(f"\n{key}:")
        if errors:
            for query, count in errors.items():
                print(f"  - '{query}': {count}")
        else:
            print("  None")
    
    
    summary["tool_wrong_usage"] = finished_wrong
    summary["tool_right_usage"] = finished_correct
    summary["tool_usage_details"] = merged_tool_results
    return summary

def extract_task_info(filename):
    # Define the regex pattern for matching the filename
    pattern = r'task_benchmark_(?P<timestamp>\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})-task_name_(?P<task_name>[\w_]+)(_it_(?P<iteration>\d+)).md'

    # Match the filename against the pattern
    match = re.match(pattern, filename)
    if not match:
        return None

    # Extract the timestamp, task name, and iteration (if present)
    timestamp = match.group('timestamp')
    task_name = match.group('task_name')
    iteration = match.group('iteration')

    return {
        'timestamp': timestamp,
        'task_name': task_name,
        'iteration': int(iteration) if iteration else None
    }

def scan_directory_for_task_files(directory):
    # Initialize the result dictionary
    task_benchmark_results = {}

    # Scan the directory for "*.md" files
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            # Get the full file path
            filepath = os.path.join(directory, filename)

            # Extract task information from the filename
            task_info = extract_task_info(filename)
            if task_info:
                # Get the task name
                task_name = task_info['task_name']

                # Initialize a list for this task if it doesn't exist
                if task_name not in task_benchmark_results:
                    task_benchmark_results[task_name] = []

                # Append the result to the list for this task
                task_benchmark_results[task_name].append({
                    'timestamp': task_info['timestamp'],
                    'iteration': task_info['iteration'],
                    'filepath': filepath
                })

    return task_benchmark_results

summary_crew_iterations = summarize_logfile(f"{logfile_path}")

# get the task results of .md files
'''
Sample structure of task_results
{
  "web_searching": [
    {
      "timestamp": "16-07-2025_20-35-30",
      "iteration": 1,
      "filepath": "/Users/reicht/Developer/onTheGo/crewai_repo_dev/repo_code_dev/crewAIreicht/src/benchmark_results/outputWebSearch/task_benchmark_16-07-2025_20-35-30-task_name_web_searching_it_1.md"
    }
  ],
  "search_terms": [
    {
      "timestamp": "16-07-2025_20-35-30",
      "iteration": 1,
      "filepath": "/Users/reicht/Developer/onTheGo/crewai_repo_dev/repo_code_dev/crewAIreicht/src/benchmark_results/outputWebSearch/task_benchmark_16-07-2025_20-35-30-task_name_search_terms_it_1.md"
    },
    {
      "timestamp": "16-07-2025_20-35-30",
      "iteration": 2,
      "filepath": "/Users/reicht/Developer/onTheGo/crewai_repo_dev/repo_code_dev/crewAIreicht/src/benchmark_results/outputWebSearch/task_benchmark_16-07-2025_20-35-30-task_name_search_terms_it_2.md"
    }
  ]
}
'''

task_result_directory_path = os.path.join(get_benchmark_base_path(),'task_result_outputDir')
'''
Sample of task_results:
timestamp equals sessionID
{
'web_searching': [{'timestamp': '25-07-2025_19-51-11', 'iteration': 1, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_web_searching_it_1.md'}, {'timestamp': '25-07-2025_19-51-11', 'iteration': 3, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_web_searching_it_3.md'},...
'search_terms': [{'timestamp': '25-07-2025_19-51-11', 'iteration': 1, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_search_terms_it_1.md'}, {'timestamp': '25-07-2025_19-51-11', 'iteration': 3, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_search_terms_it_3.md'},...
}
'''
task_results = scan_directory_for_task_files(task_result_directory_path)
# print(f"task_results:{task_results}")
# get task_names and check which ones to analyse
# analyse only tasks where we find an "expected_output_<session_id>_<task_name>....log file"
'''
Sample of task_list_dicts
"task_details": [
        "TASK_NAME:search_terms TASK_MODEL_NAME:openai/granite-3.2-8b-instruct TASK_MODEL_TEMP:0.0",
        "TASK_NAME:web_searching TASK_MODEL_NAME:openai/meta-llama-3.1-8b-instruct TASK_MODEL_TEMP:0.0"
    ],
'''

# get misc info
# ToDo Processor

task_list_dicts = get_benchmark_task_details()

all_results_compiled = {}
# summary_crew_iterations["crew_session_id"] = f"{timestamp}"
summary_crew_iterations["crew_duration"] = {f"{timestamp}":f"{formatted_duration}"}
# summary_crew_iterations["crew_note"] = f"Symbol *: includes results from unstable model behaviour"
all_results_compiled['crew']=summary_crew_iterations
# all_results_compiled['tasks']=[]
for task_dict in task_list_dicts:
    all_task_results_compiled = {}
    # collected information:
    # task_dict: "TASK_NAME:search_terms TASK_MODEL_NAME:openai/granite-3.2-8b-instruct TASK_MODEL_TEMP:0.0",...
    # task_result_info: 'web_searching': [{'timestamp': '25-07-2025_19-51-11', 'iteration': 1, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_web_searching_it_1.md'}, {'timestamp': '25-07-2025_19-51-11', 'iteration': 3, 'filepath': '/my/path/task_benchmark_25-07-2025_19-51-11-task_name_web_searching_it_3.md'},...

    print(task_dict)
    task_name = task_dict.get("TASK_NAME")
    print(f"Task Name: {task_name}")
    task_result_info = task_results.get(task_name)
    # print(f"Task Results: \n{task_result_info}")


    
    # all_task_results_compiled["task_name"] = task_name
    all_task_results_compiled["task_model_name"] = {f"{timestamp}":task_dict.get("TASK_MODEL_NAME", "na")}
    all_task_results_compiled["task_model_temp"] = {f"{timestamp}":task_dict.get("TASK_MODEL_TEMP", "na")}


    # process all answers of task
    tool_results_logfiles = tool_usage_details(benchmark_logs_dir_path, timestamp, process_finished_calls_only=False, task_name=task_name)
    tool_summary = make_stats_of_results(tool_results=tool_results_logfiles, description_str="ANALYSIS of ALL LLM answer ATTEMPTS")
    #print(f"tool_summary:{tool_summary}")
    tool_taskoutput_results = task_result_details(task_result_info, session_id=timestamp,process_finished_calls_only=False, task_name=task_name)
    #print(f"task_output_results:{tool_taskoutput_results}")
    # Count the number of True and False values
    true_count = sum(tool_taskoutput_results)
    false_count = len(tool_taskoutput_results) - true_count

    print(f"Task output matching: {true_count}")
    print(f"Task output mismatch: {false_count}")


    #process only answers from LLM stable behaviour 
    tool_results_logfiles_stable = tool_usage_details(benchmark_logs_dir_path, timestamp, process_finished_calls_only=True, task_name=task_name)
    tool_summary_stable = make_stats_of_results(tool_results=tool_results_logfiles_stable, description_str="ANALYSIS of stable LLM answer attempts (sum_bnchmrk_successfully_finished)")
    #print(f"tool_summary_stable:{tool_summary_stable}")
    tool_taskoutput_results_stable = task_result_details(task_result_info, session_id=timestamp,process_finished_calls_only=True, task_name=task_name)
    #print(f"task_utput_results_stable:{tool_taskoutput_results_stable}")
    # Count the number of True and False values
    true_count_stable = sum(tool_taskoutput_results_stable)
    false_count_stable = len(tool_taskoutput_results_stable) - true_count_stable

    print(f"Task output matching: {true_count_stable}")
    print(f"Task output mismatch: {false_count_stable}")

    # all_task_results_compiled["task_tool_usage"] = value if value != 0 else "-"
    task_tool_usage_calls = tool_summary.get("tool_total_calls", "na")
    task_tool_usage_right = tool_summary.get("tool_right_usage", "na")
    task_tool_usage_wrong = tool_summary.get("tool_wrong_usage", "na")
    #task_tool_usage_details = tool_summary.get("tool_usage_details", "na")

    all_task_tool_usage_calls = task_tool_usage_calls if task_tool_usage_calls != 0 else "-"
    all_task_tool_usage_pass_p = f"{(task_tool_usage_right / task_tool_usage_calls *100):.0f}%" if task_tool_usage_right != "na" else "-"
    all_task_tool_usage_fail_p = f"{(task_tool_usage_wrong / task_tool_usage_calls *100):.0f}%" if task_tool_usage_wrong != "na" else "-"
    all_task_tool_usage_pass = task_tool_usage_right if task_tool_usage_right != "na" else "-"
    all_task_tool_usage_fail = task_tool_usage_wrong if task_tool_usage_wrong != "na" else "-"
    

    task_tool_usage_calls_stable = tool_summary_stable.get("tool_total_calls", "na")
    task_tool_usage_right_stable = tool_summary_stable.get("tool_right_usage", "na")
    task_tool_usage_wrong_stable = tool_summary_stable.get("tool_wrong_usage", "na")

    all_task_tool_usage_calls_stable = task_tool_usage_calls_stable if task_tool_usage_calls_stable != 0 else "-"
    all_task_tool_usage_pass_p_stable = f"{(task_tool_usage_right_stable / task_tool_usage_calls_stable *100):.0f}%" if task_tool_usage_right_stable != "na" else "-"
    all_task_tool_usage_fail_p_stable = f"{(task_tool_usage_wrong_stable / task_tool_usage_calls_stable *100):.0f}%" if task_tool_usage_wrong_stable != "na" else "-"
    all_task_tool_usage_pass_stable = task_tool_usage_right_stable if task_tool_usage_right_stable != "na" else "-"
    all_task_tool_usage_fail_stable = task_tool_usage_wrong_stable if task_tool_usage_wrong_stable != "na" else "-"

    all_task_results_compiled["task_tool_usage_calls"] = {f"{timestamp}":f"{all_task_tool_usage_calls_stable} (*:{all_task_tool_usage_calls})"}
    all_task_results_compiled["task_tool_usage_pass%"] = {f"{timestamp}":f"{all_task_tool_usage_pass_p_stable} (*:{all_task_tool_usage_pass_p})"}
    all_task_results_compiled["task_tool_usage_fail%"] = {f"{timestamp}":f"{all_task_tool_usage_fail_p_stable} (*:{all_task_tool_usage_fail_p})"}
    all_task_results_compiled["task_tool_usage_pass#"] = {f"{timestamp}":f"{all_task_tool_usage_pass_stable} (*:{all_task_tool_usage_pass})"}
    all_task_results_compiled["task_tool_usage_fail#"] = {f"{timestamp}":f"{all_task_tool_usage_fail_stable} (*:{all_task_tool_usage_fail})"}


    # task_output
    task_tool_output_matches_stable = true_count_stable if len(tool_taskoutput_results_stable) != 0 else "-"
    task_tool_output_matches_stable_p = f"{((task_tool_output_matches_stable / len(tool_taskoutput_results_stable))*100):.0f}%" if len(tool_taskoutput_results_stable) != 0 else "-"
    task_tool_output_matches = true_count if len(tool_taskoutput_results) != 0 else "-"
    task_tool_output_matches_p = f"{((task_tool_output_matches / len(tool_taskoutput_results))*100):.0f}%" if len(tool_taskoutput_results) != 0 else "-"
    
    task_tool_output_mismatches_stable = false_count_stable if len(tool_taskoutput_results_stable) != 0 else "-"
    task_tool_output_mismatches_stable_p = f"{((task_tool_output_mismatches_stable / len(tool_taskoutput_results_stable))*100):.0f}%" if len(tool_taskoutput_results_stable) != 0 else "-"
    task_tool_output_mismatches = false_count if len(tool_taskoutput_results) != 0 else "-"
    task_tool_output_mismatches_p = f"{((task_tool_output_mismatches / len(tool_taskoutput_results))*100):.0f}%" if len(tool_taskoutput_results) != 0 else "-"

    all_task_results_compiled["task_output_matching%"] = {f"{timestamp}":f"{task_tool_output_matches_stable_p} (*:{task_tool_output_matches_p})"}
    all_task_results_compiled["task_output_mismatch%"] = {f"{timestamp}":f"{task_tool_output_mismatches_stable_p} (*:{task_tool_output_mismatches_p})"}
    all_task_results_compiled["task_output_matching#"] = {f"{timestamp}":f"{task_tool_output_matches_stable} (*:{task_tool_output_matches})"}
    all_task_results_compiled["task_output_mismatch#"] = {f"{timestamp}":f"{task_tool_output_mismatches_stable} (*:{task_tool_output_mismatches})"}


    all_results_compiled[task_name]=all_task_results_compiled



metric_lut = {}
metric_lut["crew_sum_bnchmrk_iterations"] =             "Iterations"
metric_lut["crew_sum_bnchmrk_successfully_finished#"] = "Success(#)"
metric_lut["crew_sum_bnchmrk_successfully_finished%"] = "Success(%)"
metric_lut["crew_sum_bnchmrk_llm_empty_response#"] =    "Fail (empty llm response #)"
metric_lut["crew_sum_bnchmrk_llm_empty_response%"] =    "Fail (empty llm response %)"
metric_lut["crew_duration"] =                           "Duration"

metric_lut["task_model_name"] =         "Model Name"
metric_lut["task_model_temp'"] =        "Model Temp."
metric_lut["task_tool_usage_calls'"] =  "Tool calls"



def format_dict_humanreadable(data:dict) -> dict:
    formatted_dict = {}
    for k, v in data.items():
        
        if isinstance(v,list): #e.g. list of tasks:
            #create list instance if not yet present in formatted_dict
            tmp_list = formatted_dict.get(k,None)
            if tmp_list == None:
                formatted_dict[k] = []
            for v_dict in v:
                formatted_metric_names = {}
                for kmetric, vmetric in v_dict.items():
                    #if metric name has a human readable version, use it, otherwise keep long name
                    name_hr = metric_lut.get(kmetric,kmetric)
                    formatted_metric_names[name_hr] = vmetric
                formatted_dict[k].append(formatted_metric_names)
        elif isinstance(v, dict):
            formatted_metric_names = {}
            for kmetric, vmetric in v.items():
                #if metric name has a human readable version, use it, otherwise keep long name
                name_hr = metric_lut.get(kmetric,kmetric)
                formatted_metric_names[name_hr] = vmetric
            formatted_dict[k] = formatted_metric_names
        else:
            print("WARNING, UNKNOWN TYPE IN COMPILED RESULTS. SKIPPING LUT RENAMING")
            formatted_dict[k] = v
        
    
    return formatted_dict
            


all_results_compiled_hr = format_dict_humanreadable(all_results_compiled)

# print("\nALL_RESULTS_COMPILED HR\n")

# for k, v in all_results_compiled_hr.items():
#     print(f"{k}: {v}")
#     print("")
# print("")

# print(f"datastructure:\n{all_results_compiled_hr}")




##########

def escape(s):
    """Escape special characters in strings for markdown representation."""
    return str(s).replace('|', '\\|').replace('\n', '\\n')

def unescape(s):
    """Unescape special characters in strings parsed from markdown."""
    return s.replace('\\|', '|').replace('\\n', '\n')

def dict_to_md(data):
    """
    Convert a nested dictionary to markdown format.

    Args:
        data: A nested dictionary with the structure as described.

    Returns:
        str: A markdown-formatted string representing the dictionary.
    """
    
	# Sort top-level keys with 'crew' first
    # Extract top-level keys
	
    top_keys = list(data.keys())
	
	# Define the key to appear first in the sorted order
    key_to_prioritize = "crew"

	# Sort the keys, ensuring 'crew' comes first
    sorted_keys = sorted(top_keys, key=lambda x: (x != key_to_prioritize))

	# Reconstruct the dictionary in the desired order
    sorted_dict = {key: data[key] for key in sorted_keys}
    
    md_lines = []
    for top_key, top_value in sorted_dict.items():
        if top_key == 'crew':
            md_lines.append(f"## Crew: {top_key}")
        else:
            md_lines.append(f"## Task: {top_key}")

        # Collect all timestamps for the section
        all_timestamps = set()
        for metric in top_value.values():
            all_timestamps.update(metric.keys())
        
        # Convert set to sorted list for consistent order
        all_timestamps = sorted(all_timestamps)
        
        # Print the header row for each section only once
        md_lines.append("| Metric | " + ' | '.join(all_timestamps) + " |")
        md_lines.append("|---" + "|---" * len(all_timestamps) + "|")
        
        # Now process each metric within this section, ensuring headers are not repeated
        for metric, timestamps in top_value.items():
            md_lines.append(f"| {metric} | " + ' | '.join(escape(timestamps.get(ts, '')) for ts in all_timestamps) + " |")
    
    return '\n'.join(md_lines)

def md_to_dict(md_text):
    """
    Convert markdown-formatted string back to the original nested dictionary.

    Args:
        md_text: A string in markdown format as produced by dict_to_md.

    Returns:
        dict: The reconstructed nested dictionary.
    """
    result = {}
    current_top_key = None
    headers = []
    timestamps = []

    for line in md_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        elif line.startswith('## Task: '):
            current_top_key = line[len('## Task: '):].strip()
            result[current_top_key] = {}
        elif line.startswith('## Crew: '):
            current_top_key = line[len('## Crew: '):].strip()
            result[current_top_key] = {}
        elif line.startswith('| Metric |'):
            headers = [header.strip() for header in line.split('|')[1:-1]]
            headers.remove('Metric')

        elif '---' in line:
            continue
        else:
            if current_top_key and headers:
                metric, *values = [unescape(value.strip()) for value in line.split('|')[1:-1]]
                if metric and values:
                    result[current_top_key][metric] = dict(zip(headers, values))
    return result


def merge_dicts(dict1, dict2):
    merged = {}
    categories = set(dict1.keys()).union(set(dict2.keys()))

    for category in categories:
        metrics_old = set(dict1[category].keys()) if category in dict1 else set()
        metrics_new = set(dict2[category].keys()) if category in dict2 else set()
        all_metrics = metrics_old.union(metrics_new)

        timestamps_old = set()
        if category in dict1:
            for metric_data in dict1[category].values():
                timestamps_old.update(metric_data.keys())
        timestamps_new = set()
        if category in dict2:
            for metric_data in dict2[category].values():
                timestamps_new.update(metric_data.keys())
        all_timestamps = sorted(timestamps_old.union(timestamps_new))

        merged_category = {}
        for metric in all_metrics:
            metric_data = {timestamp: "_" for timestamp in all_timestamps}

            if category in dict1 and metric in dict1[category]:
                for timestamp, value in dict1[category][metric].items():
                    metric_data[timestamp] = value
            if category in dict2 and metric in dict2[category]:
                for timestamp, value in dict2[category][metric].items():
                    metric_data[timestamp] = value

            merged_category[metric] = metric_data
        merged[category] = merged_category
    return merged

def md_sanityCheck(filepath, expected_md):
    with open(filepath, 'r') as f:
        existing_md_sanityCheck = f.read()
    
    if existing_md_sanityCheck != expected_md:
        print(f"\nWARNING from MARKDOWN SANITY CHECK:\n...Markdown of written file and in-memory markdown not fully matching, please check manually:")
        print(f"   Markdown from file {filepath}:\n{existing_md_sanityCheck}")
        print(f"   Markdown from in-memory:\n{expected_md}")
        print(f"MARKDOWN SANITY CHECK END\n\n")
    else:
        print("MARKDOWN INFO: [matching] Written output and expected output")





if os.path.isfile(benchmark_compiled_results_filepath):
        with open(benchmark_compiled_results_filepath, 'r') as f:
            existing_md = f.read()
            reconstructed_dict = md_to_dict(existing_md)

        updated_dict = merge_dicts(reconstructed_dict, all_results_compiled_hr)
        new_md = dict_to_md(updated_dict)
        with open(benchmark_compiled_results_filepath, 'w') as f:
            f.write(new_md)
            print(f"updated MD file: {benchmark_compiled_results_filepath}")
        #sanity check
        md_sanityCheck(filepath=benchmark_compiled_results_filepath,expected_md=new_md)
        


else:
    markdown_output = dict_to_md(all_results_compiled_hr)
    with open(benchmark_compiled_results_filepath, 'w') as f:
        f.write(markdown_output)
        print(f"written MD file: {benchmark_compiled_results_filepath}")
    #sanity check
    md_sanityCheck(filepath=benchmark_compiled_results_filepath,expected_md=markdown_output)






