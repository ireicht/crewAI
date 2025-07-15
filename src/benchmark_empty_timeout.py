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
import json

### START THIS SCRIPT in PARENT DIR of src/...


do_only_call_summarize = False

# Define the number of iterations
iterations = 3

timestamp = "26-05-2025_12-26-51" #set for debugging purpose, is ignored when do_only_call_summarize=False
if not do_only_call_summarize:
    reset_benchmark_tmp_file()
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

print(f"Timestamp Start-Session_ID: {timestamp}")
set_benchmark_session_id_mod(f"{timestamp}")


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Extract the directory name from the file path
current_dir_path = os.path.dirname(current_file_path)

# ToDo CONFIG: setup working dir and adjust paths accordingly
working_dir = current_dir_path


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
        result = subprocess.run(['python', 'src/main.py'], check=True, capture_output=True)
        
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

    # Print the summary
    print("==== SUMMARY ====")
    print(f"Iterations Total: {total_iterations} (100%)")

    for term, count in bnchmrk_counter.items():
        percentage = (count / total_iterations) * 100
        print(f"sum_{term} = {count} ({percentage:.0f}%)")

    # user_input = input(f"Write Summary to {logfile_path}?[(y)es]:")
    user_input = "y"
    if user_input == "y":
        write_log(f"{logfile_path}", "==== SUMMARY ====")
        for term, count in bnchmrk_counter.items():
            percentage = (count / total_iterations) * 100
            write_log(f"{logfile_path}", f"sum_{term} = {count} ({percentage:.0f}%)")

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


summarize_logfile(f"{logfile_path}")

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
    finished_total = len(tool_results)
    if finished_total <= 0:
        print(f"\nNo results to evaluate. Num_ool_results: {finished_total}")
        return
    
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


# get task_names and check which ones to analyse
# analyse only tasks where we find an "expected_output_<session_id>_<task_name>....log file"
'''
Sample of task_list_dicts
"task_details": [
        "TASK_NAME:search_terms TASK_MODEL_NAME:openai/granite-3.2-8b-instruct TASK_MODEL_TEMP:0.0",
        "TASK_NAME:web_searching TASK_MODEL_NAME:openai/meta-llama-3.1-8b-instruct TASK_MODEL_TEMP:0.0"
    ],
'''
task_list_dicts = get_benchmark_task_details()
for task_dict in task_list_dicts:
    print(task_dict)
    task_name = task_dict.get("TASK_NAME")
    print(task_name)

    # process all answers of task
    tool_results = tool_usage_details(benchmark_logs_dir_path, timestamp, process_finished_calls_only=False, task_name=task_name)
    make_stats_of_results(tool_results=tool_results, description_str="ANALYSIS of ALL LLM answer ATTEMPTS")

    #process only answers from LLM stable behaviour 
    tool_results = tool_usage_details(benchmark_logs_dir_path, timestamp, process_finished_calls_only=True, task_name=task_name)
    make_stats_of_results(tool_results=tool_results, description_str="ANALYSIS of stable LLM answer attempts (sum_bnchmrk_successfully_finished)")