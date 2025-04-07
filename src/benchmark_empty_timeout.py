import subprocess
import datetime
from igi_helper import write_log, get_variable_value, format_duration, set_benchmark_session_id_mod, set_benchmark_base_path, set_benchmark_log_file_path, set_benchmark_crew_iteration
from collections import Counter
import re
import os

### START THIS SCRIPT in PARENT DIR of src/...


do_only_call_summarize = False

# Define the number of iterations
iterations = 3



timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
set_benchmark_session_id_mod(f"{timestamp}")


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Extract the directory name from the file path
current_dir = os.path.dirname(current_file_path)

benchmark_results_dir = os.path.join(current_dir,"benchmark_results")
os.makedirs(benchmark_results_dir,exist_ok=True)

logfile_name = f"benchmark_{timestamp}.log"
logfile_path = os.path.join(benchmark_results_dir,logfile_name)
set_benchmark_base_path(f"{benchmark_results_dir}")
set_benchmark_log_file_path(f"{logfile_path}")

# self._remember_format_after_usages: int = 20

tool_format_remember_iteration = get_variable_value('src/crewai/tools/tool_usage.py','self._remember_format_after_usages: int')
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

        if result.returncode == 0:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated. Duration {formatted_duration}")
        elif result.returncode == 1:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated due to timeout. Duration {formatted_duration}")
        elif result.returncode == 2:
            write_log(f"{logfile_path}",f"Iteration {it_log_cnt} terminated due to timeout. Duration {formatted_duration}")
    except subprocess.CalledProcessError as e:
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        # Format the duration in HH:MM:SS format
     
        formatted_duration = format_duration(duration)
        write_log(f"{logfile_path}",f"Iteration {it_log_cnt} duration {formatted_duration} failed with error code {e.returncode}")
    
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

def list_files(directory, timestamp):
    files = []
    for file in os.listdir(directory):
        if file.startswith("benchmark_" + timestamp) and "_action_call_" in file:
            files.append(file)
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


def tool_usage_details(benchmark_results_dir, timestamp):
    tool_files = list_files(benchmark_results_dir, timestamp)
    for tool_logfile in tool_files:
        tool_logfile_path = os.path.join(benchmark_results_dir, tool_logfile)
        tool_counts, input_counts, total_tool_count, total_input_count = summarize_tool_usage(tool_logfile_path)





#filename pattern for toolusage logfile benchmark_{BENCHMARK_SESSION_ID}_action_call.log
# tool_logfile_path = os.path.join(benchmark_results_dir,f"benchmark_{timestamp}_action_call.log")
# tool_logfile_path = "/Users/reicht/Developer/onTheGo/crewai_repo_dev/repo_code_dev/crewAIreicht/src/benchmark_results/benchmark_05-04-2025_18-25-13_action_call.log"



summarize_logfile(f"{logfile_path}")
tool_usage_details(benchmark_results_dir, timestamp)