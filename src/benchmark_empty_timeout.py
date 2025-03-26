import subprocess
import datetime
from igi_helper import write_log, get_variable_value, format_duration
from collections import Counter
import re

do_only_call_summarize = False

# Define the number of iterations
iterations = 100
logfile_name = "benchmark.log"
# self._remember_format_after_usages: int = 20
tool_format_remember_iteration = get_variable_value('crewai/tools/tool_usage.py','self._remember_format_after_usages: int')

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
        write_log(f"{logfile_name}",f"Iteration Status: {it_log_cnt} / {iterations}")
        result = subprocess.run(['python', 'main.py'], check=True)
        
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time

        formatted_duration = format_duration(duration)

        write_log(f"{logfile_name}",f"Iteration {it_log_cnt} terminated. Duration {formatted_duration}")
    except subprocess.CalledProcessError as e:
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        # Format the duration in HH:MM:SS format
     
        formatted_duration = format_duration(duration)
        write_log(f"{logfile_name}",f"Iteration {it_log_cnt} duration {formatted_duration} failed with error code {e.returncode}")
    
bench_end = datetime.datetime.now()
duration = bench_end - bench_start
formatted_duration = format_duration(duration)
print(f"TOTAL Duration: {formatted_duration}")
write_log(f"{logfile_name}",f"+++ TOTAL Duration (HH:MM:SS): {formatted_duration} ++++")
          
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

    user_input = input(f"Write Summary to {logfile_name}?[(y)es]:")
    if user_input == "y":
        write_log(f"{logfile_path}", "==== SUMMARY ====")
        for term, count in bnchmrk_counter.items():
            percentage = (count / total_iterations) * 100
            write_log(f"{logfile_path}", f"sum_{term} = {count} ({percentage:.0f}%)")

summarize_logfile(f"{logfile_name}")