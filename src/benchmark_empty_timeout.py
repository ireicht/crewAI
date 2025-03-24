import subprocess
import datetime
from igi_helper import write_log
# Define the number of iterations
iterations = 5

# Loop to call main.py 'iterations' times
for i in range(iterations):
    # Start time for each iteration
    start_time = datetime.datetime.now()
    try:
        # Run the Python script
        write_log(f"benchmark.log",f"iteration: {i}")
        result = subprocess.run(['python', 'main.py'], check=True)
        
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        # Format the duration in HH:MM:SS format
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

        write_log(f"benchmark.log",f"Iteration {i+1} terminated. Duration {formatted_duration}")
    except subprocess.CalledProcessError as e:
        # End time for each iteration
        end_time = datetime.datetime.now()
        # Calculate duration of each iteration
        duration = end_time - start_time
        # Format the duration in HH:MM:SS format
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        write_log(f"benchmark.log",f"Iteration {i+1} duration {formatted_duration} failed with error code {e.returncode}")
    
    