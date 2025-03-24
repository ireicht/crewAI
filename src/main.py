#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from mycrew import Mytestcrewa1
from mysearchcrew import MySearchCrew
from igi_helper import write_log

### disable crewai telemetry!
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['LITELLM_LOG'] = 'DEBUG'

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'max_number_of_prompts': '6',
        'min_number_of_search_queries': '6',
        'current_year': str(datetime.now().year)
    }
    
    try:
        # Mytestcrewa1().crew().kickoff(inputs=inputs)
        write_log(f"benchmark.log", f"starting_crew:")
        MySearchCrew().crew().kickoff(inputs=inputs)
        write_log(f"benchmark.log",f"bnchmrk_successfully_finished:")

    except Exception as e:
        write_log(f"benchmark.log",f"bnchmrk_other_exception_occured: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")
    

run()