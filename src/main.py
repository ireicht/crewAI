#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from mycrew import Mytestcrewa1

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
        'max_number_of_prompts': '10',
        'current_year': str(datetime.now().year)
    }
    
    try:
        Mytestcrewa1().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

run()