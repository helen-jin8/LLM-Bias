
import openai

# Define a function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Define a function to save content to a file
def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

############ Set up API Key ######################################

api_key_file = '/Users/jinjiahui/Desktop/b-school lab/api_key.txt'

with open(api_key_file, 'r') as file:
    openai.api_key = file.readline().strip()

#####################################################################

# Using the provided file_id
file_id = "file-beMUSEWTT1o3B05L5ebpmHiJ"
model_name = "gpt-3.5-turbo" # Or another base model if you prefer

response = openai.FineTuningJob.create(
    training_file=file_id,
    model=model_name
)

job_id = response['id']
print(f"Fine-tuning job created successfully with ID: {job_id}")
