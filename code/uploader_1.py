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


# Assuming the file name is 'processed_data.jsonl'
with open("/Users/jinjiahui/Desktop/b-school lab/repo/LLM-Bias/training data/gender_unbiased.jsonl", "rb") as file:
    response = openai.File.create(
        file=file,
        purpose='fine-tune'
    )

file_id = response['id']
print(f"File uploaded successfully with ID: {file_id}")
