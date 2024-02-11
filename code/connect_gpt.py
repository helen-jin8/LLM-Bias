import openai


openai.api_key = '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'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

# Print the text of the completion
# The structure of the response may have changed in the new version, so you should verify the correct path to the content you want to print.
# If you encounter an error, check the structure of 'response' to determine the correct keys to use.
print(response['choices'][0]['message']['content'])

