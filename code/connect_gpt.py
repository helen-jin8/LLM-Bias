import openai
import numpy as np
import pandas as pd

# don't upload this key to github!!!
openai.api_key = ""

# embedding model used: "text-embedding-ada-002"

def embedding(token_list):
  model = "text-embedding-ada-002"
  embeddings = []
  for token in token_list:
    embedding = openai.Embedding.create(input=[token], model=model).data[0].embedding
    embeddings.append(embedding)
  return embeddings

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def cosine_matrix(target, attribute):
  # Initialize a matrix to store cosine similarities
  cosine_matrix = np.zeros((len(targets), len(attributes)))

  # Calculate cosine similarity between each target and attribute word
  for i, target in enumerate(target_embeddings):
      for j, attribute in enumerate(attribute_embeddings):
          cosine_matrix[i][j] = cosine_similarity(target, attribute)

  return pd.DataFrame(cosine_matrix, index=targets, columns=attributes)
    

def weat_score(target_embeddings_1, target_embeddings_2, attribute_embeddings_1, attribute_embeddings_2):
    def mean_attribute_similarity(target_embedding, attribute_embeddings):
        return np.mean([cosine_similarity(target_embedding, attribute_embedding) for attribute_embedding in attribute_embeddings])

    # Calculate differential association of the two target sets with the attribute sets
    s_1 = sum([mean_attribute_similarity(target_embedding, attribute_embeddings_1) - mean_attribute_similarity(target_embedding, attribute_embeddings_2) for target_embedding in target_embeddings_1])
    s_2 = sum([mean_attribute_similarity(target_embedding, attribute_embeddings_1) - mean_attribute_similarity(target_embedding, attribute_embeddings_2) for target_embedding in target_embeddings_2])

    weat_score = s_1 - s_2

    # Calculate effect size
    all_embeddings = target_embeddings_1 + target_embeddings_2
    all_associations = [mean_attribute_similarity(target_embedding, attribute_embeddings_1) - mean_attribute_similarity(target_embedding, attribute_embeddings_2) for target_embedding in all_embeddings]
    std_dev = np.std(all_associations)
    effect_size = weat_score / std_dev

    return weat_score, effect_size


# Cosine similarity test case
targets = ['Engineer', 'Nurse', 'Teacher', 'Doctor', 'Carpenter', 'Driver','Entrepreneur']
attributes = ['he', 'she', 'man', 'woman', 'mother', 'father', 'non-binary']

target_embeddings = embedding(targets) 
attribute_embeddings = embedding(attributes) 

print(cosine_matrix(target_embeddings, attribute_embeddings))

# WEAT test case
career_words = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business']
family_words = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding']
male_names = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff']
female_names = ['Amy', 'Emily', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Anna']

target_1 = embedding(career_words)
target_2 = embedding(family_words)
attribute_1 = embedding(male_names)
attribute_2 = embedding(female_names)

score, effect_size = weat_score(target_1, target_2, attribute_1, attribute_2)

print(f"WEAT Score: {score}, Effect Size: {effect_size}")



######################################################################################
# Still working on this part !!!

fine_tuned_model_id = "gpt-3.5-turbo"

def generate_response(prompt):
    response = openai.Completion.create(
        model=fine_tuned_model_id,  # Use your fine-tuned model
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()


# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
