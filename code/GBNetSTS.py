# %% [markdown]
# **Install libraries, dataset**

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import json
import numpy as np
from sklearn.preprocessing import normalize

# dataset_path = 'D:\GitHub\masterthesis\data\Stsbenchmark\stsbenchmark\output_2017_data.xlsx'
dataset_path = 'D:\GitHub\masterthesis\data\ViSentSim-600 - Standard-reviewed.xlsx'

# MODEL_PATH = 'd:/GitHub/masterthesis/data/llama_embedding_data.json'
MODEL_PATH = 'D:\GitHub\masterthesis\data\qwen25_embedding_data.json'

# %%
def load_dataset_and_groundtruth(dataset_path):

    #Dataset đã được review lại
    df = pd.read_excel(dataset_path)
    #Dataset đã được tách theo concept sử dụng cho độ đo jaccard và LCS
    # df2 = pd.read_excel('d:/GitHub/masterthesis/data/ViSTS_concept.xlsx')

    # Loại bỏ các dòng có giá trị NaN nếu có
    df = df.dropna(subset=['Câu 1', 'Câu 2'])
    # df2 = df2.dropna(subset=['Câu 1', 'Câu 2'])

    #Trích xuất các câu từ dataset
    all_sentences = list(df['Câu 1']) + list(df['Câu 2'])
    # all_sentences_concept = list(df2['Câu 1']) + list(df2['Câu 2'])

    ground_truth = list(df['Similarity Score'])
    return all_sentences, ground_truth

all_sentences, ground_truth = load_dataset_and_groundtruth(dataset_path)

# %% [markdown]
# **Define Layers**

# %%
def jaccard_similarity(sent1, sent2):
    #tách từ sinh viên => sinh_viên
    #trọng số ?
    # Tokenize the sentences into words
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())

    # Calculate the intersection and union of the word sets
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    # Jaccard similarity is the size of the intersection divided by the size of the union
    jaccard_score = len(intersection) / len(union) if len(union) > 0 else 0

    return jaccard_score

def LCSubstr(s1, s2):
    ws1 =  s1.split()
    ws2 =  s2.split()
    l1= len(ws1)
    l2= len(ws2)
    m= max(l1, l2)+1
    F = np.zeros( (m, m) )
    for i in range(len(ws1)):
        for j in range(len(ws2)):
            if ws1[i]==ws2[j]:
                F[i+1,j+1] = F[i,j] +1
            else:
                F[i+1,j+1]  = max(F[i,j+1] , F[i+1,j] )
    return F[l1, l2] / max(l1, l2)


# %% [markdown]
# **Compute the distribution of the dataset using the Cosine Similarity, Jaccard, and LCS metrics.**

# %%
def load_embedding_from_model(model_path):
    #Sử dụng embedding từ các mô hình
    with open(model_path, 'r') as file:
        data = json.load(file)
    embeddings = data["data"]
    return embeddings

def normalized_embeddings(embeddings):
    # Convert list to NumPy array before normalizing
    embeddings_array = np.array(embeddings)

    # Now apply normalization
    normalized_embeddings = normalize(embeddings_array, axis=1)
    return normalized_embeddings

embeddings = normalized_embeddings(load_embedding_from_model(MODEL_PATH))

# print(embeddings[0])
# print(normalized_embedding[0])

# %%
print(all_sentences[0], all_sentences[125])
print(len(all_sentences))

# %%
def get_distribution_cosine_similarity():
  # Lấy các similarity scores
  # n = len(df)
  n = len(all_sentences) // 2
  scores = []
  print("DF length: "  , n) 
  # Similarity giữa Câu 1 và Câu 2
  for i in range(n):
      scores.append(cosine_similarity([embeddings[i]], [embeddings[i+n]]))

  # Similarity trong nội bộ Câu 1
  for i in range(n):
      for j in range(i+1, n):
          scores.append(cosine_similarity([embeddings[i]], [embeddings[j]]))

  # Similarity trong nội bộ Câu 2
  for i in range(n, 2*n):
      for j in range(i+1, 2*n):
          scores.append(cosine_similarity([embeddings[i]], [embeddings[j]]))
  return scores


def get_distribution_jaccard_similarity():
  # Lấy các similarity scores
  # n = len(df)
  n = len(all_sentences) // 2
  scores = []
  print("DF length: "  , n)
  # Similarity giữa Câu 1 và Câu 2
  for i in range(n):
      scores.append(jaccard_similarity(all_sentences[i], all_sentences[i+n]))

  # Similarity trong nội bộ Câu 1
  for i in range(n):
      for j in range(i+1, n):
          scores.append(jaccard_similarity(all_sentences[i], all_sentences[j]))

  # Similarity trong nội bộ Câu 2
  for i in range(n, 2*n):
      for j in range(i+1, 2*n):
          scores.append(jaccard_similarity(all_sentences[i], all_sentences[j]))
  return scores

def get_distribution_lcs_similarity():
  # n = len(df)
  n = len(all_sentences) // 2
  # Lấy các similarity scores
  scores = []
  print("DF length: "  , n)
  # Similarity giữa Câu 1 và Câu 2
  for i in range(n):
      scores.append(LCSubstr(all_sentences[i], all_sentences[i+n]))

  # Similarity trong nội bộ Câu 1
  for i in range(n):
      for j in range(i+1, n):
          scores.append(LCSubstr(all_sentences[i], all_sentences[j]))

  # Similarity trong nội bộ Câu 2
  for i in range(n, 2*n):
      for j in range(i+1, 2*n):
          scores.append(LCSubstr(all_sentences[i], all_sentences[j]))
  return scores

# %%
cosine_sim_distribution = get_distribution_cosine_similarity()
jaccard_distribution = get_distribution_jaccard_similarity()
lcs_distribution = get_distribution_lcs_similarity()

# %%
def find_most_frequent_distribution(distribution):
    scores = np.array(distribution).flatten()

    # Create histogram
    hist, bin_edges = np.histogram(scores, bins=50)

    # Find the index of the bin with the highest count
    max_count_index = np.argmax(hist)

    # Get the center of the bin with the highest count
    most_frequent_value = (bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2

    print(f"Bin with highest distribution: {most_frequent_value}")
    print(f"Number of items in this bin: {hist[max_count_index]}")


find_most_frequent_distribution(cosine_sim_distribution)
find_most_frequent_distribution(jaccard_distribution)
find_most_frequent_distribution(lcs_distribution)


# %%
#Show plot
import numpy as np
def distribution_chart():

  scores = np.array(cosine_sim_distribution).flatten()

  scores2 = np.array(jaccard_distribution).flatten()

  scores3 = np.array(lcs_distribution).flatten()

  plt.figure(figsize=(12, 8))
  sns.histplot(scores, bins=50, kde=True)
  plt.title('Phân bố Cosine Similarity Scores')
  plt.xlabel('Cos Similarity Score')
  plt.ylabel('Số lượng')
  plt.show()


 
  plt.figure(figsize=(12, 8))
  sns.histplot(scores2, bins=50, kde=True)
  plt.title('Phân bố Jaccard Similarity Scores')
  plt.xlabel('Jaccard Score')
  plt.ylabel('Số lượng')
  plt.show()

 
  plt.figure(figsize=(12, 8))
  sns.histplot(scores3, bins=50, kde=True)
  plt.title('Phân bố LCS Similarity Scores')
  plt.xlabel('LCS Score')
  plt.ylabel('Số lượng')
  plt.show()


# %%
distribution_chart()

# %% [markdown]
# **Calculate JensenShannon**

# %%
import numpy as np
from scipy.spatial.distance import jensenshannon

dist1_flat = []
for arr in cosine_sim_distribution:
  # print(arr[0][0])
  dist1_flat.append(arr[0][0])

def safe_jensenshannon_for_lists(dist1, dist2, epsilon=1e-10):
    # Convert to numpy arrays and handle potential issues
    dist1_array = np.array(dist1, dtype=float)
    dist2_array = np.array(dist2, dtype=float)


    # Add small epsilon to prevent zero probabilities
    dist1_safe = dist1_array + epsilon
    dist2_safe = dist2_array + epsilon

    # Normalize distributions
    dist1_norm = dist1_safe / np.sum(dist1_safe)
    dist2_norm = dist2_safe / np.sum(dist2_safe)

    # Ensure non-negative and sum to 1
    dist1_norm = np.maximum(dist1_norm, 0)
    dist2_norm = np.maximum(dist2_norm, 0)
    dist1_norm /= np.sum(dist1_norm)
    dist2_norm /= np.sum(dist2_norm)

    # Calculate Jensen-Shannon divergence
    try:
        js_distance = jensenshannon(dist1_norm, dist2_norm)
        return js_distance
    except Exception as e:
        print("Calculation error:", e)
        return None

# Usage
result = safe_jensenshannon_for_lists(dist1_flat, jaccard_distribution)
print("Jensen-Shannon Distance CosJac:", result)

result2 = safe_jensenshannon_for_lists(dist1_flat, lcs_distribution)
print("Jensen-Shannon Distance CosLCS:", result2)

result3 = safe_jensenshannon_for_lists(lcs_distribution, jaccard_distribution)
print("Jensen-Shannon Distance LCSJac:", result3)

print("Sum: ", (result + result2 + result3))
print("Main: ", (result + result2 + result3)/3)
print("Sum - Main: ", (result + result2 + result3) - ((result + result2 + result3)/3))

# %% [markdown]
# **Build the Graph**

# %%
#Build the graph
def build_graph_for_layers(embeddings, all_sentences):
  #Cos-sim Graph
  cos_similarity_graph = []
  for i in range(len(embeddings)):
      for j in range(i+1, len(embeddings)):
        if (j -i) == 600:
          cos_similarity_graph.append((i, j, cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]))
           # Avoid redundant pairs (i.e., only upper triangle of the matrix)
        if cosine_similarity([embeddings[i]], [embeddings[j]]) >= 0.68:
          cos_similarity_graph.append((i, j, cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]))
  #Jaccard Graph
  jaccard_graph = []
  for i in range(len(all_sentences)):
      for j in range(i + 1, len(all_sentences)):
        if jaccard_similarity(all_sentences[i], all_sentences[j]) >= 0.07:
          jaccard_graph.append((i, j, jaccard_similarity(all_sentences[i], all_sentences[j])))
  #LCS Graph
  lcs_graph = []
  for i in range(len(all_sentences)):
      for j in range(i + 1, len(all_sentences)):
        if LCSubstr(all_sentences[i], all_sentences[j]) > 0:
          lcs_graph.append((i, j, LCSubstr(all_sentences[i], all_sentences[j])))
  return cos_similarity_graph, jaccard_graph, lcs_graph


cos_similarity_graph, jaccard_graph, lcs_graph = build_graph_for_layers(embeddings, all_sentences)
# cos_similarity_graph = build_graph_for_layers(embeddings, all_sentences)


# %%
print(cos_similarity_graph)

# %%

def remove_common_keys(dict1, dict2):
    # Find common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    # Remove common keys from both dictionaries
    for key in common_keys:
        del dict1[key]
        del dict2[key]
    
    return dict1, dict2

def find_adjacent(node_A, node_B, cosine_similarities):
# Identify adjacent nodes for A and B
  adjacent_to_A = {}
  for a, b, score in cosine_similarities:
        if a == node_A:
            adjacent_to_A[b] = score
        elif b == node_A:
            adjacent_to_A[a] = score
  try:
    adjacent_to_A.pop(node_B)
  except:
    pass
  adjacent_to_B = {}
  for a, b, score in cosine_similarities:
      if b == node_B:
          adjacent_to_B[a] = score
      elif a == node_B:
          adjacent_to_B[b] = score
  try:
    adjacent_to_B.pop(node_A)
  except:
    pass
  
  adjacent_to_A, adjacent_to_B = remove_common_keys(adjacent_to_A, adjacent_to_B)

  return list(dict(sorted(adjacent_to_A.items(), key=lambda item: item[1], reverse=True))), list(dict(sorted(adjacent_to_B.items(), key=lambda item: item[1], reverse=True)))[:5]

def find_sim_score(nodeA, nodeB, edges_list):
    # Find similarity score between two nodes from edges list
    for a, b, score in edges_list:
        if (a == nodeA and b == nodeB) or (a == nodeB and b == nodeA):
            return score
    return 0

def calculate_equation16(nodeA, nodeB, adj_A, adj_B, simScore):
  sum_weight = 0

  for node_A_adj in adj_A:
      for node_B_adj in adj_B:
      # Calculate weight(C,D) = cos_sim(A,C) × cos_sim(B,D)
        weight = (find_sim_score(nodeA, node_A_adj, simScore) *
                find_sim_score(nodeB, node_B_adj, simScore))
        # print("Node_pair: ", node_A_adj, node_A_adj)
        # print("Weight: ", weight)
        sum_weight += weight

  return sum_weight

def calculate_equation17(nodeA, nodeB, adj_A, adj_B, simScore):
    sum_weighted = 0
  
    for node_A_adj in adj_A:
      for node_B_adj in adj_B:  
            # Get cosine similarity between the common adjacent nodes
            sim = find_sim_score(node_A_adj, node_B_adj, simScore)
            weight = (find_sim_score(nodeA, node_A_adj, simScore) * find_sim_score(nodeB, node_B_adj, simScore))
            # print("Node_pair: ", weight)
            # Calculate final weighted value
            weighted = sim * weight
            sum_weighted += weighted
    return sum_weighted


def calculate_equation15(nodeA, nodeB, adj_A, adj_B, simScore):
    # Get cos_sim(A,B) - the direct similarity between A and B
    cos_sim_AB = find_sim_score(nodeA, nodeB, simScore)

    # Calculate sum_wtd(A,B) using equation 17
    sum_wtd = calculate_equation17(nodeA, nodeB, adj_A, adj_B, simScore)

    # Calculate sum_w_adj(A,B) using equation 16
    sum_w_adj = calculate_equation16(nodeA, nodeB, adj_A, adj_B, simScore)

    similarity = (cos_sim_AB + sum_wtd) / (1 + sum_w_adj)

    return similarity

def local_similarity(nodeA, nodeB, sim_list):

  adjacent_to_A, adjacent_to_B = find_adjacent(nodeA,nodeB, sim_list)

  final_similarity = calculate_equation15(nodeA, nodeB, adjacent_to_A, adjacent_to_B, sim_list)

  return final_similarity


def get_local_sim_from_jaccard_layer():
  data_jaccard = {}
  # calculated_sim_from_all_graph = []
  for sentence_pair_index in range(0,600):
    print(sentence_pair_index)
    nodeA = sentence_pair_index
    nodeB = nodeA + 600   
    data_jaccard[sentence_pair_index] = local_similarity(nodeA, nodeB, jaccard_graph) 
  with open('./jaccard_layer_local_sim_qwen.json', 'w') as file:
    json.dump(data_jaccard, file)

def get_local_sim_from_lcs_layer():
  data_lcs = {}
  # calculated_sim_from_all_graph = []
  for sentence_pair_index in range(0,600):
    print(sentence_pair_index)
    nodeA = sentence_pair_index
    nodeB = nodeA + 600
    data_lcs[sentence_pair_index] = local_similarity(nodeA, nodeB, lcs_graph) 
  with open('./lcs_layer_local_sim_qwen.json', 'w') as file:
    json.dump(data_lcs, file)

def get_local_sim_from_cos_layer():
  data_cos = {}
  # calculated_sim_from_all_graph = []
  for sentence_pair_index in range(0,600):
    print(sentence_pair_index)
    # print(sentence_pair_index)
    nodeA = sentence_pair_index
    nodeB = nodeA + 600
    # print(local_similarity(nodeA, nodeB, cos_similarity_graph) ) # The JSON you received from a function
    data_cos[sentence_pair_index ] = float(local_similarity(nodeA, nodeB, cos_similarity_graph) ) # The JSON you received from a function
  with open('./cos_layer_local_sim_qwen.json', 'w') as file:
    json.dump(data_cos, file)

# get_local_sim_from_lcs_layer()


# %%
get_local_sim_from_cos_layer()
get_local_sim_from_lcs_layer()
get_local_sim_from_jaccard_layer()

# %%
def overall_sentence_similarity(localsim:list, jsd_sum, jsd_main):
  sum = 0
  mul= 1
  for local in localsim[1:]:

    sum = sum + local
    mul = mul * local

  result =  ((jsd_sum - jsd_main) * localsim[0] ) + (mul/(jsd_sum*sum)) 

  return result

# %%
def get_overall_sim_from_all_layers():
    calculated_sim_from_all_graph = []
    with open('./cos_layer_local_sim_qwen.json', 'r') as file:
        cos_sim_local = json.load(file)
        # print(cos_sim_local)
    with open('./jaccard_layer_local_sim_qwen.json', 'r') as file:
        jaccard_sim_local = json.load(file)
        # print(jaccard_sim_local)
    with open('./lcs_layer_local_sim_qwen.json', 'r') as file:
        lcs_sim_local = json.load(file)

    jsd_main = 0.2895284502421509
    jsd_sum = 0.8685853507264527
    # calculated_sim_from_all_graph = overall_sentence_similarity([find_sim_score(0, 600, cos_similarity_graph), 0.17710259555547728, jaccard_sim_local["0"], lcs_sim_local["0"]],jsd_sum, jsd_main)
    # return calculated_sim_from_all_graph
    for sentence_pair_index in range(0,125):
        calculated_sim_from_all_graph.append(overall_sentence_similarity([find_sim_score(sentence_pair_index, sentence_pair_index +600, cos_similarity_graph), cos_sim_local[str(sentence_pair_index)], jaccard_sim_local[str(sentence_pair_index)], lcs_sim_local[str(sentence_pair_index)]],jsd_sum, jsd_main))
        # calculated_sim_from_all_graph.append(overall_sentence_similarity([find_sim_score(sentence_pair_index, sentence_pair_index +125, cos_similarity_graph), cos_sim_local[str(sentence_pair_index)], lcs_sim_local[str(sentence_pair_index)]],jsd_sum, jsd_main))

    return calculated_sim_from_all_graph

overall_sim_graph = get_overall_sim_from_all_layers()
print(overall_sim_graph[:10])

# %%
#Pearson score với baseline model embedding
from scipy import stats
calculated_sim_baseline = []
for i in range(0, 600):
 calculated_sim_baseline.append(cosine_similarity([embeddings[i]], [embeddings[i+600]])[0][0])

calculated_sim_baseline = [float(x) for x in calculated_sim_baseline]
pearson_correlation, p_value = stats.pearsonr(calculated_sim_baseline, ground_truth)
print(f"Pearson Correlation: {pearson_correlation}")
# print(f"P-value: {p_value}")
spearman_correlation, p_value = stats.spearmanr(calculated_sim_baseline, ground_truth)
print(f"Spearman Correlation: {spearman_correlation}")


from scipy import stats

overall_sim_graph = [float(x) for x in overall_sim_graph]


pearson_correlation, p_value = stats.pearsonr(overall_sim_graph, ground_truth)
print(f"Pearson Correlation: {pearson_correlation}")
# print(f"P-value: {p_value}")
# Compute Spearman correlation
spearman_correlation, p_value = stats.spearmanr(overall_sim_graph, ground_truth)

print(f"Spearman Correlation: {spearman_correlation}")



