#This file describes how GraphRAG works in derails 
#English

# Knowledge Graph Construction

## Step 1: Read Text and Split into Chunks
First, we read the text and divide it into chunks. This segmentation is essential for the subsequent steps in the knowledge graph construction process.

## Step 2: Entity and Relationship Recognition
From the text, identify entities and relationships. This step prepares for the graph embedding process, where entities are treated as nodes and relationships as edges.

## Step 3: Text to Vector Embedding
Convert the text into vector embeddings and store them in a vector store. This step is crucial for the later stages of analysis and computation.

## Step 4: Graph Construction Based on Relationships
Construct the graph based on the identified relationships. Here's an example to illustrate:

### Example:
**Text 1:** "Elon Musk is the CEO of SpaceX and Tesla."  
**Text 2:** "SpaceX is located in Hawthorne, California."

### Knowledge Graph Construction Process:
1. **Entity Recognition:** Identify entities like "Elon Musk," "SpaceX," "Tesla," "California," and "Hawthorne."
2. **Relationship Extraction:** Identify relationships such as "CEO" and "located in."
3. **Text Embedding:** Convert these entities and relationships into vector representations.
4. **Graph Construction:** Build a graph with the following nodes and edges:
    - **Nodes:** Elon Musk, SpaceX, Tesla, California, Hawthorne
    - **Edges:** 
        - Elon Musk -[CEO]-> SpaceX
        - Elon Musk -[CEO]-> Tesla
        - SpaceX -[located in]-> Hawthorne

## Step 5: Clustering Analysis with Leiden Algorithm
Apply the Leiden algorithm on the constructed knowledge graph for clustering analysis, dividing the graph into communities based on their relationships.

## Step 6: Graph Embedding for Community Analysis
Embed the clustered graph into a vector space to facilitate subsequent analysis and computations.

#中文版
# 知识图谱的构建

## 第一步：读入文本，将文本分成块
首先，我们读入文本并将其分成块。这一步的分段对于后续的知识图谱构建过程至关重要。

## 第二步：识别实体和关系
从文本中识别出实体和关系。这一步是为了后续的图嵌入过程做准备，在这个过程中，实体被当作节点，关系被当作边。

## 第三步：文本嵌入向量空间
将文本嵌入到向量空间，并存储到向量存储库中。这一步对于后续的分析和计算至关重要。

## 第四步：根据关系进行图构建
根据识别出的关系进行图构建。这里用一个例子来说明：

### 示例：
**文本1:** “Elon Musk 是 SpaceX 和 Tesla 的 CEO。”  
**文本2:** “SpaceX 位于加州 Hawthorne。”

### 知识图谱构建过程：
1. **实体识别：** 识别出“Elon Musk”、“SpaceX”、“Tesla”、“加州”、“Hawthorne”等实体。
2. **关系抽取：** 识别出“CEO”和“位于”等关系。
3. **文本嵌入：** 将这些实体和关系转换为向量表示。
4. **图构建：** 构建一个包含以下节点和边的图：
    - **节点：** Elon Musk, SpaceX, Tesla, 加州, Hawthorne
    - **边：** 
        - Elon Musk -[CEO]-> SpaceX
        - Elon Musk -[CEO]-> Tesla
        - SpaceX -[位于]-> Hawthorne

## 第五步：使用Leiden算法进行聚类分析
在构建好的知识图谱上运用Leiden算法进行聚类分析，根据相关性划分社区。

## 第六步：对社区图进行嵌入
（这里可以回答为什么有两个graph_embed的疑问）
将划分社区的图嵌入到向量空间，以便于之后的分析和计算。
