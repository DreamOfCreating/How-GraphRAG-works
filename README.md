#This file describes how GraphRAG works in derails 
#English

# Knowledge Graph Construction

## Step 1: Read Text and Split into Chunks
First, we read the text and divide it into chunks. This segmentation is essential for the subsequent steps in the knowledge graph construction process.

'''python code
    class TextSplitter(ABC):
    """Text splitter class definition."""

    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn
    _keep_separator: bool
    _add_start_index: bool
    _strip_whitespace: bool

    def __init__(
        self,
        # based on text-ada-002-embedding max input buffer length
        # https://platform.openai.com/docs/guides/embeddings/second-generation-models
        chunk_size: int = 8191,
        chunk_overlap: int = 100,
        length_function: LengthFn = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """Init method definition."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""


    class NoopTextSplitter(TextSplitter):
    """Noop text splitter class definition."""

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""
        return [text] if isinstance(text, str) else text


    class TokenTextSplitter(TextSplitter):
    """Token text splitter class definition."""

    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """Init method definition."""
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                log.exception("Model %s not found, using %s", model_name, encoding_name)
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special or set()
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        if cast(bool, pd.isna(text)) or text == "":
            return []
        if isinstance(text, list):
            text = " ".join(text)
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}"
            raise TypeError(msg)

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


    class TextListSplitterType(str, Enum):
    """Enum for the type of the TextListSplitter."""

    DELIMITED_STRING = "delimited_string"
    JSON = "json"


    class TextListSplitter(TextSplitter):
    """Text list splitter class definition."""

    def __init__(
        self,
        chunk_size: int,
        splitter_type: TextListSplitterType = TextListSplitterType.JSON,
        input_delimiter: str | None = None,
        output_delimiter: str | None = None,
        model_name: str | None = None,
        encoding_name: str | None = None,
    ):
        """Initialize the TextListSplitter with a chunk size."""
        # Set the chunk overlap to 0 as we use full strings
        super().__init__(chunk_size, chunk_overlap=0)
        self._type = splitter_type
        self._input_delimiter = input_delimiter
        self._output_delimiter = output_delimiter or "\n"
        self._length_function = lambda x: num_tokens_from_string(
            x, model=model_name, encoding_name=encoding_name
        )

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split a string list into a list of strings for a given chunk size."""
        if not text:
            return []

        result: list[str] = []
        current_chunk: list[str] = []

        # Add the brackets
        current_length: int = self._length_function("[]")

        # Input should be a string list joined by a delimiter
        string_list = self._load_text_list(text)

        if len(string_list) == 1:
            return string_list

        for item in string_list:
            # Count the length of the item and add comma
            item_length = self._length_function(f"{item},")

            if current_length + item_length > self._chunk_size:
                if current_chunk and len(current_chunk) > 0:
                    # Add the current chunk to the result
                    self._append_to_result(result, current_chunk)

                    # Start a new chunk
                    current_chunk = [item]
                    # Add 2 for the brackets
                    current_length = item_length
            else:
                # Add the item to the current chunk
                current_chunk.append(item)
                # Add 1 for the comma
                current_length += item_length

        # Add the last chunk to the result
        self._append_to_result(result, current_chunk)

        return result

    def _load_text_list(self, text: str | list[str]):
        """Load the text list based on the type."""
        if isinstance(text, list):
            string_list = text
        elif self._type == TextListSplitterType.JSON:
            string_list = json.loads(text)
        else:
            string_list = text.split(self._input_delimiter)
        return string_list

    def _append_to_result(self, chunk_list: list[str], new_chunk: list[str]):
        """Append the current chunk to the result."""
        if new_chunk and len(new_chunk) > 0:
            if self._type == TextListSplitterType.JSON:
                chunk_list.append(json.dumps(new_chunk))
            else:
                chunk_list.append(self._output_delimiter.join(new_chunk))


    def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
        """Split incoming text and return chunks using tokenizer."""
        splits: list[str] = []
        input_ids = tokenizer.encode(text)
        start_idx = 0
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(tokenizer.decode(chunk_ids))
            start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
            cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits
'''
## Step 2: Entity and Relationship Recognition
From the text, identify entities and relationships. This step prepares for the graph embedding process, where entities are treated as nodes and relationships as edges.
'''python code
    log.debug("entity_extract strategy=%s", strategy)
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    output = cast(pd.DataFrame, input.get_input())
    strategy = strategy or {}
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        text = row[column]
        id = row[id_column]
        result = await strategy_exec(
            [Document(text=text, id=id)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        num_started += 1
        return [result.entities, result.graphml_graph]

    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )

    to_result = []
    graph_to_result = []
    for result in results:
        if result:
            to_result.append(result[0])
            graph_to_result.append(result[1])
        else:
            to_result.append(None)
            graph_to_result.append(None)

    output[to] = to_result
    if graph_to is not None:
        output[graph_to] = graph_to_result

    return TableContainer(table=output.reset_index(drop=True))
'''
## Step 3: Text to Vector Embedding
Convert the text into vector embeddings and store them in a vector store. This step is crucial for the later stages of analysis and computation.
'''python code
    async def _text_embed_in_memory(
    input: VerbInput,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    column: str,
    strategy: dict,
    to: str,
):
    output_df = cast(pd.DataFrame, input.get_input())
    strategy_type = strategy["type"]
    strategy_exec = load_strategy(strategy_type)
    strategy_args = {**strategy}
    input_table = input.get_input()

    texts: list[str] = input_table[column].to_numpy().tolist()
    result = await strategy_exec(texts, callbacks, cache, strategy_args)

    output_df[to] = result.embeddings
    return TableContainer(table=output_df)


async def _text_embed_with_vector_store(
    input: VerbInput,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    column: str,
    strategy: dict[str, Any],
    vector_store: BaseVectorStore,
    vector_store_config: dict,
    store_in_table: bool = False,
    to: str = "",
):
    output_df = cast(pd.DataFrame, input.get_input())
    strategy_type = strategy["type"]
    strategy_exec = load_strategy(strategy_type)
    strategy_args = {**strategy}

    # Get vector-storage configuration
    insert_batch_size: int = (
        vector_store_config.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE
    )
    title_column: str = vector_store_config.get("title_column", "title")
    id_column: str = vector_store_config.get("id_column", "id")
    overwrite: bool = vector_store_config.get("overwrite", True)

    if column not in output_df.columns:
        msg = f"Column {column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)
    if title_column not in output_df.columns:
        msg = f"Column {title_column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)
    if id_column not in output_df.columns:
        msg = f"Column {id_column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)

    total_rows = 0
    for row in output_df[column]:
        if isinstance(row, list):
            total_rows += len(row)
        else:
            total_rows += 1

    i = 0
    starting_index = 0

    all_results = []

    while insert_batch_size * i < input.get_input().shape[0]:
        batch = input.get_input().iloc[
            insert_batch_size * i : insert_batch_size * (i + 1)
        ]
        texts: list[str] = batch[column].to_numpy().tolist()
        titles: list[str] = batch[title_column].to_numpy().tolist()
        ids: list[str] = batch[id_column].to_numpy().tolist()
        result = await strategy_exec(
            texts,
            callbacks,
            cache,
            strategy_args,
        )
        if store_in_table and result.embeddings:
            embeddings = [
                embedding for embedding in result.embeddings if embedding is not None
            ]
            all_results.extend(embeddings)

        vectors = result.embeddings or []
        documents: list[VectorStoreDocument] = []
        for id, text, title, vector in zip(ids, texts, titles, vectors, strict=True):
            if type(vector) is np.ndarray:
                vector = vector.tolist()
            document = VectorStoreDocument(
                id=id,
                text=text,
                vector=vector,
                attributes={"title": title},
            )
            documents.append(document)

        vector_store.load_documents(documents, overwrite and i == 0)
        starting_index += len(documents)
        i += 1

    if store_in_table:
        output_df[to] = all_results

    return TableContainer(table=output_df)


def _create_vector_store(
    vector_store_config: dict, collection_name: str
) -> BaseVectorStore:
    vector_store_type: str = str(vector_store_config.get("type"))
    if collection_name:
        vector_store_config.update({"collection_name": collection_name})

    vector_store = VectorStoreFactory.get_vector_store(
        vector_store_type, kwargs=vector_store_config
    )

    vector_store.connect(**vector_store_config)
    return vector_store

'''
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

##中文版
# 知识图谱的构建

## 第一步：读入文本，将文本分成块(chunks)
首先，我们读入文本并将其分成块。这一步的分段对于后续的知识图谱构建过程至关重要。

## 第二步：识别实体和关系
从文本中识别出实体和关系。这一步是为了后续的图嵌入过程做准备，在这个过程中，实体被当作节点，关系被当作边。

## 第三步：文本嵌入向量空间
Text —> vector 将文本嵌入（embedding）到向量空间，并存储到vector_store,将文本嵌入到向量空间，并存储到向量存储库中。。

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
