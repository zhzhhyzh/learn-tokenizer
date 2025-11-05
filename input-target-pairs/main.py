# -----------------------------
# IMPORTS
# -----------------------------
import torch                                       # PyTorch for tensors and neural networks
from torch.utils.data import Dataset, DataLoader   # For creating custom datasets and data loaders
import tiktoken                                    # OpenAI's tokenizer library for GPT-style encodings


# -----------------------------
# GPT DATASET DEFINITION
# -----------------------------
class GPTDatasetV1(Dataset):
    # Custom dataset that turns long text into overlapping sequences of token IDs
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []     # List to store tokenized input sequences
        self.target_ids = []    # List to store the shifted target sequences

        # Tokenize the entire input text (convert text → token IDs)
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Sliding window loop: break token list into overlapping chunks
        for i in range(0, len(token_ids) - max_length, stride):
            # Take one sequence of max_length tokens as input
            input_chunk = token_ids[i:i + max_length]
            # Take the same sequence shifted by one as target (predict next token)
            target_chunk = token_ids[i + 1:i + max_length + 1]

            # Convert to PyTorch tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Return total number of samples
    def __len__(self):
        return len(self.input_ids)

    # Retrieve one (input, target) pair by index
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# -----------------------------
# DATA LOADER CREATION FUNCTION
# -----------------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize GPT tokenizer (GPT-2's base encoding)
    tokenizer = tiktoken.get_encoding("r50k_base")   # "r50k_base" = GPT-2 tokenizer

    # Create dataset using the custom GPTDatasetV1 class
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap dataset with a PyTorch DataLoader for batching & shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=shuffle,        # Shuffle data each epoch (good for training)
        drop_last=drop_last,    # Drop incomplete batches
        num_workers=num_workers # Number of parallel CPU threads to load data
    )
    return dataloader


# -----------------------------
# LOAD RAW TEXT DATA
# -----------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()                             # Read the full text from file

print("PyTorch version:", torch.__version__)        # Print the installed PyTorch version


# -----------------------------
# TEST DATALOADER
# -----------------------------
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# Convert dataloader to an iterator to manually fetch batches
data_iter = iter(dataloader)

# Fetch first batch (input-target pair)
first_batch = next(data_iter)
print("First batch:", first_batch)

# Fetch second batch to show the next sequence
second_batch = next(data_iter)
print("Second batch:", second_batch)


# -----------------------------
# TEST BATCHING WITH LARGER SIZE
# -----------------------------
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)                   # Unpack first batch
print("\nInputs:\n", inputs)
print("\nTargets:\n", targets)


# -----------------------------
# TOKEN EMBEDDING EXAMPLE
# -----------------------------
input_ids = torch.tensor([2, 3, 5, 1])              # Example token IDs (toy example)
vocab_size = 6                                      # Total tokens in vocabulary
output_dim = 3                                      # Embedding dimension
torch.manual_seed(123)                              # Set random seed for reproducibility

# Create embedding layer (maps token IDs → learnable vectors)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("\nToken Embedding Weights (vocab_size x output_dim):")
print(embedding_layer.weight)                       # Print initial random embedding weights

# Get embedding for token ID 3
print("\nEmbedding for token 3:")
print(embedding_layer(torch.tensor([3])))

# Get embeddings for the entire sequence [2, 3, 5, 1]
print("\nEmbeddings for [2,3,5,1]:")
print(embedding_layer(input_ids))


# -----------------------------
# REAL TOKEN EMBEDDINGS (GPT-SIZED)
# -----------------------------
vocab_size = 50257                                  # GPT-2 vocabulary size
output_dim = 256                                    # Embedding vector size (like GPT small)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4                                      # Sequence length per training sample
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)                   # Get first batch

print("\nToken IDs:\n", inputs)                     # Print raw token IDs
print("\nInputs shape:\n", inputs.shape)            # (batch_size, max_length)

# Convert token IDs to embedding vectors
token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings shape (batch_size, seq_len, embed_dim):")
print(token_embeddings.shape)


# -----------------------------
# POSITIONAL EMBEDDINGS
# -----------------------------
context_length = max_length                         # Max sequence length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Generate position indices [0, 1, 2, 3]
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("\nPositional embeddings shape (seq_len, embed_dim):")
print(pos_embeddings.shape)


# -----------------------------
# COMBINE TOKEN + POSITION EMBEDDINGS
# -----------------------------
# Add positional embeddings to token embeddings
# → Each token now encodes both meaning (word) and position (order)
input_embeddings = token_embeddings + pos_embeddings
print("\nFinal input embeddings shape (batch_size, seq_len, embed_dim):")
print(input_embeddings.shape)
