from transformers import AutoTokenizer
from tf_layers.transformer import TransformerEncoder, TransformerDecoder

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = ["This is a sample sentence", "This is another sentence"]

encoded_input = tokenizer(text, padding=True, return_tensors='tf')

encoded_input = encoded_input["input_ids"]

VOCAB_SIZE = tokenizer.vocab_size

    
encoder = TransformerDecoder(VOCAB_SIZE, 2, 600, 8)
output = encoder(encoded_input)
print(output.shape)