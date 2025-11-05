#in the input-target appraoch, we will use the data loader and the sliding window approach
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt","r",encoding="utf-8")as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

#Next , we remove the first 50 tokens from the dataset
enc_sample = enc_text[:50]

#one of the most intuitive ways to create the input-target pairs is to create two variables, x and y, when x contain the input token and y include the target
#Context_size will include how many tokens included in the input
context_size =4
#It means  the model is trained to look at a sequence of 4 words (or tokens) to ppredict the next word in sequence
#Eg, first 4 tokens [1,2,3,4], and the target y is the next 4 token [2,3,4,5]
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")

#Processing the inputs along with the targets, whcih are the inputs shifted by one position, we can create the next-word prediction tasks as follows
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

#This is the decode part
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))








 