# EXP-05 : Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework
## NAME : HARI PRIYA M
## REGISTER NO : 212224240047
## DATE : 31.10.2025

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
In many text-based applications, it is difficult to automatically identify useful information such as names, places, or organizations from unstructured sentences. To solve this problem, this project develops a Named Entity Recognition (NER) application prototype that can detect and highlight entities like people, locations, organizations, dates, and money from user input text. For example, in the sentence “My name is Andrew and I live in California”, the system identifies Andrew as a person and California as a location.

### DESIGN STEPS:
#### STEP 1: Input Text
Accept user input text through a textbox (e.g., “Chennai is famous for its beaches”).
#### STEP 2: Entity Detection
Pass the input to the pretrained model (dslim/bert-base-NER) using the get_completion() function to get detected tokens with entity labels.
#### STEP 3: Token Merging
Use the merge_tokens() function to combine sub-tokens (e.g., “New” and “York”) into complete entities like “New York”.
#### STEP 4: Entity Highlighting
Return the merged entities and display them using a highlighted text interface for better visualization.
#### STEP 5: User Output
Show the final text with recognized entities marked by their types (Person, Location, Organization, etc.).
### PROGRAM:

```python

from transformers import pipeline

get_completion = pipeline("ner", model="dslim/bert-base-NER")

def ner(input):
    output = get_completion(input)
    return {"text": input, "entities": output}

API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "Amazon Web Services launched a new AI product."
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
```

```python
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["Chennai is famous for its beaches", "My birthday is on July 9th"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
```

```python
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["I took a train from Rome to Venice.", "Taj Mahal is located in Agra, India."])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```

### OUTPUT:
<img width="250" height="600" alt="Screenshot 2025-10-31 111530" src="https://github.com/user-attachments/assets/11798145-b78b-4800-858d-894a97f1d44d" />
<br><br>
BEFORE MERGING :
<img width="1000" height="500" alt="Screenshot 2025-10-31 110216" src="https://github.com/user-attachments/assets/c6eaf13f-5fcc-4751-8606-aef6fed9bece" />
<br>
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/0194b4d5-9f15-4c89-9f88-41471f2d026b" />
<br>
AFTER MERGING :
<img width="1000" height="500" alt="Screenshot 2025-10-31 110556" src="https://github.com/user-attachments/assets/98599af1-e81d-4d26-8047-5701515fa3d6" />
<br>

### RESULT:
Named Entity Recognition prototype successfully implemented and tested with sample inputs.
