#How to run
###  steps:-

Clone the reposetory

```bash
Project repo: https://github.com
```

### STEP 01-Create aconda environment after opening the reposetory

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

###STEP 02 -install the requirements
```bash
pip install -r requirement.txt
```

### Create  a  `.env` file in the root directory and add your Pinecone credential as follows:

```ini
PINECONE_API_KEY=" "
PINECONE_API_ENV=" "
```

#Download the Llama 2 model
llama-2-7b-chat.ggmlv3.q4_0.bin

#From  the following link

https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

```bash
#run the following command
python store_index.py
```

```bash
#Finally run the following command
python app.py

Now,
```bash
open up localhost:
```
### Techstack Used

-python
-Langchain
-Flask
-Meta Llama2
-Pinecone

### Output of the Project :-

<img width="1080" height="936" alt="image" src="https://github.com/user-attachments/assets/3a013f73-8ba4-4293-bd01-77105123c741" />

