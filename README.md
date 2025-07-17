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


