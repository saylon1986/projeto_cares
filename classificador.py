import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import glob
import random
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import re


warnings.filterwarnings("ignore")


def extrair_trecho(ementa):
    match = re.search(r'(?:.{0,500}cuid.{0,600})', ementa, re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return None



def load_data(file_path):
    df = pd.read_excel(file_path)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['classificacao'])
    texts = df['texto'].tolist()
    texts_lp = []
    for t in texts:
        texts_lp.append(t)


    # Dividir os dados em conjuntos de treinamento e validação
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_lp, labels, test_size=0.2, random_state=42)  # Ajuste test_size conforme necessário ### aumentei pra 20% e vou incluir mais casos

    return texts_train, labels_train, texts_val, labels_val, label_encoder


def train_model(texts_train, labels_train, texts_val, labels_val, learning_rate=5e-5, early_stopping_patience=3):
    model_name = "neuralmind/bert-large-portuguese-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)


    # Tokenizar os conjuntos de treinamento e validação
    train_tokens = tokenizer(texts_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
    val_tokens = tokenizer(texts_val, padding=True, truncation=True, max_length=512, return_tensors="pt")

    class TextDataset(Dataset):  
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)


    train_dataset = TextDataset(train_tokens, labels_train)
    val_dataset = TextDataset(val_tokens, labels_val)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels_train)))
    training_args = TrainingArguments(
        output_dir='./results_TJSP_2',  # Local para salvar os resultados
        learning_rate=learning_rate,  # Ajustando a taxa de aprendizado
        num_train_epochs=4,  # Pode ser ajustado se necessário ##### aumentar para 4
        per_device_train_batch_size=8,
        logging_dir='./logs',  # Diretório de logs
        logging_steps=10,
        evaluation_strategy="steps",  # Avaliação a cada 'logging_steps' passos
        eval_steps=10,  # Avaliação a cada 10 passos
        # save_strategy="steps",  # Ativa o salvamento automático de checkpoints
        # save_steps=10,  # Salva o modelo a cada 10 passos de treinamento
        load_best_model_at_end=True,  # Carrega o melhor modelo no final do treinamento
    )

    # Adicionando o Early Stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))}
    )

    trainer.train()
    model.save_pretrained('./text_classifier_model_TJSP_2')
    tokenizer.save_pretrained('./text_classifier_model_TJSP_2')

    return model, tokenizer



def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict(texts, model, tokenizer, label_encoder):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_labels_indices = probabilities.argmax(dim=-1)
    predicted_labels = label_encoder.inverse_transform(predicted_labels_indices.cpu().numpy())
    return predicted_labels, probabilities


##############################################################################################

def classificador():

    file_path = "treino_TJSP.xlsx"
    if not os.path.exists('./text_classifier_model_TJSP_2'):  ####### trocar aqui
        print("-"*30)
        print()
        print("treinando o Transformer...")
        print()
        print("-"*30)
        # texts, labels, label_encoder = load_data(file_path)
        texts_train, labels_train, texts_val, labels_val, label_encoder = load_data(file_path)
        model, tokenizer = train_model(texts_train, labels_train, texts_val, labels_val)
    else:
        model, tokenizer = load_model('./text_classifier_model_TJSP_2') ######### trocar aqui
        texts_train, labels_train, texts_val, labels_val, label_encoder = load_data(file_path)


    print('TREINO FINALIZADO!')

    # z = input("")


    #############################################################################################################################
    #############################################################################################################################

 
    # lista = glob.glob("./arquivos_2024/TJSP/*.xlsx")

    lista= ["./arquivos_2024/TJSP/outros.xlsx"]

    df_final = pd.DataFrame()

    # print(lista)

    # z= input("")

    for planilha in lista:
   
        print()
        print("-"*30)
        print()
        nome_proc = planilha.split("//")[-1][:-5]
        # print("Número", c, 'de', len(lista))
        print("lendo:",nome_proc)
        df = pd.read_excel(planilha)
        print("Linhas",len(df))
        print()
        print("-"*30)
        print()


        txts = df["ementa"].to_list()
        avals_ = df["tipo"].to_list()

        # txts = df["ementa"].str[350:2100].to_list() ## para o caso da BA

        # txts = df['ementa'].apply(extrair_trecho).to_list()

        # var = []
        # probab = []
        # # txt_ajust = [] # para o caso da Bahia
        # for m in txts:
        #     # print(type(m))
        #     # print(m)
        #     # print(len(str(m)))
        #     m = str(m)
        #     if str(m) != 'nan' or m != None:
        #         # print("entrou!")
        #         predicted_labels, probabilities = predict(m, model, tokenizer, label_encoder)
        #         max_probs, _ = probabilities.max(dim=1)
        #         print(predicted_labels, probabilities)
        #         print(max_probs.item())
        #         print()
        #         print("-"*30)
        #         print()
        #         var.append(predicted_labels[0])
        #         probab.append(max_probs.item())
        #         # txt_ajust.append(m)
        #     else:
        #         var.append(None)
        #         probab.append(None)

        var = []
        probab = []
        for m,n in zip(avals_,txts):
            # print(type(m))
            # print(m)
            # print(len(str(m)))
            m = str(m)
            # print(m)
            if str(m) == 'nan' or m == None:
                # print("novo!")
                # print(n)
                predicted_labels, probabilities = predict(str(n), model, tokenizer, label_encoder)
                max_probs, _ = probabilities.max(dim=1)
                print(predicted_labels, probabilities)
                print(max_probs.item())
                print()
                print("-"*30)
                print()
                var.append(predicted_labels[0])
                probab.append(max_probs.item())
                # txt_ajust.append(m)
            else:
                # print("anterior")
                var.append(m)
                probab.append("anterior")

            # z = input("")


        df["tipo"] = var
        df["probabilidade"] = probab
        # df["ementa_ajustada"] = txt_ajust


        df_final = pd.concat([df_final,df], ignore_index = True)
        

    df_final.to_excel("./classificados/TJSP_classificados.xlsx", index = False)


#############################################################


if __name__ == "__main__":
    # ajuda_separar_treino()
    classificador()