# Roteiro para criação de um controle de versionamento de dados com DVC

1. Crie a estrututura de pastas desejada. No meu caso, vou adotar a seguinte:
    * Criar uma pasta `data` na raiz do repositório
    * Criar uma pasta `source` na raiz do repositório
    * Criar uma pasta `stages` na raiz do repositório

2. Iniciar o git

    ```
    $ git init
    ```

3. Iniciar DVC

    ```
    $ dvc init
    ```

4. Configurar o servidor remoto do DVC. Nesse caso, vou utilizar o google drive mesmo
    * Na sua pasta do google drive, gere um link de compartilhamento
    ```https://drive.google.com/drive/folders/<<Codigo_da_pasta>>?usp=sharing```
    * Copie o código da pasta elucidado no tópico acima
    * Escolha um nome. Por exemplo, "pc_imoveis"
    * Adicione o remote ao dvc

    ```
    $ dvc remote add --default pc_imoveis \
            gdrive://<<Codigo_da_pasta>>
    ```

5. O próximo passo é configurar o arquivo de parâmetros. Para isso, crie `params.yaml` na raiz do repositório. Esse arquivo deve ter a seguinte estrutura:

```
stage_1_name:
    parameter_1: "parameter_1"

stage_2_name:
    parameter_2: True
    parameter_3: 43.1

...

stage_n_name:
    last_paramter: 0
```

6. Cada estágio descrito no arquivo de parametros pode estar associado a um arquivo em `source` (Não necessariamente. É apenas uma forma de organização que me agrada).

7. Vamos criar um arquivo `source/stg01_data_load.py` que simplesmente importa a base de dados `data/pc_imoveis.csv` e salva uma cópia em stages. Essa etapa é só representativa. No exemplo abaixo, estamos assumindo que dois parâmetros estão sendo passados: o endereço da base de dados principal "raw_data_path" e o endereço do output "stg01_data_path" configurados no `params.yaml` num stage chamado `data_load`

```
import argparse
import yaml
import pandas as pd
from typing import Text


def data_load(config_path):
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    raw_data_path = config["data_load"]["raw_data_path"]
    stg01_data_path = config["data_load"]["stg01_data_path"]

    data = pd.read_csv(raw_data_path)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    data_load(config_path=args.config)

```
8. Após esse ponto, já é possível rodar uma pipeline de um stage. No codigo abaixo, `-n` indica o nome do stage; `-d` indica as dependências desse stage; `-o` indica o output

```
dvc run --force -n data_load \
        -d data/imoveis_pocos.csv \
        -d source/stg01_data_load.py \
        -o stages/stg01_data_path.csv \
        -p data_load\
        python src/data_load.py \
        --config=params.yaml 
```