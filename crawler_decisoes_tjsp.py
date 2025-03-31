import requests
from datetime import date
from bs4 import BeautifulSoup
from urllib.parse import quote
from tqdm import tqdm

# Remove warnings a cada requisição
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)


def extract_decisoes_tjsp_grau1(termos_de_pesquisa, lista_datas=[], pagina_inicial=1, funcao=None):
    termo = data_pesquisa = numero_cnj = classe = assunto = magistrado = relator = comarca = foro = vara = orgao = data_disponibilizacao = data_julgamento = data_publicacao = decisao = ementa = ""
    grau = 1
    
    for termo in termos_de_pesquisa:
        for data_inicial, data_final in lista_datas:
            main_url = f"https://esaj.tjsp.jus.br/cjpg/pesquisar.do?conversationId=&dadosConsulta.pesquisaLivre={quote(termo)}&tipoNumero=UNIFICADO&numeroDigitoAnoUnificado=&foroNumeroUnificado=&dadosConsulta.nuProcesso=&dadosConsulta.nuProcessoAntigo=&classeTreeSelection.values=&classeTreeSelection.text=&assuntoTreeSelection.values=5566,3416,3419&assuntoTreeSelection.text=3+Registros+selecionados&agenteSelectedEntitiesList=&contadoragente=0&contadorMaioragente=0&cdAgente=&nmAgente=&dadosConsulta.dtInicio={data_inicial}&dadosConsulta.dtFim={data_final}&varasTreeSelection.values=&varasTreeSelection.text=&dadosConsulta.ordenacao=DESC"
            session = requests.Session()
            req = session.get(main_url, verify=False)
            
            if "Não foi encontrado nenhum resultado correspondente à busca realizada" in req.text:
                continue

            if pagina_inicial > 1:
                print('Acessando página escolhida como inicial')
                url_paginate = f"https://esaj.tjsp.jus.br/cjpg/trocarDePagina.do?pagina={pagina_inicial}&conversationId="
                req = session.get(url_paginate, verify=False)
                print("Iniciando a coleta\n")

            soupPrincipal = BeautifulSoup(req.content, 'html.parser')

            totalResultados = int(soupPrincipal.find("div",{"id":"resultados"}).find_all("table")[0].tr.td.text.strip().split("de ")[1])
            totalPaginas = totalResultados//10 if totalResultados/10 == 0 else (totalResultados//10)+1

            print(f"Iniciando coleta de dados com o termo: {termo}.")
            for i in tqdm(range(pagina_inicial, totalPaginas+1)):
                url_paginate = f"https://esaj.tjsp.jus.br/cjpg/trocarDePagina.do?pagina={i}&conversationId="
                req = session.get(url_paginate, verify=False)
                soupPrincipal = BeautifulSoup(req.content, 'html.parser')
                for target in soupPrincipal.find_all('tr',{'class':'fundocinza1'}):
                    contador = 0
                    flag = False

                    data_pesquisa = date.today().strftime("%d/%m/%Y")

                    for info in target.find_all('tr',{'class':'fonte'}):
                        if contador == 0:
                            numero_cnj = info.span.text.strip()
                        elif contador == 1:
                            classe = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 2:
                            assunto = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 3:
                            magistrado = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 4:
                            comarca = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 5:
                            foro = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 6:
                            vara = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 7:
                            data_disponibilizacao = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 8:
                            decisao = info.td.find_all('div',{'style':'display: none;'})[-1].span.text.strip()

                        contador += 1

                    linha = [termo, data_pesquisa, numero_cnj, classe, assunto, magistrado, relator, comarca, foro, vara, orgao, data_disponibilizacao, data_julgamento, data_publicacao, decisao, ementa, grau]
                    funcao(linha)


def extract_decisoes_tjsp_grau2(termos_de_pesquisa, lista_datas=[], pagina_inicial=1, funcao=None):
    termo = data_pesquisa = numero_cnj = classe = assunto = magistrado = relator = comarca = foro = vara = orgao = data_disponibilizacao = data_julgamento = data_publicacao = decisao = ementa = ""
    grau = 2

    for termo in termos_de_pesquisa:
        for data_inicial, data_final in lista_datas:
            data_post = {'conversationId': '',
                        'dados.buscaInteiroTeor': termo,
                        'dados.pesquisarComSinonimos': ['S','S'],
                        'dados.buscaEmenta': '',
                        'dados.nuProcOrigem': '',
                        'dados.nuRegistro': '',
                        'agenteSelectedEntitiesList': '',
                        'contadoragente': '0',
                        'contadorMaioragente': '0',
                        'codigoCr': '',
                        'codigoTr': '',
                        'nmAgente': '',
                        'juizProlatorSelectedEntitiesList': '',
                        'contadorjuizProlator': '0',
                        'contadorMaiorjuizProlator': '0',
                        'codigoJuizCr': '',
                        'codigoJuizTr': '',
                        'nmJuiz': '',
                        'classesTreeSelection.values': '',
                        'classesTreeSelection.text': '',
                        'assuntosTreeSelection.values': '',
                        'assuntosTreeSelection.text': '',
                        'comarcaSelectedEntitiesList': '',
                        'contadorcomarca': '0',
                        'contadorMaiorcomarca': '0',
                        'cdComarca': '',
                        'nmComarca': '',
                        'secoesTreeSelection.values': '',
                        'secoesTreeSelection.text': '',
                        'dados.dtJulgamentoInicio': '',
                        'dados.dtJulgamentoFim': '',
                        'dados.dtPublicacaoInicio': data_inicial,
                        'dados.dtPublicacaoFim': data_final,
                        'dados.origensSelecionadas': 'T',
                        'tipoDecisaoSelecionados': 'A',
                        'dados.ordenarPor': 'dtPublicacao',
                    }
            main_url = "https://esaj.tjsp.jus.br/cjsg/resultadoCompleta.do"
            session = requests.Session()
            req = session.post(main_url, data=data_post, verify=False)
            
            if "Não foi encontrado nenhum resultado correspondente à busca realizada" in req.text:
                continue

            if pagina_inicial > 1:
                print('Acessando página escolhida como inicial')
                url_paginate = f"https://esaj.tjsp.jus.br/cjsg/trocaDePagina.do?tipoDeDecisao=A&pagina={pagina_inicial}&conversationId="
                req = session.get(url_paginate, verify=False)
                print("Iniciando a coleta\n")

            soupPrincipal = BeautifulSoup(req.content, 'html.parser')

            totalResultados = int(soupPrincipal.find("div",{"id":"paginacaoSuperior-A"}).find("table").tr.td.text.split("de ")[1].strip())
            totalPaginas = totalResultados//20 if totalResultados/20 == 0 else (totalResultados//20)+1

            print(f"Iniciando coleta de dados com o termo: {termo}.")
            for i in tqdm(range(pagina_inicial, totalPaginas+1)):
                url_paginate = f"https://esaj.tjsp.jus.br/cjsg/trocaDePagina.do?tipoDeDecisao=A&pagina={i}&conversationId="
                req = session.get(url_paginate, verify=False)
                soupPrincipal = BeautifulSoup(req.content, 'html.parser')

                for target in soupPrincipal.find_all('tr',{'class':'fundocinza1'}):
                    contador = 0

                    data_pesquisa = date.today().strftime("%d/%m/%Y")

                    numero_cnj = target.find_all('tr',{'class':'ementaClass'})[0].td.a.text.strip()
                    for info in target.find_all('tr',{'class':'ementaClass2'}):
                        if contador == 0:
                            classe = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 1:
                            relator = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 2:
                            comarca = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 3:
                            orgao = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 4:
                            data_julgamento = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 5:
                            data_publicacao = info.td.text.strip().split("\n\t")[-1].strip()
                        elif contador == 6:
                            if len(info.td.find_all('div')) != 0:
                                soupAux = info.td.find_all('div')[-1]
                                for data in soupAux(['strong']):
                                    data.decompose()
                                ementa = ''.join(soupAux.stripped_strings)
                            else:
                                ementa = ""

                        contador += 1

                    linha = [termo, data_pesquisa, numero_cnj, classe, assunto, magistrado, relator, comarca, foro, vara, orgao, data_disponibilizacao, data_julgamento, data_publicacao, decisao, ementa, grau]
                    funcao(linha)

