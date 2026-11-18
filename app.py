"""Interface Streamlit que conecta o processamento ao reconhecedor Eigenfaces."""

import cv2
import numpy as np
import streamlit as st

from processamento import processar_imagem
from reconhecimento import ReconhecedorFacial
from webcam import capturar_rosto_webcam


def inicializar_estado():
	"""Prepara os dados que precisam sobreviver às novas execuções da página.

	O Streamlit reexecuta este arquivo a cada interação. O estado da sessão mantém
	os vetores, seus nomes e o modelo treinado disponíveis durante essa sessão.
	"""
	if "imagens_processadas" not in st.session_state:
		st.session_state.imagens_processadas = []
	if "identificadores" not in st.session_state:
		st.session_state.identificadores = []
	if "reconhecedor" not in st.session_state:
		st.session_state.reconhecedor = None


def decodificar_upload(arquivo):
	"""Converte os bytes enviados pelo navegador em uma matriz de pixels BGR.

	O OpenCV usa a ordem de canais BGR, que é a esperada pelo processamento
	quando a imagem tem três canais de cor.
	"""
	bytes_imagem = np.frombuffer(arquivo.getvalue(), dtype=np.uint8)
	imagem = cv2.imdecode(bytes_imagem, cv2.IMREAD_COLOR)
	if imagem is None:
		raise ValueError(f"Não foi possível ler o arquivo {arquivo.name} como imagem.")
	return imagem


def cadastrar_imagens(nome, arquivos):
	"""Processa as fotos e associa cada vetor ao nome informado.

	Os vetores são preparados antes de atualizar o estado: assim, um arquivo
inválido não deixa apenas parte do grupo de fotos cadastrada.
	"""
	novos_vetores = []
	for arquivo in arquivos:
		imagem = decodificar_upload(arquivo)
		novos_vetores.append(processar_imagem(imagem))

	# A posição de cada nome corresponde à mesma posição do vetor na lista de imagens.
	st.session_state.imagens_processadas.extend(novos_vetores)
	st.session_state.identificadores.extend([nome] * len(novos_vetores))
	# Como os dados de treino mudaram, as projeções antigas não representam mais o cadastro.
	st.session_state.reconhecedor = None


def reconhecer_imagem(imagem, reconhecedor):
	"""Aplica o pré-processamento compartilhado e compara pelo PCA treinado."""
	vetor = processar_imagem(imagem)
	return reconhecedor.reconhecer(vetor)


def main():
	"""Monta a página de cadastro, treinamento e comparação de fotografias."""
	# A configuração da página precisa ocorrer antes dos demais comandos Streamlit.
	st.set_page_config(page_title="Reconhecimento facial com Eigenfaces", page_icon="📐")
	inicializar_estado()

	st.title("Do Pixel à Identificação")
	st.caption("Protótipo acadêmico de reconhecimento facial com PCA e Eigenfaces")
	st.info(
		"Este protótipo compara imagens cadastradas por distância euclidiana. "
		"O resultado não valida a identidade com segurança."
	)

	st.header("1. Cadastro de pessoa")
	nome = st.text_input("Nome da pessoa", key="nome_cadastro")
	arquivos_pessoa = st.file_uploader(
		"Selecione uma ou mais fotografias",
		type=["png", "jpg", "jpeg", "bmp"],
		accept_multiple_files=True,
		key="fotos_cadastro",
	)

	if st.button("Adicionar fotografias", type="primary"):
		nome_limpo = nome.strip()
		if not nome_limpo:
			st.error("Informe o nome da pessoa antes de cadastrar.")
		elif not arquivos_pessoa:
			st.error("Selecione pelo menos uma fotografia.")
		else:
			try:
				cadastrar_imagens(nome_limpo, arquivos_pessoa)
			except (ValueError, cv2.error) as erro:
				st.error(f"Não foi possível cadastrar as fotografias: {erro}")
			else:
				st.success(f"{len(arquivos_pessoa)} fotografia(s) cadastrada(s) para {nome_limpo}.")

	quantidade = len(st.session_state.imagens_processadas)
	st.write(f"Fotografias cadastradas nesta sessão: **{quantidade}**")
	if quantidade:
		contagem_por_pessoa = {}
		for identificador in st.session_state.identificadores:
			contagem_por_pessoa[identificador] = contagem_por_pessoa.get(identificador, 0) + 1
		st.write("Pessoas cadastradas:", ", ".join(
			f"{pessoa} ({total})" for pessoa, total in contagem_por_pessoa.items()
		))

	st.header("2. Treinamento do reconhecedor")
	n_componentes = st.number_input(
		"Quantidade máxima de componentes PCA",
		min_value=1,
		value=50,
		step=1,
		help="O modelo limita automaticamente esse valor conforme a quantidade de imagens.",
	)

	if st.button("Treinar reconhecedor"):
		if quantidade == 0:
			st.warning("Cadastre fotografias antes de treinar o reconhecedor.")
		elif quantidade < 2:
			st.warning("O PCA precisa de pelo menos duas fotografias para ser treinado.")
		else:
			try:
				# Cada linha é uma imagem vetorizada; os identificadores seguem a mesma ordem.
				reconhecedor = ReconhecedorFacial(n_componentes=int(n_componentes))
				reconhecedor.treinar(
					np.asarray(st.session_state.imagens_processadas),
					st.session_state.identificadores,
				)
			except (ValueError, np.linalg.LinAlgError) as erro:
				st.session_state.reconhecedor = None
				st.error(f"Não foi possível treinar o reconhecedor: {erro}")
			else:
				st.session_state.reconhecedor = reconhecedor
				st.success(
					"Treinamento concluído com "
					f"{reconhecedor.n_componentes_} componente(s) principal(is)."
				)

	st.header("3. Reconhecimento")
	arquivo_consulta = st.file_uploader(
		"Selecione uma fotografia para comparar",
		type=["png", "jpg", "jpeg", "bmp"],
		accept_multiple_files=False,
		key="foto_consulta",
	)

	if st.button("Reconhecer pessoa"):
		reconhecedor = st.session_state.reconhecedor
		if quantidade == 0:
			st.warning("Ainda não há fotografias cadastradas.")
		elif reconhecedor is None:
			st.warning("Treine o reconhecedor antes de enviar uma imagem para reconhecimento.")
		elif arquivo_consulta is None:
			st.error("Selecione uma fotografia para realizar a comparação.")
		else:
			try:
				imagem = decodificar_upload(arquivo_consulta)
				pessoa, distancia = reconhecer_imagem(imagem, reconhecedor)
			except (ValueError, cv2.error, RuntimeError) as erro:
				st.error(f"Não foi possível processar ou comparar a fotografia: {erro}")
			else:
				st.subheader("Pessoa cadastrada mais próxima")
				st.write(f"**Nome:** {pessoa}")
				st.write(f"**Distância euclidiana:** {distancia:.6f}")
				st.caption(
					"A distância é uma medida de comparação, não uma porcentagem de confiança "
					"nem uma validação segura de identidade."
				)

	st.header("4. Reconhecimento pela webcam")
	st.caption("Na janela da câmera, pressione c para capturar o rosto ou q/Esc para encerrar.")
	if st.button("Capturar e reconhecer pela webcam", key="reconhecimento_webcam"):
		reconhecedor = st.session_state.reconhecedor
		if reconhecedor is None:
			st.warning("Treine o reconhecedor antes de usar a webcam.")
		else:
			try:
				rosto = capturar_rosto_webcam()
				if rosto is None:
					st.info("Captura encerrada sem fotografar um rosto.")
				else:
					pessoa, distancia = reconhecer_imagem(rosto, reconhecedor)
					st.image(
						cv2.cvtColor(rosto, cv2.COLOR_BGR2RGB),
						caption="Recorte facial analisado",
						alt="Recorte do rosto capturado pela webcam",
					)
					st.subheader("Pessoa cadastrada mais próxima")
					st.write(f"**Nome:** {pessoa}")
					st.write(f"**Distância euclidiana:** {distancia:.6f}")
					st.caption(
						"O Haar Cascade localiza o rosto; a identidade é comparada pelo PCA/Eigenfaces. "
						"A distância não valida a identidade com segurança."
					)
			except (ValueError, RuntimeError, cv2.error) as erro:
				st.error(f"Não foi possível capturar ou reconhecer o rosto: {erro}")


if __name__ == "__main__":
	main()
