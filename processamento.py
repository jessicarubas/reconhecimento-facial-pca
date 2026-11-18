"""Validação e conversão de fotografias para vetores de pixels padronizados."""

import cv2
import numpy as np


LARGURA_PADRAO = 64
ALTURA_PADRAO = 64


def carregar_imagem(imagem):
	"""Recebe uma matriz NumPy ou carrega uma imagem a partir de um caminho.

	Após a leitura, valida os dados para que as próximas operações do OpenCV
	recebam uma imagem com formato e valores de pixel compatíveis.
	"""
	if isinstance(imagem, str):
		imagem = cv2.imread(imagem, cv2.IMREAD_UNCHANGED)
		if imagem is None:
			raise ValueError("Não foi possível abrir a imagem. Verifique o caminho e o formato do arquivo.")

	validar_imagem(imagem)
	return imagem


def validar_imagem(imagem):
	"""Confirma que a imagem não está vazia e tem forma, canais e pixels válidos.

	As dimensões precisam representar uma imagem em tons de cinza ou com 1, 3
	ou 4 canais; valores não finitos e pixels fora de 0 a 255 são rejeitados.
	"""
	if imagem is None:
		raise ValueError("A imagem não foi fornecida ou não pôde ser carregada.")
	if not isinstance(imagem, np.ndarray):
		raise ValueError("A imagem deve ser uma matriz NumPy ou o caminho de um arquivo.")
	if imagem.size == 0:
		raise ValueError("A imagem está vazia.")
	if imagem.ndim not in (2, 3):
		raise ValueError("A imagem deve ter duas dimensões ou incluir canais de cor.")
	if imagem.ndim == 3 and imagem.shape[2] not in (1, 3, 4):
		raise ValueError("A imagem deve ter 1, 3 ou 4 canais de cor.")
	if not np.issubdtype(imagem.dtype, np.number) or np.iscomplexobj(imagem):
		raise ValueError("Os pixels da imagem devem ter valores numéricos.")
	if not np.all(np.isfinite(imagem)):
		raise ValueError("A imagem contém valores de pixel inválidos.")
	if np.min(imagem) < 0 or np.max(imagem) > 255:
		raise ValueError("Os pixels devem estar entre 0 e 255 (ou entre 0 e 1 para imagens normalizadas).")


def converter_para_cinza(imagem):
	"""Converte BGR/BGRA em intensidade de cinza e preserva imagens já cinzas.

	A conversão combina os canais de cor em um valor de intensidade por pixel.
	Isso reduz cada imagem colorida a um único canal, mantendo sua estrutura
espacial para as etapas seguintes.
	"""
	imagem = carregar_imagem(imagem)

	if imagem.ndim == 2:
		return imagem
	if imagem.shape[2] == 1:
		return imagem[:, :, 0]
	if imagem.shape[2] == 3:
		# O código de cor explicita que os canais recebidos pelo OpenCV estão em ordem BGR.
		return cv2.cvtColor(imagem.astype(np.float32), cv2.COLOR_BGR2GRAY)
	return cv2.cvtColor(imagem.astype(np.float32), cv2.COLOR_BGRA2GRAY)


def redimensionar_imagem(imagem, largura=LARGURA_PADRAO, altura=ALTURA_PADRAO):
	"""Padroniza a resolução para que todas as imagens tenham o mesmo número de pixels.

	Com o padrão de 64 por 64, cada face terá 4096 posições, requisito para
organizar as imagens em uma matriz de dados com colunas correspondentes.
	"""
	if largura <= 0 or altura <= 0:
		raise ValueError("A largura e a altura devem ser maiores que zero.")

	imagem = carregar_imagem(imagem).astype(np.float32)
	# INTER_AREA combina pixels vizinhos ao reduzir a imagem para a resolução-alvo.
	return cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)


def normalizar_imagem(imagem):
	"""Coloca os pixels em uma escala numérica comum entre 0 e 1.

	Para pixels usuais de 0 a 255, a transformação é I_norm = I / 255.
	Se os dados já estiverem normalizados, eles não são divididos novamente.
	"""
	imagem = carregar_imagem(imagem).astype(np.float32)

	# Matrizes já normalizadas não precisam ser divididas novamente por 255.
	divisor = 1.0 if np.max(imagem) <= 1.0 else 255.0
	return imagem / divisor


def imagem_para_vetor(imagem):
	"""Achata a matriz em um vetor NumPy, preservando a ordem dos pixels.

	Uma matriz 64 x 64 passa a ter 64 * 64 = 4096 coordenadas, cada uma
	correspondendo à intensidade de uma posição da imagem.
	"""
	return imagem.reshape(-1)


def processar_imagem(imagem):
	"""Valida, converte, redimensiona, normaliza e vetoriza uma imagem.

	O vetor final tem 4096 elementos e pode ser usado como uma observação
	na matriz de treinamento do PCA.
	"""
	imagem = carregar_imagem(imagem)
	# Primeiro uniformiza as cores, depois o tamanho e a escala dos pixels.
	imagem = converter_para_cinza(imagem)
	imagem = redimensionar_imagem(imagem)
	imagem = normalizar_imagem(imagem)
	# A matriz 64 x 64 vira uma sequência de 4096 valores.
	return imagem_para_vetor(imagem)
