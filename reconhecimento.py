import numpy as np
from sklearn.decomposition import PCA


TAMANHO_VETOR = 64 * 64


class ReconhecedorFacial:
	"""Treina um modelo Eigenfaces e compara imagens no espaço da PCA.

	Cada imagem processada é um vetor de 4096 pixels. O PCA resume a variação
	do conjunto em componentes principais, e a identificação escolhe a imagem
	cadastrada com projeção de menor distância euclidiana.
	"""

	def __init__(self, n_componentes=50):
		"""Configura o número máximo de componentes principais a usar."""
		if not isinstance(n_componentes, int) or isinstance(n_componentes, bool):
			raise ValueError("A quantidade de componentes deve ser um número inteiro.")
		if n_componentes <= 0:
			raise ValueError("A quantidade de componentes deve ser maior que zero.")

		self.n_componentes = n_componentes
		self.face_media_ = None
		self.imagens_projetadas_ = None
		self.identificadores_ = None
		self.pca_ = None
		self.n_componentes_ = None

	def _validar_matriz_treinamento(self, imagens):
		"""Valida a matriz X, organizada com uma imagem por linha.

		Para N imagens de 4096 pixels, X tem dimensão N x 4096. Cada coluna
		representa a mesma posição de pixel em todas as imagens; essa organização
		permite calcular médias e variações por pixel.
		"""
		try:
			matriz = np.asarray(imagens, dtype=np.float64)
		except (TypeError, ValueError):
			raise ValueError("As imagens devem ser uma matriz de valores numéricos.") from None

		if matriz.ndim != 2:
			raise ValueError("Informe uma matriz 2D: cada linha deve representar uma imagem.")
		if matriz.shape[0] == 0:
			raise ValueError("O conjunto de treinamento não pode estar vazio.")
		if matriz.shape[1] != TAMANHO_VETOR:
			raise ValueError(f"Cada imagem deve conter exatamente {TAMANHO_VETOR} valores.")
		if matriz.shape[0] < 2:
			raise ValueError("São necessárias pelo menos duas imagens para treinar o PCA.")
		if not np.all(np.isfinite(matriz)):
			raise ValueError("As imagens contêm valores inválidos (NaN ou infinito).")

		return matriz

	def treinar(self, imagens, identificadores):
		"""Ajusta o PCA às imagens e guarda as projeções junto aos seus nomes.

		A ordem dos identificadores deve corresponder à ordem das linhas da matriz.
		O resultado é uma matriz menor: uma linha por imagem e uma coluna por
		componente principal selecionado.
		"""
		matriz = self._validar_matriz_treinamento(imagens)
		try:
			rotulos = list(identificadores)
		except TypeError:
			raise ValueError("Forneça um identificador para cada imagem.") from None

		if len(rotulos) != matriz.shape[0]:
			raise ValueError("A quantidade de identificadores deve ser igual à quantidade de imagens.")

		# X reúne as imagens como vetores: linhas são faces, colunas são posições de pixel.
		# A média por coluna forma a face média, que resume a intensidade média em cada posição.
		face_media = np.mean(matriz, axis=0)

		# X_c = X - mu: subtrair a média desloca os dados para que o PCA analise variações
		# em torno da face média, em vez de tratar o brilho médio como uma variação principal.
		imagens_centralizadas = matriz - face_media

		# Com N imagens centralizadas, há no máximo N-1 direções independentes de variação.
		n_componentes = min(self.n_componentes, matriz.shape[0] - 1, matriz.shape[1])
		modelo_pca = PCA(n_components=n_componentes)
		imagens_projetadas = modelo_pca.fit_transform(imagens_centralizadas)

		# Os componentes principais são direções de maior variação: teoricamente, autovetores
		# da matriz de covariância; seus autovalores indicam a variância de cada direção.
		# O Scikit-learn obtém essas direções por decomposição SVD equivalente.
		# fit_transform ajusta essas direções e projeta cada face em suas novas coordenadas.
		self.face_media_ = face_media
		self.imagens_projetadas_ = imagens_projetadas
		self.identificadores_ = rotulos
		self.pca_ = modelo_pca
		self.n_componentes_ = n_componentes

		return imagens_projetadas

	def obter_eigenfaces(self):
		"""Retorna os componentes PCA como matrizes de 64 x 64 pixels.

		Cada Eigenface é um padrão de pesos aprendido nas posições dos pixels,
		associado a uma direção importante de variação entre as faces do treino.
		Não representa, por si só, a fotografia de uma pessoa cadastrada.
		"""
		if self.pca_ is None:
			raise RuntimeError("Treine o reconhecedor antes de obter as Eigenfaces.")

		# Reorganizar os 4096 pesos do componente permite visualizar seu padrão espacial.
		return self.pca_.components_.reshape(self.n_componentes_, 64, 64)

	def reconhecer(self, imagem):
		"""Encontra o cadastro mais próximo e retorna seu identificador e distância.

		A nova imagem passa pela mesma centralização e projeção aprendidas no treino,
		para que a comparação seja feita nas mesmas coordenadas reduzidas.
		"""
		if self.pca_ is None:
			raise RuntimeError("Treine o reconhecedor antes de reconhecer uma imagem.")

		try:
			vetor = np.asarray(imagem, dtype=np.float64)
		except (TypeError, ValueError):
			raise ValueError("A imagem deve ser um vetor numérico com 4096 valores.") from None

		if vetor.ndim != 1 or vetor.size != TAMANHO_VETOR:
			raise ValueError(f"A nova imagem deve ser um vetor com exatamente {TAMANHO_VETOR} valores.")
		if not np.all(np.isfinite(vetor)):
			raise ValueError("A imagem contém valores inválidos (NaN ou infinito).")

		imagem_centralizada = vetor - self.face_media_
		projecao = self.pca_.transform(imagem_centralizada.reshape(1, -1))[0]

		# A norma L2 mede a distância reta entre a nova projeção e cada face cadastrada.
		distancias = np.linalg.norm(self.imagens_projetadas_ - projecao, axis=1)
		indice_mais_proximo = int(np.argmin(distancias))

		# O índice da menor distância aponta para o nome associado àquela linha de treino.
		return self.identificadores_[indice_mais_proximo], float(distancias[indice_mais_proximo])
