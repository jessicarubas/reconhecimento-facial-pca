"""Captura frames da webcam e recorta o maior rosto detectado pelo Haar Cascade."""

from threading import Lock

import cv2
import numpy as np


_CAPTURA_ATIVA = Lock()
_NOME_JANELA = "Captura facial - c: capturar | q/Esc: encerrar"
_ARQUIVO_CASCADE = "haarcascade_frontalface_default.xml"


def carregar_detector_facial():
	"""Carrega o Haar Cascade distribuído junto com a instalação do OpenCV."""
	caminho = cv2.data.haarcascades + _ARQUIVO_CASCADE
	detector = cv2.CascadeClassifier(caminho)
	if detector.empty():
		raise RuntimeError(f"Não foi possível carregar o detector facial: {caminho}")
	return detector


def localizar_rosto(imagem, detector):
	"""Retorna a caixa do maior rosto detectado em uma imagem BGR, se houver."""
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("O frame da câmera está vazio ou inválido.")
	if imagem.ndim != 3 or imagem.shape[2] != 3:
		raise ValueError("O frame da câmera deve ter três canais BGR.")

	cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	rostos = detector.detectMultiScale(
		cinza,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
	)
	if len(rostos) == 0:
		return None

	x, y, largura, altura = max(rostos, key=lambda caixa: caixa[2] * caixa[3])
	return int(x), int(y), int(largura), int(altura)


def recortar_rosto(imagem, caixa):
	"""Retorna uma cópia do recorte facial BGR para o processamento existente."""
	if caixa is None:
		return None
	x, y, largura, altura = caixa
	rosto = imagem[y:y + altura, x:x + largura]
	if rosto.size == 0:
		raise ValueError("O rosto detectado não gerou uma região válida na imagem.")
	return rosto.copy()


def capturar_rosto_webcam(indice_camera=0):
	"""Abre uma única webcam, permite capturar um rosto ou encerrar com q/Esc.

	Retorna o recorte facial BGR ao pressionar c, ou None ao encerrar sem captura.
	A câmera e a janela são liberadas mesmo quando a abertura ou leitura falha.
	"""
	if not _CAPTURA_ATIVA.acquire(blocking=False):
		raise RuntimeError("Já existe uma captura de webcam ativa nesta aplicação.")

	captura = None
	try:
		detector = carregar_detector_facial()
		try:
			captura = cv2.VideoCapture(indice_camera)
		except cv2.error as erro:
			raise RuntimeError("Não foi possível iniciar a câmera. Verifique se está disponível.") from erro

		if captura is None or not captura.isOpened():
			raise RuntimeError(
				"Não foi possível abrir a webcam. Verifique se está conectada, "
				"se outra aplicação a está utilizando e tente novamente."
			)

		while True:
			leitura_ok, frame = captura.read()
			if not leitura_ok or frame is None:
				raise RuntimeError("A webcam foi aberta, mas não foi possível ler um frame.")

			caixa = localizar_rosto(frame, detector)
			frame_exibicao = frame.copy()
			if caixa is not None:
				x, y, largura, altura = caixa
				cv2.rectangle(frame_exibicao, (x, y), (x + largura, y + altura), (0, 200, 0), 2)
				cv2.putText(
					frame_exibicao,
					"Rosto detectado - pressione c para capturar",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 200, 0),
					2,
				)
			else:
				cv2.putText(
					frame_exibicao,
					"Nenhum rosto detectado - pressione q para encerrar",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 0, 255),
					2,
				)

			cv2.imshow(_NOME_JANELA, frame_exibicao)
			tecla = cv2.waitKey(1) & 0xFF
			if tecla in (ord("q"), 27):
				return None
			if tecla == ord("c"):
				if caixa is None:
					raise ValueError("Nenhum rosto foi detectado. Posicione o rosto e tente novamente.")
				return recortar_rosto(frame, caixa)
	finally:
		if captura is not None:
			captura.release()
		try:
			cv2.destroyWindow(_NOME_JANELA)
		except cv2.error:
			pass
		_CAPTURA_ATIVA.release()