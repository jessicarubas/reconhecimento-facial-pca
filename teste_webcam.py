import unittest
from threading import Event, Thread
from unittest.mock import Mock, patch

import numpy as np

from processamento import processar_imagem
from reconhecimento import ReconhecedorFacial
from webcam import capturar_rosto_webcam, localizar_rosto, recortar_rosto


class TesteWebcam(unittest.TestCase):
	def setUp(self):
		self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
		self.detector = Mock()

	def test_localiza_maior_rosto(self):
		self.detector.detectMultiScale.return_value = np.array([
			[5, 8, 30, 30],
			[40, 20, 70, 80],
		])

		caixa = localizar_rosto(self.frame, self.detector)

		self.assertEqual(caixa, (40, 20, 70, 80))
		self.detector.detectMultiScale.assert_called_once()

	def test_sem_rosto_retorna_none(self):
		self.detector.detectMultiScale.return_value = ()

		self.assertIsNone(localizar_rosto(self.frame, self.detector))

	def test_recorta_apenas_a_regiao_detectada(self):
		rosto = recortar_rosto(self.frame, (10, 15, 35, 40))

		self.assertEqual(rosto.shape, (40, 35, 3))
		self.assertFalse(np.shares_memory(rosto, self.frame))

	@patch("webcam.cv2.waitKey", return_value=ord("c"))
	@patch("webcam.cv2.imshow")
	@patch("webcam.carregar_detector_facial")
	@patch("webcam.cv2.VideoCapture")
	def test_captura_recorte_libera_camera(
		self, video_capture, carregar_detector, _imshow, _wait_key
	):
		captura = Mock()
		captura.isOpened.return_value = True
		captura.read.return_value = (True, self.frame)
		video_capture.return_value = captura
		carregar_detector.return_value = self.detector
		self.detector.detectMultiScale.return_value = np.array([[10, 15, 35, 40]])

		rosto = capturar_rosto_webcam()

		self.assertEqual(rosto.shape, (40, 35, 3))
		video_capture.assert_called_once_with(0)
		captura.release.assert_called_once()

	@patch("webcam.cv2.destroyWindow")
	@patch("webcam.carregar_detector_facial")
	@patch("webcam.cv2.VideoCapture")
	def test_camera_indisponivel_e_liberada(self, video_capture, carregar_detector, _destroy):
		captura = Mock()
		captura.isOpened.return_value = False
		video_capture.return_value = captura
		carregar_detector.return_value = self.detector

		with self.assertRaisesRegex(RuntimeError, "Não foi possível abrir a webcam"):
			capturar_rosto_webcam()

		captura.release.assert_called_once()

	@patch("webcam.cv2.waitKey", return_value=ord("c"))
	@patch("webcam.cv2.imshow")
	@patch("webcam.carregar_detector_facial")
	@patch("webcam.cv2.VideoCapture")
	def test_captura_sem_rosto_gera_erro_e_libera_camera(
		self, video_capture, carregar_detector, _imshow, _wait_key
	):
		captura = Mock()
		captura.isOpened.return_value = True
		captura.read.return_value = (True, self.frame)
		video_capture.return_value = captura
		carregar_detector.return_value = self.detector
		self.detector.detectMultiScale.return_value = ()

		with self.assertRaisesRegex(ValueError, "Nenhum rosto foi detectado"):
			capturar_rosto_webcam()

		captura.release.assert_called_once()

	@patch("webcam.cv2.waitKey", return_value=ord("q"))
	@patch("webcam.cv2.imshow")
	@patch("webcam.carregar_detector_facial")
	@patch("webcam.cv2.VideoCapture")
	def test_tecla_q_encerra_sem_capturar_e_libera_camera(
		self, video_capture, carregar_detector, _imshow, _wait_key
	):
		captura = Mock()
		captura.isOpened.return_value = True
		captura.read.return_value = (True, self.frame)
		video_capture.return_value = captura
		carregar_detector.return_value = self.detector
		self.detector.detectMultiScale.return_value = ()

		self.assertIsNone(capturar_rosto_webcam())
		captura.release.assert_called_once()

	@patch("webcam.cv2.waitKey")
	@patch("webcam.cv2.imshow")
	@patch("webcam.carregar_detector_facial")
	@patch("webcam.cv2.VideoCapture")
	def test_nao_abre_segunda_camera_durante_captura_ativa(
		self, video_capture, carregar_detector, _imshow, wait_key
	):
		captura = Mock()
		captura.isOpened.return_value = True
		captura.read.return_value = (True, self.frame)
		video_capture.return_value = captura
		carregar_detector.return_value = self.detector
		self.detector.detectMultiScale.return_value = ()
		tecla_aguardando = Event()
		encerrar_espera = Event()
		falhas_thread = []

		def aguardar_encerramento(_atraso):
			tecla_aguardando.set()
			encerrar_espera.wait(timeout=5)
			return ord("q")

		def executar_captura():
			try:
				capturar_rosto_webcam()
			except Exception as erro:
				falhas_thread.append(erro)

		wait_key.side_effect = aguardar_encerramento
		thread = Thread(target=executar_captura)
		thread.start()
		try:
			self.assertTrue(tecla_aguardando.wait(timeout=5))
			with self.assertRaisesRegex(RuntimeError, "captura de webcam ativa"):
				capturar_rosto_webcam()
			video_capture.assert_called_once_with(0)
		finally:
			encerrar_espera.set()
			thread.join(timeout=5)

		self.assertFalse(thread.is_alive())
		self.assertEqual(falhas_thread, [])
		captura.release.assert_called_once()

	def test_recorte_usa_processamento_e_reconhecedor_existentes(self):
		self.detector.detectMultiScale.return_value = np.array([[10, 15, 35, 40]])
		rosto = recortar_rosto(self.frame, localizar_rosto(self.frame, self.detector))
		vetor = processar_imagem(rosto)

		imagens_treino = np.random.default_rng(42).random((3, 4096))
		modelo = ReconhecedorFacial(n_componentes=2)
		modelo.treinar(imagens_treino, ["A", "B", "C"])
		pessoa, distancia = modelo.reconhecer(imagens_treino[0])

		self.assertEqual(vetor.shape, (4096,))
		self.assertGreaterEqual(float(vetor.min()), 0.0)
		self.assertLessEqual(float(vetor.max()), 1.0)
		self.assertEqual(pessoa, "A")
		self.assertAlmostEqual(distancia, 0.0, places=10)


if __name__ == "__main__":
	unittest.main()