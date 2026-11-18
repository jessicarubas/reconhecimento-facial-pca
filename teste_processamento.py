import numpy as np
from processamento import processar_imagem

# Cria uma imagem fictícia de 100 x 100 pixels
imagem = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

resultado = processar_imagem(imagem)

print("Formato do resultado:", resultado.shape)
print("Quantidade de pixels:", len(resultado))
print("Valor mínimo:", resultado.min())
print("Valor máximo:", resultado.max())
print("Tipo dos dados:", resultado.dtype)