
import numpy as np
from reconhecimento import ReconhecedorFacial

# Criamos imagens fictícias de três pessoas.
np.random.seed(42)

pessoa_1 = np.random.rand(3, 4096)
pessoa_2 = np.random.rand(3, 4096)
pessoa_3 = np.random.rand(3, 4096)

# Juntamos as imagens em uma única matriz.
imagens = np.vstack([pessoa_1, pessoa_2, pessoa_3])

# Cada imagem recebe um identificador.
identificadores = [
    "Pessoa 1", "Pessoa 1", "Pessoa 1",
    "Pessoa 2", "Pessoa 2", "Pessoa 2",
    "Pessoa 3", "Pessoa 3", "Pessoa 3"
]

# Criamos e treinamos o reconhecedor.
reconhecedor = ReconhecedorFacial(n_componentes=5)

projecoes = reconhecedor.treinar(imagens, identificadores)

# Verificamos as dimensões das projeções.
print("Formato das projeções:", projecoes.shape)

# Verificamos as Eigenfaces.
eigenfaces = reconhecedor.obter_eigenfaces()
print("Formato das Eigenfaces:", eigenfaces.shape)

# Usamos uma imagem que já está no conjunto de treinamento.
nova_imagem = imagens[0]

pessoa, distancia = reconhecedor.reconhecer(nova_imagem)

print("Pessoa mais próxima:", pessoa)
print("Distância encontrada:", distancia)
