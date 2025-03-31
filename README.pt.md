# Upper Limb Fracture Identifier

<p align="center">
  <img src="img/banner.jpg" alt="Banner" style="width: 100%; height: auto;">
</p>

## Idiomas:
<p align="center">
  <a href="README.md" style="display: inline-block; padding: 10px 20px; font-size: 16px; text-align: center; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">English</a>
</p>

---

 # Identificador de Fraturas em Membros Superiores

## Objetivo do Projeto

Este projeto tem como objetivo desenvolver um modelo de visão computacional capaz de identificar fraturas ósseas nos membros superiores a partir de imagens radiográficas. Inicialmente, a abordagem adotada foi a de um modelo binário de classificação para determinar se uma fratura estava presente ou não. Com a evolução do projeto, foi implementada a detecção de fraturas utilizando o modelo YOLO, permitindo não apenas a classificação, mas também a localização exata da fratura na imagem.

## Evolução do Projeto

### Fase 1: Modelo Binário

Na primeira fase, foi treinado um modelo de rede neural convolucional (CNN) para realizar a detecção binária de fraturas. O modelo foi alimentado com um conjunto de imagens radiográficas classificadas em duas categorias:

- **Com fratura**
- **Sem fratura**

A partir desse modelo, foi possível prever se uma nova imagem continha ou não uma fratura. No entanto, essa abordagem apresentava limitações, pois não fornecia informações sobre a localização exata da fratura, dificultando a interpretação clínica.

### Fase 2: Implementação do YOLO para Detecção de Fraturas

Para superar as limitações da abordagem inicial, a segunda fase do projeto envolveu a transição para um modelo **YOLO (You Only Look Once)**, que permite a detecção e localização precisa das fraturas nas imagens.

Foi utilizado um novo conjunto de dados, encontrado no Kaggle, contendo imagens anotadas com as regiões de fratura. O dataset incluía três subdivisões (train, test, val) e possuía anotações para sete categorias de fraturas:

- **Elbow Positive**
- **Fingers Positive**
- **Forearm Fracture**
- **Humerus Fracture**
- **Humerus**
- **Shoulder Fracture**
- **Wrist Positive**

O treinamento do modelo YOLO foi realizado com essa base de dados, permitindo não apenas detectar a presença de fraturas, mas também localizar sua posição na imagem.

## Processo de Maturação

### 1. Preparação dos Dados

Antes do treinamento, foi necessário realizar um pré-processamento das imagens para garantir a qualidade dos dados. Para isso, foi implementada uma função de pré-processamento utilizando a biblioteca **OpenCV**:

python
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

    # Converter para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Equalizar histograma para melhorar o contraste
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Aplicar filtro de suavização para reduzir ruído
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    
    # Converter de volta para RGB para compatibilidade com YOLO
    processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    
    return processed_image


### 2. Treinamento do Modelo YOLO

O modelo YOLO foi treinado utilizando a base de dados estruturada, garantindo que ele aprendesse a identificar diferentes tipos de fraturas ósseas nos membros superiores. O treinamento exigiu ajustes em hiperparâmetros, incluindo número de épocas e tamanho do batch, para otimizar a acurácia sem comprometer a velocidade de inferência.

### 3. Implementação da Detecção e Classificação

Com o modelo treinado, foi possível desenvolver um pipeline que realiza as seguintes etapas:

1. Carregamento e pré-processamento da imagem.
2. Aplicação do modelo YOLO para detecção de fraturas.
3. Exibição das regiões afetadas por fraturas com caixas delimitadoras (bounding boxes).
4. Exibição do nome da classe correspondente à fratura.
5. Se nenhuma fratura for detectada, a imagem é classificada como "sem fratura".

## Conclusão e Próximos Passos

O projeto evoluiu de uma simples classificação binária para um sistema robusto de detecção e localização de fraturas ósseas nos membros superiores. Futuras melhorias podem incluir:

- Expansão do modelo para detectar fraturas em outras partes do corpo.
- Aprimoramento da base de dados com imagens anotadas por especialistas.
- Implementação de um sistema de apoio à decisão médica baseado nas detecções do modelo.

Esse projeto representa um avanço significativo na utilização de visão computacional para suporte diagnóstico em radiologia, contribuindo para uma identificação mais precisa e rápida de fraturas ósseas.