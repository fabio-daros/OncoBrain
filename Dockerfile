# Usar imagem oficial do Python 3.12 slim (leve) — ideal para CPU
FROM python:3.12-slim

# Alternativa para suporte a GPU futuramente:
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Diretório de trabalho
WORKDIR /app

# Copiar apenas o requirements.txt primeiro para aproveitar cache de builds
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Agora copiar o restante do código
COPY . .

# Expor a porta que a API vai usar
EXPOSE 8000

# Comando para iniciar a aplicação FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
