
# FastAPI Flower Classification

A FastAPI-based application that classifies flower images using a pre-trained Vision Transformer (ViT) model. The application provides an API for image classification and is containerized using Docker for easy deployment.


## Installation


```bash
  git clone Flower-Classification
  cd Flower-Classification
```
```bash
docker build -t flower-classification .

docker run -p 8000:8000 flower-classification

```

Access the application at http://localhost:8000/



    
## Run Locally
To run the application locally without Docker, you can use uvicorn:

```bash
 pip install -r requirements.txt
```


```bash
  uvicorn app:app --reload
```





## Acknowledgements

- Vision Transformer (ViT) for the pre-trained model.
- FastAPI: for building fast APIs.
- PyTorch: for providing the deep learning framework.
- Docker: for containerizing the application.
