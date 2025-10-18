# Aplicación de Cuestionario para Profesionales de la Salud con ML

## 📋 Descripción

Esta es una aplicación web desarrollada con Streamlit que presenta un cuestionario interactivo de 10 preguntas diseñado específicamente para profesionales de la salud. La aplicación recopila información sobre el uso de tecnología y herramientas de inteligencia artificial en el diagnóstico médico.

### Finalidad de la Solución

La aplicación tiene como objetivo:
- **Recopilar datos** de profesionales de la salud sobre sus prácticas actuales
- **Evaluar** el nivel de adopción de tecnologías de IA en el sector salud
- **Identificar** necesidades de capacitación y barreras en la implementación de IA
- **Generar insights** sobre el uso de imágenes médicas y herramientas de diagnóstico

Además, incluye una estructura completa de procesamiento de machine learning con módulos para:
- Procesamiento de imágenes médicas
- Análisis de texto médico
- Modelos CNN (Convolutional Neural Networks)
- Entrenamiento de modelos
- Inferencia y predicciones

## 🏗️ Estructura del Proyecto

```
mock_ml_app_w_docker/
│
├── app.py                      # Aplicación principal de Streamlit
├── Dockerfile                  # Archivo de configuración Docker
├── requirements.txt            # Dependencias de Python
├── README.md                   # Este archivo
│
├── source/                     # Módulos de procesamiento ML
│   ├── __init__.py            
│   ├── image_processing.py    # Procesamiento de imágenes médicas
│   ├── text_processing.py     # Procesamiento de texto médico
│   ├── cnn_models.py          # Arquitecturas CNN
│   ├── training.py            # Entrenamiento de modelos
│   └── inference.py           # Inferencia y predicciones
│
└── notebooks/                  # Notebooks de Jupyter (vacío)
```

## 🚀 Instrucciones de Instalación y Uso

### Opción 1: Ejecución Local con Python

#### Prerrequisitos
- Python 3.10 o superior
- pip (gestor de paquetes de Python)

#### Pasos:

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/JhonattanReales21/mock_ml_app_w_docker.git
   cd mock_ml_app_w_docker
   ```

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En Windows:
   venv\Scripts\activate
   
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run app.py
   ```

5. **Acceder a la aplicación**
   
   La aplicación se abrirá automáticamente en su navegador. Si no, acceda manualmente a:
   ```
   http://localhost:8501
   ```

### Opción 2: Ejecución con Docker (Recomendado)

#### Prerrequisitos
- Docker instalado en su sistema ([Descargar Docker](https://www.docker.com/get-started))

#### Pasos:

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/JhonattanReales21/mock_ml_app_w_docker.git
   cd mock_ml_app_w_docker
   ```

2. **Construir la imagen Docker**
   ```bash
   docker build -t health-questionnaire-app .
   ```
   
   Este comando:
   - `-t health-questionnaire-app`: Asigna el nombre "health-questionnaire-app" a la imagen
   - `.`: Usa el Dockerfile en el directorio actual

3. **Ejecutar el contenedor**
   ```bash
   docker run -p 8501:8501 health-questionnaire-app
   ```
   
   Este comando:
   - `-p 8501:8501`: Mapea el puerto 8501 del contenedor al puerto 8501 de su máquina
   - `health-questionnaire-app`: Nombre de la imagen a ejecutar

4. **Acceder a la aplicación**
   
   Abra su navegador y vaya a:
   ```
   http://localhost:8501
   ```

#### Comandos Docker Adicionales

**Ejecutar en segundo plano (modo detached):**
```bash
docker run -d -p 8501:8501 --name health-app health-questionnaire-app
```

**Ver contenedores en ejecución:**
```bash
docker ps
```

**Detener el contenedor:**
```bash
docker stop health-app
```

**Ver logs del contenedor:**
```bash
docker logs health-app
```

**Eliminar el contenedor:**
```bash
docker rm health-app
```

**Eliminar la imagen:**
```bash
docker rmi health-questionnaire-app
```

## 📝 Uso de la Aplicación

1. **Completar el Cuestionario**: Responda las 10 preguntas del formulario
2. **Enviar Respuestas**: Haga clic en el botón "Enviar Respuestas"
3. **Ver Resumen**: Revise el resumen de sus respuestas
4. **Descargar Datos**: Opcionalmente, descargue sus respuestas en formato CSV

### Las 10 Preguntas del Cuestionario:

1. Especialidad médica
2. Años de experiencia
3. Frecuencia de uso de herramientas de diagnóstico por imágenes
4. Uso de herramientas de IA para diagnóstico
5. Tipos de imágenes médicas analizadas
6. Importancia de la IA en el diagnóstico (escala 1-10)
7. Principales desafíos en el diagnóstico médico
8. Comodidad con tecnología avanzada
9. Interés en capacitación sobre IA
10. Institución médica que representa

## 🧪 Módulos de Machine Learning

La carpeta `source/` contiene módulos completos para procesamiento de ML:

### `image_processing.py`
- Carga y redimensionamiento de imágenes
- Normalización de píxeles
- Aplicación de CLAHE para mejora de contraste
- Eliminación de ruido (bilateral, gaussiano, mediano)
- Aumento de datos (rotación, volteo)
- Pipeline completo de preprocesamiento

### `text_processing.py`
- Limpieza y normalización de texto médico
- Tokenización
- Extracción de términos médicos
- Anonimización de información del paciente
- Extracción de palabras clave
- Estadísticas de texto

### `cnn_models.py`
- SimpleCNN: Arquitectura CNN básica
- MedicalCNN: CNN avanzada con bloques residuales
- ResNetBlock: Bloques residuales para mejor aprendizaje
- Factory function para crear modelos

### `training.py`
- Dataset personalizado para imágenes médicas
- Loop de entrenamiento completo
- Validación de modelos
- Guardado y carga de modelos
- Scheduler de learning rate

### `inference.py`
- Predicción en imágenes individuales
- Predicción en lotes
- Estimación de incertidumbre con Monte Carlo dropout
- Top-K predicciones
- Explicación de predicciones
- Pipeline completo de inferencia

## 🔧 Configuración y Personalización

### Modificar el Puerto
Para cambiar el puerto de la aplicación, modifique el comando en el Dockerfile o al ejecutar:

```bash
# Python local
streamlit run app.py --server.port=8080

# Docker
docker run -p 8080:8080 health-questionnaire-app
```

### Agregar Nuevas Preguntas
Edite el archivo `app.py` y agregue nuevas preguntas dentro del bloque `with st.form()`.

## 📊 Obtener las Respuestas

Las respuestas se pueden obtener de dos formas:

1. **Descarga Individual**: Cada usuario puede descargar sus respuestas como archivo CSV
2. **Integración con Base de Datos**: Modifique `app.py` para guardar respuestas en una base de datos

## 🐛 Solución de Problemas

### Error: Puerto ya en uso
```bash
# Encontrar proceso usando el puerto 8501
# En Linux/Mac:
lsof -ti:8501 | xargs kill -9

# En Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Error de dependencias
```bash
pip install --upgrade -r requirements.txt
```

### Problemas con Docker
```bash
# Limpiar caché de Docker
docker system prune -a

# Reconstruir la imagen
docker build --no-cache -t health-questionnaire-app .
```

## 📦 Dependencias Principales

- **streamlit**: Framework para aplicaciones web
- **pandas**: Manipulación de datos
- **numpy**: Computación numérica
- **torch**: Deep learning framework
- **opencv-python**: Procesamiento de imágenes
- **Pillow**: Manipulación de imágenes

## 👥 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

## 📧 Contacto

Para preguntas o sugerencias sobre la aplicación, por favor abra un issue en el repositorio de GitHub.

---

**Desarrollado con ❤️ para profesionales de la salud** 
