# 🚀 Guía de Inicio Rápido

## Para Usuarios Nuevos

### ¿Qué es esta aplicación?

Esta es una aplicación web interactiva que presenta un cuestionario de 10 preguntas para profesionales de la salud. El objetivo es recopilar información sobre el uso de tecnología e inteligencia artificial en el diagnóstico médico.

### ¿Cómo empezar?

#### Opción Más Rápida: Docker Compose

Si tienes Docker instalado, simplemente ejecuta:

```bash
git clone https://github.com/JhonattanReales21/mock_ml_app_w_docker.git
cd mock_ml_app_w_docker
docker-compose up
```

Luego abre tu navegador en: http://localhost:8501

#### Opción Alternativa: Docker

```bash
git clone https://github.com/JhonattanReales21/mock_ml_app_w_docker.git
cd mock_ml_app_w_docker
docker build -t health-app .
docker run -p 8501:8501 health-app
```

Luego abre tu navegador en: http://localhost:8501

#### Opción Local: Python

```bash
git clone https://github.com/JhonattanReales21/mock_ml_app_w_docker.git
cd mock_ml_app_w_docker
pip install -r requirements.txt
streamlit run app.py
```

Luego abre tu navegador en: http://localhost:8501

## Para Desarrolladores

### Estructura del Código

```
├── app.py                  # App Streamlit principal
├── source/                 # Módulos de ML
│   ├── image_processing.py
│   ├── text_processing.py
│   ├── cnn_models.py
│   ├── training.py
│   └── inference.py
├── notebooks/             # Para experimentación
├── Dockerfile            # Containerización
└── docker-compose.yml    # Orquestación
```

### Verificar el Entorno

```bash
./check_env.sh
```

### Probar los Módulos

```bash
python test_modules.py
```

### Modificar la Aplicación

1. Edita `app.py` para cambiar las preguntas
2. Edita módulos en `source/` para cambiar la lógica ML
3. Reconstruye el contenedor Docker si es necesario:
   ```bash
   docker-compose up --build
   ```

## Preguntas Frecuentes

### ¿Dónde se guardan las respuestas?

Las respuestas se pueden descargar como archivo CSV. No se almacenan automáticamente en una base de datos.

### ¿Puedo personalizar las preguntas?

Sí, edita el archivo `app.py` y modifica el contenido dentro del bloque `with st.form()`.

### ¿Cómo cambio el puerto?

**Docker:**
```bash
docker run -p 8080:8501 health-app
```

**Python local:**
```bash
streamlit run app.py --server.port=8080
```

**Docker Compose:** Edita `docker-compose.yml` y cambia el mapeo de puertos.

### ¿Los módulos ML funcionan sin datos?

Los módulos están completos y funcionales. Necesitarás proporcionar tus propios datasets de imágenes médicas para entrenamiento. Los módulos incluyen:
- Preprocesamiento de imágenes
- Modelos CNN pre-definidos
- Pipeline de entrenamiento
- Sistema de inferencia

## Recursos Adicionales

- **README.md**: Documentación completa
- **PROJECT_SUMMARY.md**: Resumen técnico del proyecto
- **check_env.sh**: Verificador de dependencias
- **test_modules.py**: Tests de los módulos

## Soporte

Si encuentras algún problema:
1. Verifica que tienes las dependencias instaladas
2. Revisa los logs de Docker si usas contenedores
3. Abre un issue en GitHub con detalles del error

## Próximos Pasos

1. ✅ Ejecutar la aplicación
2. ✅ Completar el cuestionario
3. ✅ Explorar los módulos de ML en `source/`
4. 📝 Experimentar en notebooks de Jupyter
5. 🔧 Personalizar según tus necesidades
6. 🚀 Desplegar en producción

---

**¡Disfruta usando la aplicación!** 🎉
