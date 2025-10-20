import streamlit as st
import pandas as pd
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Cuestionario para Profesionales de la Salud",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Cuestionario para Profesionales de la Salud")
st.markdown("""
Este cuestionario está diseñado para recopilar información de profesionales de la salud 
sobre sus prácticas y uso de tecnología en el diagnóstico médico.
""")

# Create a form
with st.form("health_professional_form"):
    st.subheader("Por favor responda las siguientes preguntas:")
    
    # Question 1
    q1 = st.text_input(
        "1. ¿Cuál es su especialidad médica?",
        placeholder="Ej: Radiología, Cardiología, etc."
    )
    
    # Question 2
    q2 = st.number_input(
        "2. ¿Cuántos años de experiencia tiene en su especialidad?",
        min_value=0,
        max_value=50,
        value=0
    )
    
    # Question 3
    q3 = st.selectbox(
        "3. ¿Con qué frecuencia utiliza herramientas de diagnóstico por imágenes?",
        ["Diariamente", "Semanalmente", "Mensualmente", "Raramente", "Nunca"]
    )
    
    # Question 4
    q4 = st.radio(
        "4. ¿Ha utilizado alguna herramienta de inteligencia artificial para diagnóstico?",
        ["Sí", "No", "En proceso de implementación"]
    )
    
    # Question 5
    q5 = st.multiselect(
        "5. ¿Qué tipos de imágenes médicas analiza con más frecuencia?",
        ["Rayos X", "Tomografía Computarizada (CT)", "Resonancia Magnética (MRI)", 
         "Ecografía", "Mamografía", "PET Scan", "Otro"]
    )
    
    # Question 6
    q6 = st.slider(
        "6. En una escala del 1 al 10, ¿qué tan importante considera la IA en el diagnóstico médico?",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Question 7
    q7 = st.text_area(
        "7. ¿Cuáles son los principales desafíos que enfrenta en el diagnóstico médico?",
        placeholder="Describa brevemente los desafíos..."
    )
    
    # Question 8
    q8 = st.selectbox(
        "8. ¿Qué tan cómodo se siente utilizando tecnología avanzada en su práctica?",
        ["Muy cómodo", "Cómodo", "Neutral", "Incómodo", "Muy incómodo"]
    )
    
    # Question 9
    q9 = st.radio(
        "9. ¿Estaría interesado en recibir capacitación sobre herramientas de IA para diagnóstico?",
        ["Sí, definitivamente", "Tal vez", "No estoy seguro", "No"]
    )
    
    # Question 10
    q10 = st.text_input(
        "10. ¿Qué institución médica o hospital representa?",
        placeholder="Nombre de la institución"
    )
    
    # Submit button
    submitted = st.form_submit_button("Enviar Respuestas")
    
    if submitted:
        # Validate that at least some questions are answered
        if not q1 or not q10:
            st.error("Por favor complete al menos las preguntas 1 y 10 (especialidad e institución).")
        else:
            # Create a dictionary with all responses
            responses = {
                "Fecha y Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Especialidad": q1,
                "Años de Experiencia": q2,
                "Frecuencia Uso Imágenes": q3,
                "Uso de IA": q4,
                "Tipos de Imágenes": ", ".join(q5) if q5 else "N/A",
                "Importancia IA (1-10)": q6,
                "Desafíos": q7,
                "Comodidad con Tecnología": q8,
                "Interés en Capacitación": q9,
                "Institución": q10
            }
            
            # Display success message
            st.success("¡Gracias por completar el cuestionario!")
            
            # Show a summary of responses
            st.subheader("Resumen de sus respuestas:")
            for key, value in responses.items():
                st.write(f"**{key}:** {value}")
            
            # Create a downloadable dataframe
            df = pd.DataFrame([responses])
            
            # Offer download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar respuestas como CSV",
                data=csv,
                file_name=f"respuestas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("*Aplicación desarrollada con Streamlit para la recopilación de datos de profesionales de la salud*")
