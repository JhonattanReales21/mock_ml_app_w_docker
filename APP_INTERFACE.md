# 📱 Streamlit App Interface Preview

## What the App Looks Like

When users access http://localhost:8501, they will see:

### Header Section
```
🏥 Cuestionario para Profesionales de la Salud
═══════════════════════════════════════════════

Este cuestionario está diseñado para recopilar información de profesionales 
de la salud sobre sus prácticas y uso de tecnología en el diagnóstico médico.
```

### The Form
```
Por favor responda las siguientes preguntas:
─────────────────────────────────────────────

1. ¿Cuál es su especialidad médica?
   [Text Input Box: Ej: Radiología, Cardiología, etc.]

2. ¿Cuántos años de experiencia tiene en su especialidad?
   [Number Input: ▼0▲] (0-50 range)

3. ¿Con qué frecuencia utiliza herramientas de diagnóstico por imágenes?
   [Dropdown: ▼ Diariamente]
   Options: Diariamente, Semanalmente, Mensualmente, Raramente, Nunca

4. ¿Ha utilizado alguna herramienta de inteligencia artificial para diagnóstico?
   ○ Sí
   ○ No
   ○ En proceso de implementación

5. ¿Qué tipos de imágenes médicas analiza con más frecuencia?
   ☐ Rayos X
   ☐ Tomografía Computarizada (CT)
   ☐ Resonancia Magnética (MRI)
   ☐ Ecografía
   ☐ Mamografía
   ☐ PET Scan
   ☐ Otro

6. En una escala del 1 al 10, ¿qué tan importante considera la IA en el diagnóstico médico?
   [━━━━━●━━━━━] 5

7. ¿Cuáles son los principales desafíos que enfrenta en el diagnóstico médico?
   [Text Area]
   Describa brevemente los desafíos...

8. ¿Qué tan cómodo se siente utilizando tecnología avanzada en su práctica?
   [Dropdown: ▼ Muy cómodo]
   Options: Muy cómodo, Cómodo, Neutral, Incómodo, Muy incómodo

9. ¿Estaría interesado en recibir capacitación sobre herramientas de IA para diagnóstico?
   ○ Sí, definitivamente
   ○ Tal vez
   ○ No estoy seguro
   ○ No

10. ¿Qué institución médica o hospital representa?
    [Text Input: Nombre de la institución]

[Button: Enviar Respuestas]
```

### After Submission

When the user clicks "Enviar Respuestas", they see:

```
✅ ¡Gracias por completar el cuestionario!

Resumen de sus respuestas:
───────────────────────────

Fecha y Hora: 2025-10-18 21:39:15
Especialidad: Radiología
Años de Experiencia: 5
Frecuencia Uso Imágenes: Diariamente
Uso de IA: Sí
Tipos de Imágenes: Rayos X, Tomografía Computarizada (CT), Resonancia Magnética (MRI)
Importancia IA (1-10): 8
Desafíos: Tiempo limitado para análisis detallado, necesidad de segunda opinión
Comodidad con Tecnología: Muy cómodo
Interés en Capacitación: Sí, definitivamente
Institución: Hospital General

[Button: 💾 Descargar respuestas como CSV]
```

### Footer
```
─────────────────────────────────────────────────────────────
Aplicación desarrollada con Streamlit para la recopilación 
de datos de profesionales de la salud
```

## Key Features

✨ **Interactive Form**: All 10 questions with appropriate input types
📊 **Real-time Validation**: Required fields are checked before submission
💾 **CSV Export**: Download button appears after submission
🎨 **Professional Design**: Clean, medical-themed interface with emojis
📱 **Responsive**: Works on desktop, tablet, and mobile
⚡ **Fast**: Instant form submission and response display

## User Experience Flow

1. User opens the app → Sees welcome message and form
2. User fills out the 10 questions → Form validates inputs
3. User clicks "Enviar Respuestas" → Validation occurs
4. If valid → Success message + summary + download button
5. If invalid → Error message with specific field requirements
6. User downloads CSV → Gets timestamped file with all responses

## Technical Details

- **Framework**: Streamlit
- **Port**: 8501
- **Response Format**: CSV with UTF-8 encoding
- **Timestamp Format**: YYYY-MM-DD HH:MM:SS
- **File Naming**: respuestas_YYYYMMDD_HHMMSS.csv

## Accessibility

- Clear labels for all inputs
- Logical tab order
- Keyboard navigation support
- Error messages are descriptive
- Color contrast follows WCAG guidelines

---

**Note**: This is a textual representation. The actual Streamlit app has:
- Beautiful widgets with Streamlit's native styling
- Smooth animations and transitions
- Professional color scheme (blue/white/gray)
- Responsive layout that adapts to screen size
