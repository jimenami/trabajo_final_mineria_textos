import os
import zipfile
import gradio as gr
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer

if not os.path.isdir('./modelo_guardado') and os.path.isfile('./modelo_guardado.zip'):
    with zipfile.ZipFile('./modelo_guardado.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

sentiment_pipeline = pipeline('text-classification', model='./modelo_guardado')

flan_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
flan_model     = T5ForConditionalGeneration.from_pretrained(
    'google/flan-t5-small',
    low_cpu_mem_usage=True,
)

def generate_explanation(prompt, max_new_tokens=80):
    inputs  = flan_tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True)
    outputs = flan_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def analizar_sentimiento(texto):
    res        = sentiment_pipeline(texto)[0]
    label      = res['label']
    confidence = res['score']

    prompt = (
        f'The following tweet has been classified as {label} sentiment '
        f'with {confidence:.0%} confidence.\n'
        f'Tweet: "{texto}"\n'
        f'Explain in one or two sentences why this tweet expresses {label} sentiment:'
    )
    explanation = generate_explanation(prompt)

    return label, round(confidence, 4), explanation

demo = gr.Interface(
    fn=analizar_sentimiento,
    inputs=gr.Textbox(placeholder='Escribe un tweet...', label='Tweet'),
    outputs=[
        gr.Label(label='Sentimiento'),
        gr.Number(label='Confianza'),
        gr.Textbox(label='Explicación (Flan-T5)'),
    ],
    title='Análisis de Sentimiento en Tweets',
    examples=[
        ['I loved Eurovision 2026. It is the best celebration ever!'],
        ['I am so disappointed and frustrated about how Trump is dealing with the immigration problem. I dont think he is treating these people like humans but animals'],
        ['The Met Gala was okay, nothing memorable about it though. Not worth the hype in my opinion. And neither the money spent on it.'],
    ],
)

demo.launch()