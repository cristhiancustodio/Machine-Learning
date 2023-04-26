#pip install transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
from transformers import AutoTokenizer, MarianMTModel,MarianTokenizer
from colab_utils import getAudio
## librerías para el procesamiento del audio
import librosa
import numpy as np
## pytorch para la conversión de voz a texto
import torch

def convertir_audio_texto():
    #detecta voz en español y lo pasa a texto con este modelo entrado
    modelo_espanol = "facebook/wav2vec2-large-xlsr-53-spanish"
    w2v2_processor = Wav2Vec2Processor.from_pretrained(modelo_espanol)
    w2v2 = Wav2Vec2ForCTC.from_pretrained(modelo_espanol)
    audio, sr = getAudio()
    audio_float = audio.astype(np.float32)
    audio_16k = librosa.resample(audio_float, sr, 16000)
    entrada = w2v2_processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_values
    probabilidades = w2v2(entrada).logits
    predicciones = torch.argmax(probabilidades, dim=-1)
    texto_generado = w2v2_processor.decode(predicciones[0])
    return texto_generado

def traductor(src,trg,texto):
  model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
  model = MarianMTModel.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  sample_text = texto
  batch = tokenizer([sample_text], return_tensors="pt")

  generated_ids = model.generate(**batch)
  text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return text

def espanol_ingles(texto) : 
  src = "es"  # source language
  trg = "en"  # target language
  texto = traductor(src,trg, texto)
  return texto
  

def ingles_espanol(texto):
  src = "en"  # source language
  trg = "es"  # target language
  texto = traductor(src,trg, texto)
  return texto

def generar_conversacion():
    #nuestro modelo que genera conversacion a partir de texto
    conversador = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(conversador)

    texto = convertir_audio_texto() # obtenemos nuestro texto generado de nuestra voz
    print(f"->Yo : {texto}")
    
    texto = espanol_ingles(texto)
    entradaBlender = tokenizer([texto], return_tensors='pt')
    blender = AutoModelForSeq2SeqLM.from_pretrained(conversador)
    ids_respuesta = blender.generate(**entradaBlender)
    respuesta = tokenizer.batch_decode(ids_respuesta)
    respuesta = respuesta[0].replace('<s>','').replace('</s>','')
    respuesta = ingles_espanol(respuesta)
    print(f'->ChatBot : {respuesta}')
contador = 5
i=1
while i<=5:

  generar_conversacion()
  i+=1