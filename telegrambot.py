from google import genai
from google.genai import types
import numpy as np
from teleapi.httpx_transport import httpx_teleapi_factory
from teleapi.teleapi import Update, Teleapi
from dotenv import load_dotenv
import os
import time
import requests
import io
from collections import deque
import threading
import queue

load_dotenv()

# --- CONFIGURACIÓN ---
GOOGLE_KEY = os.getenv("GOOGLE_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")
FILESEARCHSTORE = os.getenv("FILESEARCHSTORE")

# Configuración de tiempos
TIEMPO_INACTIVIDAD_MAX = 300  # 300 segundos (5 minutos) para eliminar de memoria
INTERVALO_LIMPIEZA = 10       # Revisar inactividad cada 10 segundos

client = genai.Client(api_key=GOOGLE_KEY)

# --- VOCES DISPONIBLES ---
voces = {
    "es": "es-ES-AlvaroNeural",
    "en": "en-US-GuyNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural"
}

voces_largo = {
    "es": "ESPAÑOL", "en": "INGLÉS", "fr": "FRANCÉS", "de": "ALEMÁN", "it": "ITALIANO"
}

# --- FUNCIONES DE LA IA ---

def detectar_idioma(mensaje: str) -> str:
    """
    Función para detectar el idioma con los primeros 50 caracteres del mensaje utilizando el modelo gemini-2.5-flash-lite.
    
    :param mensaje: Mensaje de entrada
    :type mensaje: str
    :return: Idioma detectado en código ISO 639-1
    :rtype: str
    """

    try:
        prompt = f"Detecta el idioma y devuelve SOLO el código ISO 639-1 (es, en, fr...): {mensaje[:50]}"
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )

        idioma = response.text.strip().lower()
    
    except:

        idioma = "es"
    
    return idioma

def response_gemini_consulta_documentos(mensaje: str, historial_previo="") -> str:
    """
    Función para generar la respuesta con gemini-2.5-flash a partir de la documentación almacenada en el File Search Store.
    
    :param mensaje: Mensaje del usuario
    :type mensaje: str
    :param historial_previo: Historial previo del chat
    :type historial_previo: str
    :return: Mensaje generado por gemini-2.5-flash
    :rtype: str
    """

    idioma = detectar_idioma(mensaje)

    prompt_completo = f"""
    Respondes exclusivamente en {voces_largo.get(idioma,"ESPAÑOL")}.

    HISTORIAL DE LA CONVERSACIÓN:
    {historial_previo}
    
    NUEVA CONSULTA DEL USUARIO:
    {mensaje}'"""
    
    config = types.GenerateContentConfig(
        tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[FILESEARCHSTORE]))]
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt_completo, config=config
        )

        respuesta = response.text

    except Exception as e:
        respuesta = f"Lo siento, hubo un error procesando tu solicitud: {e}"
    
    return respuesta

def response_openweb_transcriptor(audio: bytes) -> str:
    """
    Transcripción de audio a texto usando el modelo base de Whisper.
    
    :param audio: Audio de entrada en bytes
    :type audio: bytes
    :return: Audio transcrito a texto
    :rtype: str
    """

    url = f"{BASE_URL}/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    files = {'file': ("audio.mp3", audio, "audio/mpeg")}

    try:
        res = requests.post(url, headers=headers, data={"model": "base"}, files=files)
        transcripcion = res.json()["text"]

    except:
        transcripcion = ""
    
    return transcripcion

def response_openweb_lectura(mensaje: str) -> bytes:
    """
    Conversión de texto a audio utilizando Azure AI Speech
    
    :param mensaje: Mensaje de entrada
    :type mensaje: str
    :return: Mensaje convertido a audio
    :rtype: bytes
    """

    idioma = detectar_idioma(mensaje)
    url = f"{BASE_URL}/api/v1/audio/speech"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}
    data = {"model": "tts-1", "input": mensaje, "voice": voces.get(idioma, "es-ES-AlvaroNeural")}

    try:
        audio = requests.post(url, headers=headers, json=data).content
    except: 
        audio = None
    
    return audio

def obt_audio(bot_instance: Teleapi, file_id: str) -> bytes:
    """
    Descargar el audio del mensaje de entrada
    
    :param bot_instance: Instancia del bot
    :type bot_instance: Teleapi
    :param file_id: Identificador del audio
    :type file_id: str
    :return: Audio en bytes
    :rtype: bytes
    """

    archivo = bot_instance.getFile(file_id=file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{archivo.file_path}"
    return requests.get(url).content

def adecuar_respuesta(respuesta: str) -> list:
    """
    Adecuar la respuesta generada a las limitaciones de Telegram
    
    :param respuesta: Respuesta generada por el modelo
    :type respuesta: str
    :return: Respuesta en formato lista donde la longitud de sus elementos es menor a 4000 caracteres
    :rtype: list
    """

    if len(respuesta) < 4000: 
        respuesta_list = [respuesta]
    else:
        respuesta_list = [respuesta[i:i+4000] for i in range(0, len(respuesta), 4000)]

    return respuesta_list

# --- CLASE GESTOR DE CHAT INDIVIDUAL ---

class GestorChat:
    """
    Gestor de un chat individual
    """

    def __init__(self: GestorChat, chat_id: str, bot_instance: Teleapi):
        """
        Constructor de GestorChat
        
        :param self: GestorChat
        :type self: GestorChat
        :param chat_id: Id del chat de telegram
        :type chat_id: str
        :param bot_instance: Instancia del bot
        :type bot_instance: Teleapi
        """

        self.chat_id = chat_id
        self.bot = bot_instance
        self.cola_mensajes = queue.Queue()
        self.ultima_actividad = time.time()
        self.memoria = deque(maxlen=6)
        self.activo = True
        
        # Cargar historial si existe
        self.cargar_historial()
        
        # Iniciar hilo de procesamiento exclusivo para este chat
        self.hilo = threading.Thread(target=self.procesar_cola, daemon=True)
        self.hilo.start()
        return

    def cargar_historial(self: GestorChat):
        """
        Carga el historial previo al iniciar una conversación que no está en memoria
        
        :param self: GestorChat
        :type self: GestorChat
        """

        ruta = f"chats/{self.chat_id}.txt"
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                for linea in f:
                    self.memoria.append(linea.strip())
        return

    def guardar_historial(self: GestorChat):
        """
        Al finalizar una conversación, guarda los mensajes en un fichero .txt en la carpeta chats
        Si no existe la carpeta chats la crea
        
        :param self: GestorChat
        :type self: GestorChat
        """

        if not os.path.exists("chats"): os.makedirs("chats")
        with open(f"chats/{self.chat_id}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(self.memoria))
        return

    def agregar_mensaje(self: GestorChat, update: Update):
        """
        Método público para meter mensajes en la cola
        
        :param self: GestorChat
        :type self: GestorChat
        :param update: Actualizacón del chat de telegram
        :type update: Update
        """

        self.cola_mensajes.put(update)
        self.ultima_actividad = time.time()
        return

    def procesar_cola(self: GestorChat):
        """
        Bucle infinito que procesa mensajes uno a uno (FIFO)
        
        :param self: GestorChat
        :type self: GestorChat
        """

        while self.activo:
            try:
                # Esperamos un mensaje. Timeout de 1s para poder revisar self.activo
                update = self.cola_mensajes.get(timeout=1)
            except queue.Empty:
                continue

            # Procesamiento del mensaje
            try:
                self.procesar_update(update)
            except Exception as e:
                print(f"Error en chat {self.chat_id}: {e}")
            finally:
                self.cola_mensajes.task_done()
                self.ultima_actividad = time.time()
        return

    def procesar_update(self: GestorChat, update: Update):
        """
        Procesar una actualización del Chat
        
        :param self: Descripción
        :type self: GestorChat
        :param update: Descripción
        :type update: Update
        """

        print(f"[{self.chat_id}] Procesando mensaje...")
        pregunta = None

        # Obtener pregunta (Texto o Audio)
        if update.message.audio or update.message.voice:
            fid = update.message.audio.file_id if update.message.audio else update.message.voice.file_id
            audio_bytes = obt_audio(self.bot, fid)
            pregunta = response_openweb_transcriptor(audio_bytes)
        elif update.message.text:
            pregunta = update.message.text

        if pregunta != None:

            # Consultar Gemini con Historial
            historial_str = "\n".join(self.memoria)
            respuesta = response_gemini_consulta_documentos(pregunta, historial_str)

            # Actualizar memoria
            self.memoria.append(f"Usuario: {pregunta}")
            self.memoria.append(f"Asistente: {respuesta}")

            # Enviar Respuestas (Texto)
            for parte in adecuar_respuesta(respuesta):
                self.bot.sendMessage(chat_id=self.chat_id, text=parte)
            
            # Enviar Audio
            audio_resp = response_openweb_lectura(respuesta)
            if audio_resp:
                self.bot.sendAudio(chat_id=self.chat_id, audio=io.BytesIO(audio_resp))
            
            print(f"[{self.chat_id}] Respuesta enviada.")

        return

    def detener(self: GestorChat):
        """
        Detiene el hilo y guarda datos
        
        :param self: Descripción
        :type self: GestorChat
        """

        self.activo = False
        self.guardar_historial()
        print(f"[{self.chat_id}] Hibernando por inactividad. Datos guardados.")
        return

# --- GESTIÓN GLOBAL ---

chats_activos = {} # Diccionario: chat_id -> Instancia de GestorChat
lock_chats = threading.Lock() # Para modificar el diccionario de forma segura

def monitor_inactividad():
    """
    Hilo en segundo plano que limpia chats viejos
    """

    while True:
        time.sleep(INTERVALO_LIMPIEZA)
        ahora = time.time()
        eliminar = []

        with lock_chats:
            for chat_id, gestor in chats_activos.items():
                # Si pasó el tiempo límite Y la cola está vacía
                if (ahora - gestor.ultima_actividad > TIEMPO_INACTIVIDAD_MAX) and gestor.cola_mensajes.empty():
                    gestor.detener() # Guarda fichero y para su hilo interno
                    eliminar.append(chat_id)
            
            for chat_id in eliminar:
                del chats_activos[chat_id]
        
        if eliminar:
            print(f"Limpieza: Se han eliminado {len(eliminar)} chats inactivos de la memoria RAM.")
    
    return

def main():
    print("Iniciando Bot Multi-Hilo...")
    if not os.path.exists("chats"): os.makedirs("chats")

    bot = httpx_teleapi_factory(BOT_TOKEN, timeout=60)
    
    # Arrancar el monitor de limpieza
    hilo_limpieza = threading.Thread(target=monitor_inactividad, daemon=True)
    hilo_limpieza.start()

    offset = 0
    while True:
        try:
            updates = bot.getUpdates(offset=offset, timeout=10)
            
            for update in updates:
                if not update.message: 
                    offset = update.update_id + 1
                    continue

                chat_id = update.message.chat.id
                
                with lock_chats:
                    # Si el chat no está en memoria, lo "despertamos" (creamos/cargamos)
                    if chat_id not in chats_activos:
                        print(f"[{chat_id}] Nueva sesión (o recuperada de disco).")
                        chats_activos[chat_id] = GestorChat(chat_id, bot)
                    
                    # Encolamos el mensaje en el gestor específico de ese chat
                    chats_activos[chat_id].agregar_mensaje(update)

                offset = update.update_id + 1

        except Exception as e:
            print(f"Error en polling principal: {e}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("Cerrando...")
            with lock_chats:
                for gestor in chats_activos.values():
                    gestor.detener()
            break

if __name__ == "__main__":
    main()