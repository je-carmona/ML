**TAREA 1: Ejercicio de rimas**

_Enlace Google Colab:_ https://colab.research.google.com/drive/1phRzdGjHBnLQvY0ZUbRSVYL8eFfn9vgN?usp=sharing

**1.1. Codigo para encontrar las palabras con las letras dadas:**

      # Libreria que se usa para expresiones regulares
      import re 
      
      archivo = "5000_palabras_comunes.txt"
      
      # ¿Cuáles son las letras con las que quiere buscar las palabras?
      # Mantener el símbolo $ al final de las letras y escribir todo en comillas
      # Ejm "ión$"
      terminacion = "das$"  
      
      # Función para leer las lineas del archivo
      with open(archivo, "r", encoding="utf-8") as f:
        palabras = f.read().splitlines()  # Lee todas las líneas y las convierte en una lista
      
      # Función para filtrar palabras que terminan con la terminación escogida
      palabras_filtradas = [palabra for palabra in palabras if re.search(terminacion, palabra)]
      
      # Mostrar las palabras encontradas
      print("Palabras que terminan con 'das':")
      print(palabras_filtradas)
      print("")
      print("Desarrollado por: J.E. Carmona-Álvarez")


**RESULTADO**

Palabras que terminan con 'das':
['gracioso', 'esposo', 'hermoso', 'maravilloso', 'peligroso', 'grandioso', 'nervioso', 'orgulloso', 'sospechoso', 'famoso', 'poderoso', 'precioso', 'oso', 'mentiroso', 'curioso', 'asqueroso', 'asombroso', 'fabuloso', 'celoso', 'delicioso', 'furioso', 'generoso']

Desarrollado por: J.E. Carmona-Álvarez

**1.2. Aplicacion para subir archivo y buscar las palabras con la terminacion deseada:**

    import re
    #permite usar los archivos en el entorno de Colab
    from google.colab import files
    
    # Funcion para subir el archivo
    print("Por favor suba un archivo en formato .txt con una palabra por fila")
    archivo_subido = files.upload()
    nombre_archivo = list(archivo_subido.keys())[0]
    
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        palabras = f.read().splitlines()  
    
    # Bucle para permitir intentos hasta encontrar palabras con la terminacion escogida
    while True:
        # Función para qeu el usuario vuelva a escribir la terminación
        terminacion = input("\nIngresa las letras con las que deben terminar las palabras (o escribe 'salir' para terminar): ")
    
        # Opción para salir del programa
        if terminacion.lower() == "salir":
            print("Programa terminado.")
            break
    
        # Expresión regular para encontrar palabras que terminan con la entrada del usuario
        patron = terminacion + "$"  # El signo $ indica "al final de la palabra"
        palabras_filtradas = [palabra for palabra in palabras if re.search(patron, palabra)]
    
        # Función para mostrar los resultados
        if palabras_filtradas:
            print("\nPalabras que terminan con '{}':".format(terminacion))
            for palabra in palabras_filtradas:
                print(palabra)
            break  
        else:
            print("\nNo se encontraron palabras que terminen con '{}'. Intenta con otra terminación.".format(terminacion))
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")


**RESULTADOS**

Por favor suba un archivo en formato .txt con una palabra por fila
5000_palabras_comunes.txt
5000_palabras_comunes.txt(text/plain) - 38164 bytes, last modified: 23/3/2025 - 100% done
Saving 5000_palabras_comunes.txt to 5000_palabras_comunes.txt

Ingresa las letras con las que deben terminar las palabras (o escribe 'salir' para terminar): oso

Palabras que terminan con 'oso':
gracioso
esposo
hermoso
maravilloso
peligroso
grandioso
nervioso
orgulloso
sospechoso
famoso
poderoso
precioso
oso
mentiroso
curioso
asqueroso
asombroso
fabuloso
celoso
delicioso
furioso
generoso

Desarrollado por: J.E. Carmona-Álvarez
